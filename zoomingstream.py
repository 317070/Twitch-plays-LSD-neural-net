from twitch import TwitchOutputStream
import numpy as np
import threading
import time
import collections
from skimage import transform
import multiprocessing
from functools import partial
import Queue
from PIL import Image, ImageFont, ImageDraw
from collections import deque
import sys

font = ImageFont.truetype(font="Helvetica.ttf", size=20) #load once
THE_TEXT_AS_IMAGE = None


def put_current_text_on_image(image):
    if THE_TEXT_AS_IMAGE is not None:
        image[4:6+THE_TEXT_AS_IMAGE.shape[0],4:6+THE_TEXT_AS_IMAGE.shape[1],:] = 0
        image[5:5+THE_TEXT_AS_IMAGE.shape[0],5:5+THE_TEXT_AS_IMAGE.shape[1],:] = THE_TEXT_AS_IMAGE
    return image


ZOOMING_FRAME = None
ZOOMING_FRAME_INDEX = None
LAST_ZOOMING_FRAME = None
LAST_ZOOMING_FRAME_INDEX = None

ZOOM_SPEED_PER_FRAME = None

ZOOMING_OUTPUT_SHAPE = None

def fast_warp(img, matrix, output_shape=(53,53), mode='reflect', order=1, cval=0):
    """
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    #matrix = np.linalg.inv(matrix)
    img_wf = np.empty((output_shape[0], output_shape[1], 3), dtype='float32')
    for k in xrange(3):
        img_wf[..., k] = transform._warps_cy._warp_fast(img[..., k], matrix, output_shape=output_shape, mode=mode, order=order, cval=cval)
    return img_wf


def zoom_newshape(id):
    start = time.time()
    try:
        global ZOOMING_FRAME, ZOOMING_FRAME_INDEX, LAST_ZOOMING_FRAME
        global LAST_ZOOMING_FRAME_INDEX, ZOOMING_OUTPUT_SHAPE, ZOOM_SPEED_PER_FRAME

        output_shape = ZOOMING_OUTPUT_SHAPE
        frame_1 = LAST_ZOOMING_FRAME
        id_1 = LAST_ZOOMING_FRAME_INDEX
        frame_2 = ZOOMING_FRAME
        id_2 = ZOOMING_FRAME_INDEX


        frame_cnt_1 = float(id - id_1)

        zoomlevel = ZOOM_SPEED_PER_FRAME**frame_cnt_1

        extra_zoom = np.min([ 1.0*output_shape[0]/frame_1.shape[0], 1.0*output_shape[1]/frame_1.shape[1]] )

        shift_1_y, shift_1_x = 1.0*np.array(frame_1.shape[:2])/2.0
        tf_shift = transform.SimilarityTransform(translation=[shift_1_x, shift_1_y])

        tf_rotate = transform.SimilarityTransform(scale=1.0/(zoomlevel*extra_zoom),
                                                  rotation=-np.deg2rad(0.01 * frame_cnt_1))

        shift_2_y, shift_2_x = 1.0*np.array(output_shape)/2.0
        tf_shift_inv = transform.SimilarityTransform(translation=[-shift_2_x, -shift_2_y])

        #print "a:", time.time()-start
        frame_rotated_1 = fast_warp(frame_1, (tf_shift_inv + (tf_rotate + tf_shift)).params,
                                       order=1, mode="nearest",
                                       output_shape=output_shape, #frame.shape[:2],
                                       )
        #print "b:", time.time()-start
        frame_cnt_2 = float(id_2 - id)
        zoomlevel = ZOOM_SPEED_PER_FRAME ** (-frame_cnt_2)

        tf_rotate = transform.SimilarityTransform(scale=1.0/(zoomlevel*extra_zoom),
                                                  rotation=np.deg2rad(0.01 * frame_cnt_2 ))
        #print "c:", time.time()-start
        frame_rotated_2 = fast_warp(frame_2, (tf_shift_inv + (tf_rotate + tf_shift)).params,
                                       output_shape=output_shape, #frame.shape[:2],
                                       order=1, mode="constant", cval=np.nan,
                                       )

        #print "d:", time.time()-start
        frame_rotated = frame_cnt_2 / float(id_2-id_1) * frame_rotated_1 + frame_cnt_1 / float(id_2-id_1) * frame_rotated_2
        mask = np.isnan(frame_rotated_2)
        #print "e:", time.time()-start
        frame_rotated[mask] = frame_rotated_1[mask]
        #print "f:", time.time()-start
        #print frame_cnt_1 / float(id_2-id_1), frame_cnt_2 / float(id_2-id_1)

        frame_rotated = put_current_text_on_image(frame_rotated)
        #print "g:", time.time()-start

        return id, frame_rotated
    except:
        import traceback
        import sys
        print(traceback.format_exc())
        sys.exit(1)

def zoom(frame, frames_to_skip):
    global ZOOM_SPEED_PER_FRAME

    zoomlevel = ZOOM_SPEED_PER_FRAME ** frames_to_skip

    output_shape = frame.shape[:2]
    shift_1_y, shift_1_x = 1.0*np.array(frame.shape[:2])/2.0
    tf_shift = transform.SimilarityTransform(translation=[shift_1_x, shift_1_y])

    tf_rotate = transform.SimilarityTransform(scale=1.0/zoomlevel,
                                              rotation=-np.deg2rad(0.01 * frames_to_skip ))

    shift_2_y, shift_2_x = 1.0*np.array(output_shape)/2.0
    tf_shift_inv = transform.SimilarityTransform(translation=[-shift_2_x, -shift_2_y])

    frame = np.clip(frame,0.0,1.0)

    frame_rotated = fast_warp(frame, (tf_shift_inv + (tf_rotate + tf_shift)).params,
                                   order=3, mode="nearest",
                                   output_shape=output_shape
                                   )
    return frame_rotated


class ZoomingStream(TwitchOutputStream):
    def __init__(self, zoomspeed=1.01, estimated_input_fps=0.5, *args, **kwargs):
        global ZOOM_SPEED_PER_FRAME, ZOOMING_OUTPUT_SHAPE
        super(ZoomingStream, self).__init__(*args, **kwargs)
        ZOOMING_OUTPUT_SHAPE = (self.height, self.width)
        ZOOM_SPEED_PER_FRAME = zoomspeed ** (1. / self.fps)

        self.framenumber = 0
        self.zoomspeed = zoomspeed
        #self.p = multiprocessing.Pool(multiprocessing.cpu_count() - 4, maxtasksperchild=5) #leaf room for main, 2xavconv, other
        self.t = None
        self.estimated_input_fps = 1./10 #self.fps/3 # needed to factor in zoom speed
        self.q = Queue.PriorityQueue() #collections.deque()
        self.last_frame_time = None
        self.next_send_time = None
        self.last_frame = np.zeros((self.height, self.width, 3))
        t = threading.Timer(2.0/self.estimated_input_fps, self.send_me_last_frame_again)
        t.daemon = True
        t.start()

        self.killcount = deque(maxlen=100)#kill yourself after 100 failed frames within 10 seconds: 33% of frames
        self.harakiri = False

    # The program seems to stop generating frames after 2-10 hours
    # exit when that happens! Don't just hang there.
    def increase_kill_count(self):
        self.killcount.append(time.time())
        if len(self.killcount) == self.killcount.maxlen:
            # if the first element was less than 10 seconds ago
            if self.killcount[0] > time.time() - 10:
                self.harakiri = True



    def send_me_last_frame_again(self):
        start_time = time.time()
        try:
            frame = self.q.get_nowait()
            #print frame[0] #frame id number
            frame = frame[1]
        except IndexError:
            frame = self.last_frame
            self.increase_kill_count()
            print "NO FRAMES LEFT!"
        except Queue.Empty:
            frame = self.last_frame
            self.increase_kill_count()
            print "NO FRAMES LEFT!"
        else:
            self.last_frame = frame

        try:
            super(ZoomingStream, self).send_frame(frame)
        except IOError:
            pass #stream has been closed. This function is still called once when that happens.


        if self.harakiri:
            return #don't start a new thread

        #send the next frame at the appropriate time
        if self.next_send_time is None:
            threading.Timer(1./self.fps, self.send_me_last_frame_again).start()
            self.next_send_time = start_time + 1./self.fps
        else:
            self.next_send_time += 1./self.fps
            next_event_time = self.next_send_time - start_time
            if next_event_time>0:
                threading.Timer(next_event_time, self.send_me_last_frame_again).start()
            else:
                # not allowed for recursion problems :-( (maximum recursion depth)
                # self.send_me_last_frame_again()
                t = threading.Thread(target=self.send_me_last_frame_again).start()




    def zoom_frames_and_add_to_queue(self, frame, start_frame_id, text):
        # Multiprocess this line
        #for i in xrange(frames_before_next_image):
        #    self.q.append(ZoomingStream.zoom(frame, self.zoomspeed**(1.0*i/self.fps), order=3))
        global ZOOMING_FRAME, ZOOMING_FRAME_INDEX
        global LAST_ZOOMING_FRAME, LAST_ZOOMING_FRAME_INDEX
        global THE_TEXT_AS_IMAGE

        LAST_ZOOMING_FRAME = ZOOMING_FRAME
        LAST_ZOOMING_FRAME_INDEX = ZOOMING_FRAME_INDEX

        ZOOMING_FRAME = np.clip(frame, 0.0, 1.0)
        ZOOMING_FRAME_INDEX = start_frame_id

        if LAST_ZOOMING_FRAME is None:
            return
        """ DO NOT USE PARTIALS AS THEY CAUSE MEMORY LEAKS!!!"""
        #zoom_frame = partial(zoom, output_shape=output_shape)
        """
        result = map(zoom_newshape, [self.zoomspeed**(1.0*i/self.fps) for i in xrange(frames_before_next_image)])

        """
        thetext = np.asarray(font.getmask(text, mode="1")) #mode: no anti-aliasing.
        textsize = font.getsize(text)



        if thetext.shape[0]>0:
            #print thetext.shape, textsize
            #because fuck you, that's why!
            thetext = thetext[:thetext.shape[0] // textsize[0] * textsize[0]]

            thetext = thetext.reshape((-1,textsize[0])) #needs to be -1 on some machines, don't know why
            THE_TEXT_AS_IMAGE = np.dstack([thetext]*3)


        ids = [LAST_ZOOMING_FRAME_INDEX+i for i in xrange(ZOOMING_FRAME_INDEX - LAST_ZOOMING_FRAME_INDEX)]

        #create the pool at this point, such that global variables are shared with the new processes
        #this saves on memory

        p = multiprocessing.Pool(multiprocessing.cpu_count() - 4) #leaf room for main, 1xavconv, other
        while ids:
            result = p.map(zoom_newshape, ids[:20])
            #result = [r.copy() for r in result]
            ids = ids[20:]
            for r in result:
                self.q.put(r)
        p.close()
        p.join()
        #"""



    def send_frame(self, frame, text=""):
        if self.last_frame_time is None:
            self.last_frame_time = time.time()
        else:
            estimated_fps = 1./(time.time() - self.last_frame_time)
            self.estimated_input_fps = 0.1 * self.estimated_input_fps + 0.9 * estimated_fps
            print "estimated_fps_input", 1./estimated_fps
            self.last_frame_time = time.time()


        # now, calculate the next frames in the video, and return the next image to work on
        frames_before_next_image = int(np.ceil(self.fps / self.estimated_input_fps))

        # if buffer is full, STOP FILLING !
        while self.q.qsize() > 2*frames_before_next_image:
            self.last_frame_time = time.time()

        """
        self.zoom_frames_and_add_to_queue(frame, frames_before_next_image, output_shape= (self.height, self.width))
        """
        if self.t:
            t=None
            if not self.t.is_alive():
                pass
            self.t.join() #Wait for the previous run to finish
        print "Buffer underflow?", self.q.qsize()

        #This is best spot to kill itself. No other threads or processes at this moment.
        if self.harakiri:
            print "HARAKIRI!"
            return None
            #upon exiting, this python program is rebooted on the machine

        self.t = threading.Thread(target=self.zoom_frames_and_add_to_queue,
                         args=(frame, self.framenumber, text)
                         )
        self.t.daemon = True
        self.t.start()
        self.framenumber += frames_before_next_image
        #"""
        return zoom(frame, frames_before_next_image)




