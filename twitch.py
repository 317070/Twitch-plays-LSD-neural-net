import numpy as np
import subprocess as sp
import time
import signal
import socket
import sys
import re
import fcntl, os
import errno
import threading
from models.default import TWITCH_STREAM_KEY, TWITCH_OAUTH, TWITCH_USERNAME


class TwitchChatStream(object):

    user = ""
    oauth = ""
    s = None

    @staticmethod
    def twitch_login_status(data):
        if not re.match(r'^:(testserver\.local|tmi\.twitch\.tv) NOTICE \* :Login unsuccessful\r\n$', data):
            return True
        else:
            return False

    def __init__(self, user=TWITCH_USERNAME, oauth=TWITCH_OAUTH):
        self.user = user
        self.oauth= oauth
        self.last_sent_time = time.time()
        self.twitch_connect()

    def twitch_connect(self):
        print("Connecting to twitch.tv")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #s.setblocking(False)
        #s.settimeout(1.0)
        connect_host = "irc.twitch.tv"
        connect_port = 6667
        try:
            s.connect((connect_host, connect_port))
        except:
            pass #expected, because is non-blocking socket
            sys.exit()
        print("Connected to twitch")
        print("Sending our details to twitch...")
        #s.send('USER %s\r\n' % self.user)
        s.send('PASS %s\r\n' % self.oauth)
        s.send('NICK %s\r\n' % self.user)

        if not TwitchChatStream.twitch_login_status(s.recv(1024)):
            print("... and they didn't accept our details")
            sys.exit()
        else:
            print("... they accepted our details")
            print("Connected to twitch.tv!")
            fcntl.fcntl(s, fcntl.F_SETFL, os.O_NONBLOCK)
            if self.s is not None:
                self.s.close()
            self.s = s
            s.send('JOIN #%s\r\n' % self.user)

    def send(self, message):
        if time.time() - self.last_sent_time > 5:
            if len(message) > 0:
                try:
                    self.s.send(message + "\n")
                    #print message
                finally:
                    self.last_sent_time = time.time()

    @staticmethod
    def check_has_ping(data):
        return re.match(r'^PING :(tmi\.twitch\.tv|\.testserver\.local)$', data)

    def send_pong(self):
        self.send("PONG")

    def send_chat_message(self, message):
        self.send("PRIVMSG #{0} :{1}".format(self.user, message))

    @staticmethod
    def check_has_message(data):
        return re.match(r'^:[a-zA-Z0-9_]+\![a-zA-Z0-9_]+@[a-zA-Z0-9_]+(\.tmi\.twitch\.tv|\.testserver\.local) PRIVMSG #[a-zA-Z0-9_]+ :.+$', data)

    def parse_message(self, data):
        if TwitchChatStream.check_has_ping(data):
            self.send_pong()
        if TwitchChatStream.check_has_message(data):
            return {
                'channel': re.findall(r'^:.+\![a-zA-Z0-9_]+@[a-zA-Z0-9_]+.+ PRIVMSG (.*?) :', data)[0],
                'username': re.findall(r'^:([a-zA-Z0-9_]+)\!', data)[0],
                'message': re.findall(r'PRIVMSG #[a-zA-Z0-9_]+ :(.+)', data)[0].decode('utf8')
            }
        else:
            return None

    def twitch_recieve_messages(self, amount=1024):
        result = []
        while True: #process the complete buffer, until no data is left no more
            try:
                msg = self.s.recv(4096) # NON-BLOCKING RECEIVE!
                if msg:
                    pass
                    #print msg
            except socket.error, e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                    #print 'No more data available'
                    return result
                else:
                    # a "real" error occurred
                    import traceback
                    import sys
                    print(traceback.format_exc())

                    text_file = open("log1.txt", "w")
                    text_file.write(traceback.format_exc())
                    text_file.close()

                    self.twitch_connect()
                    return result
            except:
                import traceback
                import sys
                print(traceback.format_exc())

                text_file = open("log2.txt", "w")
                text_file.write(traceback.format_exc())
                text_file.close()

                self.twitch_connect()
                return result
            else:
                rec = [self.parse_message(line) for line in filter(None, msg.split('\r\n'))]
                rec = [r for r in rec if r] #remove None's
                result.extend(rec)



class TwitchOutputStream(object):
    def __init__(self, width=640, height=480, fps=30., twitch_stream_key=TWITCH_STREAM_KEY):
        self.twitch_stream_key = twitch_stream_key
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe = None
        self.reset()

    def reset(self):
        if self.pipe is not None:
            try:
                self.pipe.send_signal(signal.SIGINT)
            except OSError:
                pass


        FFMPEG_BIN = "avconv" # on Linux and Mac OS
        command = [ FFMPEG_BIN,
                        '-y', # overwrite
                        #'-re',# native frame-rate
                '-analyzeduration','1',
                '-f', 'rawvideo',
                '-r', '%d'%self.fps,
                '-vcodec','rawvideo',
                '-s', '%dx%d'%(self.width, self.height), # size of one frame
                '-pix_fmt', 'rgb24',
                #'-an', # Tells FFMPEG not to expect any audio
                #'-r', '%d'%fps, # frames per second
                '-i', '-', # The input comes from a pipe

                #'-i', 'silence.mp3', #otherwise, there is no sound in the output, which twitch doesn't like
                #'-ar', '48000', '-ac', '2', '-f', 's16le', '-i', '/dev/zero', #silence alternative, works forever. (Memory hole?)
                '-i','http://stream1.radiostyle.ru:8001/tunguska',
                        #'-filter_complex', '[0:1][1:0]amix=inputs=2:duration=first[all_audio]'
                #'-vcodec', 'libx264',
                '-vcodec', 'libx264', '-r', '%d'%self.fps, '-b:v', '3000k', '-s', '%dx%d'%(self.width, self.height),
                                '-preset', 'veryfast',#'-tune', 'film',
                                '-crf','23',
                                '-pix_fmt', 'yuv420p', #'-force_key_frames', r'expr:gte(t,n_forced*2)',
                                '-minrate', '3000k', '-maxrate', '3000k', '-bufsize', '12000k',
                                '-g','60',
                                '-keyint_min','1',
                #'-filter:v "setpts=0.25*PTS"'
                '-vsync','passthrough',
                #'-acodec', 'libmp3lame', '-ar', '44100', '-b', '160k',
                #               '-bufsize', '8192k', '-ac', '2',
                #'-acodec', 'aac', '-strict', 'normal', '-ab', '128k',
                #'-vcodec', 'libx264', '-s', '%dx%d'%(width, height), '-preset', 'libx264-fast',
                #'my_output_videofile2.avi'
                '-map', '0:v', '-map', '1:a', #use only video from first input and only audio from second
                '-threads', '1',
                '-f', 'flv', 'rtmp://live-fra.twitch.tv/app/%s'%self.twitch_stream_key
                ]

        fh = open("/dev/null", "w")
        #fh = None #uncomment for viewing ffmpeg output
        self.pipe = sp.Popen( command, stdin=sp.PIPE, stderr=fh, stdout=fh)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        #sigint so avconv can clean up the stream nicely
        self.pipe.send_signal(signal.SIGINT)
        #waiting doesn't work because reasons
        #self.pipe.wait()

    #send frame of shape (height, width, 3) with values between 0 and 1
    def send_frame(self, frame):
        if self.pipe.poll():
            self.reset()
        assert frame.shape == (self.height, self.width, 3), "Frame has the wrong shape %s iso %s"%(frame.shape,(self.height, self.width, 3))
        frame = np.clip(255*frame, 0, 255).astype('uint8')
        self.pipe.stdin.write( frame.tostring() )


"""
    This stream makes sure a steady framerate is kept by repeating the last frame when needed
"""
class TwitchOutputStreamRepeater(TwitchOutputStream):
    def __init__(self, *args, **kwargs):
        super(TwitchOutputStreamRepeater, self).__init__(*args, **kwargs)
        self.lastframe = np.ones((self.height, self.width, 3))
        self.send_me_last_frame_again()

    def send_me_last_frame_again(self):
        try:
            super(TwitchOutputStreamRepeater, self).send_frame(self.lastframe)
        except IOError:
            pass #stream has been closed. This function is still called once when that happens.
        else:
            #send the next frame at the appropriate time
            threading.Timer(1./self.fps, self.send_me_last_frame_again).start()

    def send_frame(self, frame):
        self.lastframe = frame

if __name__=="__main__":
    """
    i=0
    with TwitchOutputStream(640, 480, 30) as stream:
        for _ in xrange(3000):
            t = time.time()
            #image_array = np.random.randint(255, size=(height, width, 3)).astype('uint8')
            stream.send_frame( (i%255)/255.0 *np.ones((480, 640, 3)) )
            #print "frame", i
            i+=15
    #"""
    """
    chatstream = TwitchChatStream()
    while True:
        received = chatstream.twitch_recieve_messages()
        if received:
            print received
        import random
        #chatstream.send_chat_message("Hello %f"%random.random())
        time.sleep(1)
    #"""
