import numpy as np
import re
from collections import deque
from twitch import TwitchChatStream
import random
import time
import exrex
import copy

EXREX_REGEX_ONE =  ("(@__username__: (Wow|Amazing|Fascinating|Incredible|Marvelous|Wonderful|AAAAAah|OMG)\. __WORD__, that's (deep|wild|trippy|dope|weird|spacy), (man|dude|brother|bro|buddy|my man|mate|homie|brah|dawg)\. (Thanks|Kudos|Props|Respect) for (writing stuff|sending ideas) to steer my trip\.)|"
                    "(@__username__: __WORD__, __WORD__, __WORD__ (EVERYWHERE|ALL AROUND|ALL OVER)\. (WaaaaAAAah|Wooooooooooow))|"
                    #"(Wow, very @__username__, such __word__, much (amazing|bazinga|space|woop)\.)" #disliked by native english people
                    )

EXREX_REGEX_TWO = ("(@__username__: One __word0__ with __word1__ coming (up next|your way)!)|"
                   "(@__username__: Yeah, let's (try|create|dream of|watch) __word0__ with a (topping|layer) of __word1__!)"
                   )

EXREX_REGEX_MORE = ("(@__username__: __words__, I'll mash them all up for ya\.)")



class ChatReader(TwitchChatStream):

    def __init__(self, *args, **kwargs):
        super(ChatReader, self).__init__(*args, **kwargs)

        #make a data structure to easily parse chat messages
        self.classes = np.load("data/classes.npy")

        ignore_list = ['the', 'of', 't', 'and', "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
                       "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by",
                       "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
                       "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
                       "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
                       "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
                       "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
                       "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
                       "want", "because", "any", "these", "give", "day", "most", "us"]

        # First, check if the complete string matches
        d = {d[0]:i for i,d in enumerate(self.classes)}
        self.full_dictionary = {}
        for str,i in d.iteritems():
            for word in str.split(','):
                word = word.lower().strip()
                if word in ignore_list:
                    continue
                if word in self.full_dictionary:
                    self.full_dictionary[word].append(i)
                else:
                    self.full_dictionary[word] = [i]


        # r'\bAND\b | \bOR\b | \bNOT\b'
        self.regexes = []

        regex_string = " | ".join([r"^%s$"%word.replace(" ",r"\ ") for word in self.full_dictionary.keys()])
        self.regexes.append((self.full_dictionary,  re.compile(regex_string, flags=re.I | re.X)))

        regex_string2 = " | ".join([r"\b%s\b"%word.replace(" ",r"\ ") for word in self.full_dictionary.keys()])


        self.dictionary = copy.deepcopy(self.full_dictionary)
        # Second, check if complete string matches a word
        for str,i in d.iteritems():
            for word in re.findall(r"[\w']+", str):
                word = word.lower()
                if word in ignore_list:
                    continue
                if word in self.dictionary:
                    self.dictionary[word].append(i)
                else:
                    self.dictionary[word] = [i]

        regex_string = " | ".join([r"^%s$"%word.replace(" ",r"\ ") for word in self.dictionary.keys()])
        self.regexes.append((self.dictionary, re.compile(regex_string, flags=re.I | re.X)))

        # This was deemed too sensitive by a lot of people
        """
        # third, check if complete thing is in string
        self.regexes.append((self.full_dictionary, re.compile(regex_string2, flags=re.I | re.X)))

        # fourth, check if words are found in the string
        regex_string = " | ".join([r"\b%s\b"%word.replace(" ",r"\ ") for word in self.dictionary.keys()])
        self.regexes.append((self.dictionary, re.compile(regex_string, flags=re.I | re.X)))
        """

        self.currentwords = deque(maxlen=1)
        self.current_features = [random.randint(0,999)]
        self.last_read_time = 0
        self.hold_subject_seconds = 60
        self.display_string = ""
        self.message_queue = deque(maxlen=100)

        self.max_features = 2


    @staticmethod
    def get_cheesy_chat_message(username, words):
        if len(words)==1:
            return exrex.getone(EXREX_REGEX_ONE).replace("__username__", username)\
                                                .replace("__USERNAME__", username.capitalize())\
                                                .replace("__word__",words[0])\
                                                .replace("__WORD__",words[0].capitalize())
        elif len(words)==2:
            return exrex.getone(EXREX_REGEX_TWO).replace("__username__", username)\
                                                .replace("__USERNAME__", username.capitalize())\
                                                .replace("__word0__",words[0])\
                                                .replace("__WORD0__",words[0].capitalize())\
                                                .replace("__word1__",words[1])\
                                                .replace("__WORD1__",words[1].capitalize())
        else:
            wordstring = " & ".join(words)
            return exrex.getone(EXREX_REGEX_MORE).replace("__username__", username)\
                                                .replace("__USERNAME__", username.capitalize())\
                                                .replace("__words__",wordstring)\
                                                .replace("__WORDS__",wordstring.capitalize())






    def process_the_chat(self):
        display_string = self.display_string
        features = self.current_features

        messages = self.twitch_recieve_messages() #you always need to check for ping messages

        self.message_queue.extend(messages)
        # [{'username': '317070', 'message': u'test again', 'channel': '#317070'}]
        if time.time() - self.last_read_time < self.hold_subject_seconds:
            return features, display_string
        try:
            messages = list(self.message_queue)
            random.shuffle(messages)
            self.message_queue.clear()

            #spaghetti code warning ahead
            found = False
            for message in messages:

                queries = filter(None, [w.strip() for w in message['message'].split('+')])
                total_features = []
                total_correct_terms = []
                for query in queries:

                    for dictionary, regex in self.regexes:

                        hits = regex.findall(query)

                        if hits:
                            print hits
                            correct_terms = []
                            features = []
                            words_used = []
                            for h in set(hits):
                                word = h.lower()

                                if any(current_feature in dictionary[word] for current_feature in self.current_features):
                                    continue
                                feature = random.choice(dictionary[word])
                                features.append(feature)

                                correct_term = ""
                                #print self.classes[feature][0].lower()
                                for term in self.classes[feature][0].lower().split(','):
                                    if word in term:
                                        correct_term = term.strip()
                                        break
                                correct_terms.append(correct_term)
                                words_used.append(word)

                            if len(features)==0:
                                continue

                            #We want at most (max_features) features
                            #print features, correct_terms
                            features, correct_terms, words_used = zip(*random.sample(zip(features, correct_terms, words_used), min(len(features), self.max_features)))

                            if len(words_used)>1:
                                if message['message'].index(words_used[1]) < message['message'].index(words_used[0]):
                                    features = features.reverse()
                                    correct_terms = correct_terms.reverse()
                                    words_used = words_used.reverse()

                            #print regex.pattern
                            total_features.extend(features)
                            total_correct_terms.extend(correct_terms)
                            break


                if len(total_features)==0:
                    continue

                total_features = total_features[:2]
                total_correct_terms = total_correct_terms[:2]

                username = message['username']

                if len(total_features)==1:
                    display_string = "@"+username+": "+total_correct_terms[0]
                else:
                    display_string = " & ".join(total_correct_terms)

                chat_message = ChatReader.get_cheesy_chat_message(username, total_correct_terms)

                self.send_chat_message(chat_message)
                self.last_read_time = time.time()
                found = True
                break


            if not found:
                return self.current_features, self.display_string


            self.current_features = total_features
            self.display_string = display_string

            print [self.classes[feature][0] for feature in total_features]
            return total_features, display_string
        except:
            # let the chat users not crash the entire program
            self.message_queue.clear()

            import traceback
            import sys
            print "current things:", self.display_string
            print "messages", list(self.message_queue)
            print(traceback.format_exc())
            return features, display_string #return default and continue with work





