from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
from nltk.metrics import distance


class DeepSpeechDecoder(object):
    def __init__(self, labels, blank_index=28):
        self.labels = labels
        self.blank_index = blank_index
        self.int_to_char = dict(enumerate(labels))

    def convert_to_string(self, sequence):
        return ''.join([self.int_to_char[i] for i in sequence])

    def wer(self, decode, target):
        words = set(decode.split() + target.split())
        word2char = dict(zip(words, range(len(words))))
        new_decode = [chr(word2char[w]) for w in decode.split()]
        new_target = [chr(word2char[w]) for w in target.split()]
        return distance.edit_distance(''.join(new_decode), ''.join(new_target))

    def cer(self, decode, target):
        return distance.edit_distance(decode, target)

    def decode(self, logits):
        best = list(np.argmax(logits, axis=1))
        merge = [k for k, _ in itertools.groupby(best)]
        merge_remove_blank = [k for k in merge if k != self.blank_index]
        return self.convert_to_string(merge_remove_blank)
