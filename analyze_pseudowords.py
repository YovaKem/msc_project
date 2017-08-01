from __future__ import division
from __future__ import print_function
import time
import math
import sys
import argparse
import _pickle as pickle
import codecs
from collections import defaultdict
import itertools
from math import log
import numpy as np
from chainer import cuda, Variable, FunctionSet, serializers
import chainer.functions as F
from CharRNN import CharRNN

vocab = pickle.load(open('data/en/vocab.bin', 'rb'))
ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c
model = CharRNN(len(vocab), 256, 1)
serializers.load_npz('cv/charrnn_1.59.chainermodelnew', model)
suffixes = ['ment']
tagg = 'VERB'
pseudowords = pickle.load(open('results_dict/pseudowords', 'rb'))
for pseudow in pseudowords:
    print(pseudow)
    print(pseudowords[pseudow][4])
space_adj,space_other =[], []
tags = {word:pseudowords[word][4] for word in pseudowords.keys()}
probability = {i:[[] for j in range(len(i))] for i in suffixes}
for word in pseudowords:
    follow_state,init_state,follow_prob,init_prob,tag = pseudowords[word]
    if tag==tagg:
        for suff in suffixes:
            prob = follow_prob
            for i,lstm_name in zip(range(1),model.lstm_enc):
                    model[lstm_name].set_state(follow_state['c{0:d}'.format(i)],follow_state['h{0:d}'.format(i)])
            for i,char in enumerate(suff):
                probability[suff][i].append(log(prob.data[0][vocab[char]],2))
                prev_char = np.ones((1,), dtype=np.int32) * vocab[char]
                state, prob = model.forward_one_step(prev_char, prev_char, train=False)
print(tagg)
for suff in probability.keys():
    for i,item in enumerate(probability[suff]):

        print(suff[i])
        print(np.mean(item))
