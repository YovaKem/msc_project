#%%
from __future__ import division
from __future__ import print_function
import time
import math
import sys
import argparse
import _pickle as pickle
import codecs

import numpy as np
from chainer import cuda, Variable, FunctionSet, serializers
import chainer.functions as F
from CharRNN import CharRNN

vocab = pickle.load(open('data/en/vocab.bin', 'rb'))
model = CharRNN(len(vocab), 256, 1)
serializers.load_npz('cv/charrnn_1.59.chainermodelnew', model)

def feed_word(word):
    prev_char = np.array([vocab[' ']], dtype=np.int32)
    state,prob = model.forward_one_step(prev_char, prev_char, train=False)
    init_state = state
    init_prob = prob
    for char in word:
        if char in vocab:
            prev_char = np.array([vocab[char]], dtype=np.int32)
        else:
            prev_char = np.array([vocab['UNK']], dtype=np.int32)
        state,prob = model.forward_one_step(prev_char, prev_char, train=False)
    return state,init_state,prob,init_prob

data = open('data/en/raw_english.txt').read().split()
tags = open('data/en/tags_english.txt').read().split()
state = 0
emb = {}
count = {i:0 for i in ['ADJ','NOUN','VERB']}
suff = (['acy','al','ence','dom','ity','ness','ful','ion','ate','hood','ship','ize','ing',
        'ed','able','s','ly','ant','ory','ary','ic','ify','ment','ism','er','ist','ance',
        'age','or','ium','ible','ee','less','ous','ish','ive'])


for i,word,tag in zip(range(30000),data[-30000:],tags[-30000:]):
    state,init_state,prob,init_prob = feed_word(word)
    if i>10 and tag in ['ADJ','NOUN','VERB'] and not any(word.endswith(seq) for seq in suff) and count[tag]<300 and word[0].islower() and all([c.isalpha() for c in word]) and word not in emb.keys():
        emb[word]=state,init_state,prob,init_prob,tag
        count[tag]+=1
        print('Added {} {}'.format(count[tag],tag))
    if count['ADJ']>=300 and count['NOUN']>=300 and count['VERB']>=300:
        break
pickle.dump(emb,open('realwords','wb'))
