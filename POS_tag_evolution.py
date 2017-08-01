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
from POStagger import POStagger as POStaggerBasic
parser = argparse.ArgumentParser()

parser.add_argument('--model',      type=str,   default = 'charrnn_1.59.chainermodelnew')
parser.add_argument('--vocabulary', type=str,   default = 'data/en/vocab.bin')
parser.add_argument('--n_units',    type=int,   default=256)
parser.add_argument('--n_layers',    type=int,   default=1)

args = parser.parse_args()

#np.random.seed(args.seed)
n_units = args.n_units
n_layers = args.n_layers
# load vocabulary
vocab = pickle.load(open(args.vocabulary, 'rb'))
ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c
# load model
model = CharRNN(len(vocab), n_units, n_layers)
serializers.load_npz(args.model, model)


tag_vocab = pickle.load(open('data/en/tag_vocab.bin', 'rb'))
tag_ivocab = {}
for c, i in tag_vocab.items():
    tag_ivocab[i] = c
prev_tag = len(tag_vocab)-1
POSmodel = POStaggerBasic(len(tag_vocab),256,2)
serializers.load_npz('cv/tagger_.9075.chainermodelnew', POSmodel)


#sequences = ['you solidify ', 'solidify ', 'you longify ', 'longify', 'the betrayal', 'betrayal ' ]
seq1 = 'After attending an April 11 , 1865 , speech in which Lincoln promoted voting rights for blacks , an incensed Booth changed his plans and became'
seq2 = " determined to assassinate "
prob_tag = {'one':[]}

for char in seq1:
    prev_char = np.array([vocab[char]], dtype=np.int32)
    state,prob = model.forward_one_step(prev_char, prev_char, train=False)

for i,char in enumerate(list(seq2)):
    prev_char = np.array([vocab[char]], dtype=np.int32)
    state,prob = model.forward_one_step(prev_char, prev_char, train=False)
    emb = state['h0'].data
    probability = (cuda.to_cpu(POSmodel.forward_one_step(np.array(emb),np.array([prev_tag],dtype=np.int32),
                    np.array([prev_tag],dtype=np.int32),train_dev=True, train = False).data)[0])
    probability /= np.sum(probability)
    print(seq2[0:i+1], tag_ivocab[np.argmax(probability)],np.max(probability), probability[tag_vocab[b'NOUN']] )

    prob_tag['one'].append(probability)
pickle.dump(prob_tag, open('tag_evolution_assassinate','wb'))
