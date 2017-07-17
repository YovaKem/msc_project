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
import enchant

d = enchant.Dict('en-US')
text = open('/afs/inf.ed.ac.uk/user/s12/s1233656/chainer1/generate_output').read().split()
#sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

#%% arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model',      type=str,   default = 'CharRNN/charrnn_1.59.chainermodelnew')
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


from sklearn.metrics.pairwise import cosine_similarity
sports = open('data/en/sport.txt','r').readlines()
states = open('data/en/states.txt','r').readlines()
people = open('data/en/people.txt','r').readlines()
months = ['January','March','April','May','June','July','August','September']
random1 = ['January','Alabama','wrestling','George']
random2 = ['March', 'Wisconsin','rugby','Alexander']
random3 = ['August','Mississippi','Fernand','swim']
random4 = ['June','Georgia','UEFA','Erich']
similarity = {i:{} for i in ['random1','random2','random3','random4','sports','states','people','months','prepositions']}
for label,subset in zip(['random1','random2','random3','random4','sports','states','people','months','prepositions'],[random1,random2,random3,random4,sports,states,people,months]):
    final_emb = {}
    for word in subset:
        emb = []
        for i in range(1000):
            model.reset_state()
            for char in word.strip():
                prev_char = np.array([vocab[char]], dtype=np.int32)
                state,prob = model.forward_one_step(prev_char, prev_char, train=False)
            emb.append(state['h0'].data)
        final_emb[word.strip()] = np.mean(emb,axis=0)
    dist = []
    pairs = [(x.strip(),y.strip()) for x in subset for y in subset if x!=y]
    #print(final_emb)
    for x,y in pairs:
        dist.append(cosine_similarity(final_emb[x].reshape(1,-1),final_emb[y].reshape(1,-1)))
        #print(x,y)
        #print(cosine_similarity(final_emb[x].reshape(1,-1),final_emb[y].reshape(1,-1)))
    similarity[label]['mean'] = np.mean(dist)
    similarity[label]['std'] = np.std(dist)
    print(label, similarity[label]['mean'], similarity[label]['std'])
pickle.dump(similarity, open('similarity','wb'))
