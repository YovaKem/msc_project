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
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()

parser.add_argument('--model',      type=str,   default = 'cv/charrnn_1.59.chainermodelnew')
parser.add_argument('--vocabulary', type=str,   default = 'data/en/vocab.bin')
parser.add_argument('--n_units',    type=int,   default=256)
parser.add_argument('--n_layers',    type=int,   default=1)

args = parser.parse_args()
data = open('data/en/train_input.txt').read().split()
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

sports = open('data/en/sport.txt','r').read().strip().split()
states = open('data/en/states.txt','r').read().strip().split()
people = open('data/en/people.txt','r').read().strip().split()
months = ['January','March','April','May','June','July','August','September']
random1 = ['January','Alabama','wrestling','George']
random2 = ['March', 'Wisconsin','rugby','Alexander']
random3 = ['August','Mississippi','Fernand','swim']
random4 = ['June','Georgia','UEFA','Erich']

similarity = {i:{} for i in ['random1','random2','random3','random4','sports','states','people','months','prepositions']}

def get_contexts(word):
    indices = [i for i, x in enumerate(data) if x == word]
    contexts = [data[i-10:i] for i in indices]
    return contexts

def get_embeddings(word, contexts):
    emb = []
    for con in contexts:
        con.append(word)
        con = ' '.join(w for w in con)
        model.reset_state()
        for char in con:
            prev_char = np.array([vocab[char]], dtype=np.int32)
            state,prob = model.forward_one_step(prev_char, prev_char, train=False)
        emb.append(state['h0'].data)
    return np.mean(emb,axis=0)


for label,subset in zip(['random1','random2','random3','random4','sports','states','people','months','prepositions'],[random1,random2,random3,random4,sports,states,people,months]):
    final_emb = {}
    for word in subset:
        contexts = get_contexts(word)
        final_emb[word] = get_embeddings(word,contexts)

    dist = []
    pairs = [(x.strip(),y.strip()) for x in subset for y in subset if x!=y]

    for x,y in pairs:
        dist.append(cosine_similarity(final_emb[x].reshape(1,-1),final_emb[y].reshape(1,-1)))

    similarity[label]['mean'] = np.mean(dist)
    similarity[label]['std'] = np.std(dist)
    print(label, similarity[label]['mean'], similarity[label]['std'])
pickle.dump(similarity, open('similarity','wb'))
