#%%

#tasks:
#extract nominal contexts from UD data
#think bout syntactic restrictions on suffix combination
#get wuggy output and run some gr 1 suffixes and some gr 1 suffixes on it --> plot length against perplexity; compare prob of gr1 v gr1 suffixes in this context across all lengths
#generate 'suffixed' context and append gr 1 and gr 1 suffixes to them --> lower pplx for gr 1? Factor out length effects and general productivity effects (learned from line above)

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
import matplotlib.pyplot as plt

#sys.stdout = codecs.getwriter('utf8')(sys.stdout)

#%% arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model',      type=str,   default = 'cv/charrnn_1.59.chainermodelnew')
parser.add_argument('--vocabulary', type=str,   default = 'data/en/vocab.bin')
parser.add_argument('--seed',       type=int,   default=123)
parser.add_argument('--sample',     type=int,   default=1)
parser.add_argument('--primetext',  type=str,   default='')
parser.add_argument('--length',     type=int,   default=2000)
parser.add_argument('--gpu',        type=int,   default=-1)
parser.add_argument('--n_units',    type=int,   default=256)
parser.add_argument('--n_layers',    type=int,   default=1)
parser.add_argument('--train_pickle',    type=int,   default=0)

args = parser.parse_args()

#np.random.seed(args.seed)
n_units = args.n_units
n_layers = args.n_layers
# load vocabulary
vocab = pickle.load(open(args.vocabulary, 'rb'))
ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c
model = CharRNN(len(vocab), n_units, n_layers)
serializers.load_npz(args.model, model)

pseudowords = pickle.load(open('results_dict/pseudowords_new', 'rb'))
tags = {word:pseudowords[word][4] for word in pseudowords.keys()}

suffixes = {}
#suffixes['deverbal'] = ['ance','ment','ant','ory','ive','ion','able','ably','al']
#suffixes['denominal'] = ['age','ful','ous','an','ic','ify','ate','ary','hood','less']
#suffixes['deadjectival'] = ['ify','ly','ism','ist','ize','ish','ness','ity','en']

suffixes['deverbal'] = ['ance','ment','ant','ory','ive','ion','able','ably']
suffixes['denominal'] = ['ous','an','ic','ate','ary','hood','less','ish']
suffixes['deadjectival'] = ['ness','ity','en']
excluded=0
#main dictionary with pseudowords as keys to dictionaries with suffixes as keys and their probability as values
final_prob = {i:{} for i in pseudowords.keys() if not i.endswith('lar')}
# primetext in [list(pseudowords.keys())[i] for i in [2,3,4,145,146,147,245,246]]:
for primetext in pseudowords.keys():
    if not primetext.endswith('lar'):
        print(primetext)
        follow_state,init_state,follow_prob,init_prob,tag = pseudowords[primetext]
        for suff in [item for sublist in suffixes.values() for item in sublist]:
            prob = follow_prob
            for i,lstm_name in zip(range(n_layers),model.lstm_enc):
                    model[lstm_name].set_state(follow_state['c{0:d}'.format(i)],follow_state['h{0:d}'.format(i)])
            total_prob = 0
            for char in suff:
                total_prob += log(prob.data[0][vocab[char]],2)
                prev_char = np.ones((1,), dtype=np.int32) * vocab[char]
                state, prob = model.forward_one_step(prev_char, prev_char, train=False)
            final_prob[primetext][suff] = total_prob/len(suff)
    else:
        excluded+=1
print('EXCLUDED',excluded)
#average probability of suffix category per base category
cat_catsuff = {i:{} for i in ['NOUN','VERB','ADJ']}
#average probability of suffix per base category
cat_suf = {i:{j:{} for j in suffixes.keys()} for i in ['NOUN','VERB','ADJ']}
for cat in cat_catsuff.keys():
    for suf_cat in suffixes.keys():
        cat_catsuff[cat][suf_cat] = np.mean([final_prob[base][suf] for base in final_prob.keys() if tags[base] ==cat for suf in final_prob[base].keys() if suf in suffixes[suf_cat] ])
        for suf in suffixes[suf_cat]:
            cat_suf[cat][suf_cat][suf] = np.mean([final_prob[base][suf] for base in final_prob.keys() if tags[base]==cat])

pickle.dump(cat_suf, open('results_dict/prob_suff_per_basecat_unamb_new', 'wb'))
pickle.dump(cat_catsuff, open('results_dict/prob_suffcat_per_basecat_unamb_new', 'wb'))

#plot(final_prob, el_1, el_2,suffix, gr1,contexts)
#compare_contexts(final_prob,'suffixed',el_1, el_2, suffix,gr1,contexts)
#compare_contexts(final_prob,'bare',el_1, el_2, suffix,gr1,contexts)
