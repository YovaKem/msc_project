#%%
from __future__ import division
from __future__ import print_function
import time
import math
import sys
import argparse
import _pickle as pickle
import codecs
from POStagger import POStagger as POStaggerBasic

import numpy as np
from chainer import cuda, Variable, FunctionSet, serializers
import chainer.functions as F
from CharRNN import CharRNN
import enchant

d = enchant.Dict('en-US')
e = enchant.Dict('en-UK')
#text = open('/afs/inf.ed.ac.uk/user/s12/s1233656/chainer1/generate_output').read().split()
#sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

#%% arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model',      type=str,   default = 'cv/charrnn_1.59.chainermodelnew')
parser.add_argument('--vocabulary', type=str,   default = 'data/en/vocab.bin')

parser.add_argument('--seed',       type=int,   default=123)
parser.add_argument('--sample',     type=int,   default=1)
parser.add_argument('--primetext',  type=str,   default='')
parser.add_argument('--length',     type=int,   default=50000000)
parser.add_argument('--gpu',        type=int,   default=-1)
parser.add_argument('--n_units',    type=int,   default=256)
parser.add_argument('--n_layers',    type=int,   default=1)


args = parser.parse_args()

#np.random.seed(args.seed)
n_units = args.n_units
n_layers = args.n_layers
#pseudowords = {}
real_words = {}
# load vocabulary
vocab = pickle.load(open(args.vocabulary, 'rb'))
ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c

# load model
model = CharRNN(len(vocab), n_units, n_layers)

# load model
#model = pickle.load(open(args.model, 'rb'))
serializers.load_npz(args.model, model)

tag_vocab = pickle.load(open('data/en/tag_vocab.bin', 'rb'))
tag_ivocab = {}
for c, i in tag_vocab.items():
    tag_ivocab[i] = c

POSmodel = POStaggerBasic(len(tag_vocab),256,2)
# load model
serializers.load_npz('cv/tagger_.9075.chainermodelnew', POSmodel)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
try:
    pseudowords = pickle.load(open('pseudowords', 'rb'))
except:
    pseudowords = {}

prev_char = np.array([0], dtype=np.int32)

word = ''
prev_tag = len(tag_vocab)-1

pplx = 0
ave_real_tag,ave_real_pplx,ave_real_space,real_words = [],[],[],[]
ave_pseudo_pplx, ave_pseudo_tag = [],[]
count = 0
tags = {i:0 for i in ['NOUN','VERB','ADJ']}
save_next=1

for i in range(args.length):
    state, prob = model.forward_one_step(prev_char, prev_char, train=False)
    if save_next:
        state_init = state
        prob_init = prob
        save_next=0
    if args.sample > 0:
        probability = cuda.to_cpu(prob.data)[0]
        probability /= np.sum(probability)
        index = np.random.choice(range(len(probability)), p=probability)
    else:
        index = np.argmax(cuda.to_cpu(prob.data))
    #sys.stdout.write(ivocab[index])
    #sys.stdout.write(str(np.max(prob.data)))
    prev_char = np.array([index], dtype=np.int32)

    if ivocab[index] != ' ':
        word+=ivocab[index]
        pplx += math.log(probability[index],2)
    else:
        save_next = 1
        try:
            suff = (['acy','al','ence','dom','ity','ness','ful','ion','ate','hood','ship','ize','ing',
                    'ed','able','s','ly','ant','ory','ary','ic','ify','ment','ism','er','ist','ance',
                    'age','or','ium','ible','ee','less','ous','ish','ive','ian','ian','ent','logy','est'])
            if len(ave_real_tag)<301 and d.check(word) and e.check(word) and not word[0].isupper() and all([c.isalpha() for c in word]):
                print(len(ave_real_tag))
                prob_tag = POSmodel.forward_one_step(state['h0'].data,np.array([prev_tag],dtype=np.int32),np.array([prev_tag],dtype=np.int32),train_dev=True, train = False)
                if word not in real_words:
                    ave_real_tag.append(np.max(cuda.to_cpu(prob_tag.data)))
                    ave_real_pplx.append(pplx/len(word))
                    ave_real_space.append(prob.data[0][vocab[' ']])
                    real_words.append(word)
                ave_tag = np.mean(ave_real_tag)
                ave_space = np.mean(ave_real_space)
                ave_pplx = np.mean(ave_real_pplx)

            if len(ave_real_space)>=300 and not d.check(word) and not e.check(word) and not word[0].isupper() and all([c.isalpha() for c in word]) and not any([word.endswith(seq) for seq in suff]):
                print(word)
                prob_tag = POSmodel.forward_one_step(state['h0'].data,np.array([prev_tag],dtype=np.int32),np.array([prev_tag],dtype=np.int32),train_dev=True, train = False)
                #print('Word: {} -- {},{}'.format(word, ivocab[(np.argmax(cuda.to_cpu(prob.data)))],np.max(prob.data)))
                if np.max(prob_tag.data) >= np.mean(ave_real_tag)-np.std(ave_real_tag) and pplx/len(word) >= np.mean(ave_real_pplx)-np.std(ave_real_pplx) and  prob.data[0][vocab[' ']] >= np.mean(ave_real_space) - np.std(ave_real_space):
                    tag = tag_ivocab[(np.argmax(cuda.to_cpu(prob_tag.data)))].decode()

                    if tags[tag]<10 and word not in pseudowords.keys():
                        tags[tag]+=1
                        ave_pseudo_pplx.append(pplx/len(word))
                        ave_pseudo_tag.append(np.max(cuda.to_cpu(prob_tag.data)))
                        pseudowords[word] = (state, state_init,prob, prob_init, tag)
                        print(tag, tags[tag])
                        print('stored {} pseudowords'.format(len(pseudowords)))

        except:
            pass
        word = ''
        pplx = 0

    if tags['NOUN']>=10 and tags['VERB']>=10 and tags['ADJ']>=10:
        break
    '''if i%5000==0:
        try:
            print('pseudowords',sys.getsizeof(index))
        except:
            pass
        try:
            print('prob_tag',sys.getsizeof(prob_tag))
        except:
            pass
        print('state',sys.getsizeof(state))
        print('prob',sys.getsizeof(prob))'''
pickle.dump(pseudowords, open('pseudowords', 'wb'))
print(len(pseudowords))
print('Real words')
print(ave_tag, np.std(ave_real_tag))
print(ave_pplx, np.std(ave_real_pplx))
print(ave_space, np.std(ave_real_space))
print('Pseudo words')
print(np.mean(ave_pseudo_tag), np.std(ave_pseudo_tag))
print(np.mean(ave_pseudo_pplx),np.std(ave_pseudo_pplx))
#print ('real', np.mean(tot_prob),np.std(tot_prob))
#print ('pseudo', np.mean(tot_prob_ps),np.std(tot_prob_ps))
