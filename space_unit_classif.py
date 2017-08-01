import _pickle as pickle
import numpy as np
import argparse
from CharRNN import CharRNN
from POStagger import POStagger as POStaggerBasic
from chainer import cuda, Variable, FunctionSet, serializers
import math
import operator
import matplotlib.pyplot as plt
import seaborn as sns
import enchant

d = enchant.Dict('en-US')
e = enchant.Dict('en-UK')

parser = argparse.ArgumentParser()
parser.add_argument('--subset',  type=int,   default = 0)
args = parser.parse_args()
subset = args.subset

vocab = pickle.load(open('data/en/vocab.bin', 'rb'))
model = CharRNN(len(vocab), 256, 1)
serializers.load_npz('cv/charrnn_1.59.chainermodelnew', model)
#try:
    #print('loaded top k dict')
    #top_k_contexts = pickle.load(open('top_10_context','rb'))
#except:
top_k_contexts = {i:{} for i in range(256)}

def feed_next_char(char):
    prev_char = np.array([vocab[char]], dtype=np.int32)
    state, prob = model.forward_one_step(prev_char, prev_char, train=False)
    return state['h0'].data[0]

def update_top_k(h_state, context):
    for i in range(256):
        activation = h_state[i]
        unit = top_k_contexts[i]
        sorted_unit = sorted(unit.items(), key=operator.itemgetter(1))
        if len(sorted_unit)==10 and activation > sorted_unit[0][1]:
            del top_k_contexts[i][sorted_unit[0][0]]
            top_k_contexts[i][context]=activation
        elif len(sorted_unit)<10:
            top_k_contexts[i][context]=activation

def main():
    subset=0
    data = open('data/en/train_input.txt').read()
    data = data[100000*subset:100000*(subset+1)]
    classif = {i:0 for i in ['tp','fp','tn','fn']}
    word = ''
    for i in range(len(data)-1):
        h_state = feed_next_char(data[i])
        if data[i].isalpha():
            word+=data[i]
            if word[0].islower():
                if h_state[13]>=0.5 and (d.check(word) or e.check(word) or not data[i+1].isalpha()):
                    classif['tp']+=1
                    #print('True POSITIVE',data[i-10:i+1])
                elif h_state[13]>=0.5 and (data[i+1].isalpha() and (not d.check(word) and not e.check(word))):
                    classif['fp']+=1
                    #print('False POSITIVE',data[i-10:i+1])
                elif h_state[13]<0.5 and (not data[i+1].isalpha() or d.check(word) or e.check(word)):
                    classif['fn']+=1
                    #print('False NEGATIVE',data[i-10:i+1])
                elif h_state[13]<0.5 and data[i+1].isalpha():
                    classif['tn']+=1
                    #print('True NEGATIVE',data[i-10:i+1])
        else:
            word=''
    #fig = plt.figure(figsize=(10,7))
    #plt.plot(range(len(evolution_12)),evolution_12)
    #for i,char in enumerate(string):
    #    if (char==' ' or char==')') and i!=16:
    #        plt.plot((i-1, i-1), (1, -1), 'k--')
    #plt.xticks(range(len(evolution_12)),string)
    #plt.title('Evolution of Unit 13')
    #plt.show()
    #fig.savefig('unit_13_evolution.png')
    pickle.dump(classif, open('classif_unit13','wb'))
    print(classif)
main()