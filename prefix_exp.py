import _pickle as pickle
import numpy as np
import argparse
from CharRNN import CharRNN
from POStagger import POStagger as POStaggerBasic
from chainer import cuda, Variable, FunctionSet, serializers
#from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import math
parser = argparse.ArgumentParser()
parser.add_argument('--base_type',  type=str,   default = 'Real')
parser.add_argument('--pref_type',  type=str,   default = 'Real')
parser.add_argument('--unamb',      type=int,   default=0)
args = parser.parse_args()

training_data = open('data/en/train_input.txt','r').read().split()
print(training_data[5])
words = pickle.load(open('results_dict/{}words'.format(args.base_type.lower()),'rb'))
print(len(words))
for i in range(5):
    print(list(words.keys())[i])
vocab = pickle.load(open('data/en/vocab.bin', 'rb'))
tag_vocab = pickle.load(open('data/en/tag_vocab.bin', 'rb'))
tag_ivocab = {}
for c, i in tag_vocab.items():
    tag_ivocab[i] = c

if args.pref_type == 'Pseudo':
    prefixes = ({'NOUN':['letno','latno','sace','olsh','leni','lafi'],
                'VERB':['la','nef','aip','afel','omgel','ot','gorm'],
                'ADJ':['ofpla','ompe','clo']})
elif args.pref_type == 'Real' and args.unamb:
    prefixes = ({'NOUN':['micro','macro','vice','arch','mini','maxi'],
                'VERB':['re','mis','out','over','under','up','down'],
                'ADJ':['extra','anti','pro']})

elif args.pref_type == 'Real' and not args.unamb:
    prefixes = ({'NOUN':['un'],
                'VERB':['re'],#'mis','out','over','under','up','down','counter','pre','un','in'],
                'ADJ':['anti']})#,'pro','extra','in','un','pre','sub','super','hyper','hypo','ultra','post','ante','pseudo','ex','multi','meta']})
#http://alehrer.faculty.arizona.edu/sites/alehrer.faculty.arizona.edu/files/Prefixes%20in%20English%20word%20formation.pdf
#['micro','macro','vice','arch','mini','maxi','counter','pre','sub','super','hyper','hypo','ultra','post','ante','pseudo','ex','multi','meta'],
model = CharRNN(len(vocab), 256, 1)
serializers.load_npz('cv/charrnn_1.59.chainermodelnew', model)

POSmodel = POStaggerBasic(len(tag_vocab),256,2)
serializers.load_npz('cv/tagger_.9075.chainermodelnew', POSmodel)

#main dictionary with tuples of original word and its tag as keys and list of tuples of modified word and its tag as values
results = {(word, words[word][4]):[] for word in words.keys()}
#overall counts of tags of modified word averaged over the tags of all original words
results_cumm = {i:{j:0 for j in tag_vocab.keys()} for i in ['NOUN','VERB','ADJ']}
#tracks the percentage of modified word that pass the threshold for probability of POS tag
correct, incorrect = {i:[] for i in ['NOUN','VERB','ADJ']},{i:[] for i in ['NOUN','VERB','ADJ']}
words_match = {}
len_match = []
total_considered = {}
len_mismatch = []
softmax = {i:{j:[] for j in prefixes[i]} for i in ['NOUN','VERB','ADJ']}
results_per_pref = {}
prob_real_tag = {i:[] for i in ['NOUN','VERB','ADJ']}
prev_tag = len(tag_vocab)-1
length = {i:[] for i in ['NOUN','VERB','ADJ']}
total_pplx ={i:[] for i in ['NOUN','VERB','ADJ']}
for word in words.keys():
    follow_state,init_state,prob,init_prob,real_tag = words[word]
    length[real_tag].append(len(word))
    for j in prefixes[real_tag]:
        comb = j+word
        pplx = 0
        if comb not in training_data:
            if j in total_considered.keys():
                total_considered[j]+=1
            else:
                total_considered[j]=1

            for i,lstm_name in zip(range(1),model.lstm_enc):
                model[lstm_name].set_state(init_state['c{0:d}'.format(i)],init_state['h{0:d}'.format(i)])
            prob = init_prob
            for char in comb:

                try:
                    pplx+= math.log(cuda.to_cpu(prob.data)[0][vocab[char]],2)
                except: pass
                prev_char = np.array([vocab[char]], dtype=np.int32)
                state, prob = model.forward_one_step(prev_char, prev_char, train=False)
            total_pplx[real_tag].append(pplx/len(comb))
            prob_tag = (POSmodel.forward_one_step(state['h0'].data,np.array([prev_tag],dtype=np.int32),
                        np.array([prev_tag],dtype=np.int32),train_dev=True, train = False).data)
            prob_real_tag[real_tag].append(np.max((POSmodel.forward_one_step(follow_state['h0'].data,np.array([prev_tag],dtype=np.int32),
                        np.array([prev_tag],dtype=np.int32),train_dev=True, train = False).data)))
            tag = tag_ivocab[np.argmax(prob_tag)]
            softmax[real_tag][j].append(prob_tag)
            results[(word,real_tag)].append((comb,tag))
            results_cumm[real_tag][tag]+=1



            if real_tag==tag.decode():
                if j in words_match.keys():
                    words_match[j].append(word)
                else:
                    words_match[j] = [word]
                correct[real_tag].append(np.max(prob_tag))
                len_match.append(len(word))
                if j in results_per_pref.keys():
                    results_per_pref[j]+=1
                else:
                    results_per_pref[j]=1
            else:
                incorrect[real_tag].append(np.max(prob_tag))
                len_mismatch.append(len(word))

#pickle.dump(results_per_pref,open('results_dict/results_per_pref_{}_{}_unamb{}'.format(args.pref_type, args.base_type,args.unamb),'wb'))
#pickle.dump(words_match,open('results_dict/words_match_{}_pref_{}_unamb{}_base'.format(args.pref_type,args.base_type,args.unamb),'wb'))
#pickle.dump(results,open('results_dict/results_{}_{}_unamb{}_new'.format(args.pref_type,args.base_type,args.unamb),'wb'))
#pickle.dump(results_cumm,open('results_dict/results_cumm_{}_{}_unamb{}_new'.format(args.pref_type,args.base_type,args.unamb),'wb'))
pickle.dump(softmax, open('results_dict/softmax_un_NOUN_{}_{}_unamb{}'.format(args.pref_type,args.base_type,args.unamb),'wb'))
print('Base type ', args.base_type)
for i in ['NOUN','VERB','ADJ']:
    print(i)
    print('Average certainty of real tag', np.mean(prob_real_tag[i]), np.std(prob_real_tag[i]))
    print('Average certainty of matching tags', np.mean(correct[i]), np.std(correct[i]))
    print('Average certainty of mismatching  tags', np.mean(incorrect[i]), np.std(incorrect[i]))
    print('Average log probability of the modified words', np.mean(total_pplx[i]), np.std(total_pplx[i]))
    print('Average length of base',np.mean(length[i]), np.std(length[i]))
print('Average length of words with matching tags: ', np.mean(len_match), np.std(len_match))
#pickle.dump(total_considered,open('results_dict/total_considered_{}_pref_{}_unamb{}_base'.format(args.pref_type,args.base_type,args.unamb),'wb'))
print('Average length of words with mismatching tags: ', np.mean(len_mismatch), np.std(len_mismatch))
#cm = ConfusionMatrix(y_true, y_pred)
#cm.plot(normalized=True)
#plt.title('Effect of Prepending a Prefix to a {} Base'.format(args.base_type))
#plt.show()
