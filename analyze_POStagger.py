import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers, serializers, links
import chainer.functions as F
from POStagger import POStagger as POStaggerBasic
import _pickle as pickle
from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import codecs

tag_vocab = pickle.load(open('data/en/tag_vocab.bin', 'rb'))
ivocab = {}
for c, i in tag_vocab.items():
    ivocab[i] = c

words = codecs.open('data/raw_data_pos/raw_english_{}.txt'.format(10), 'rb', 'utf-8').read().split()
all_words = codecs.open('data/en/raw_english.txt', 'r', 'utf-8').read().split()

model = POStaggerBasic(len(tag_vocab),256,2)
# load model
serializers.load_npz('cv/tagger_.9075.chainermodelnew', model)

word_emb_dict = pickle.load(open('data/en/final_word_emb_Partial_10','rb'))

tags = [word_emb_dict[i][1] for i in sorted(list(word_emb_dict.keys()))]
len_dev = len(tags)
word_emb = [word_emb_dict[i][0] for i in sorted(list(word_emb_dict.keys()))]
word_emb = np.array(word_emb)


y_true = []
y_pred = []

def draw_confusion_matrix(num):
    req = {}
    title = {}
    prev_tag = len(tag_vocab)-1
    for j,tag in enumerate(tags):
        prob, loss,accu = model.forward_one_step(word_emb[j],np.array([prev_tag],dtype=np.int32),np.array([tag], dtype=np.int32),train_dev=True)
        prev_tag = (np.argmax(cuda.to_cpu(prob.data)))

        req[1] = np.max(cuda.to_cpu(prob.data))<0.857
        title[1] = 'Inconfident POS Tagger Predicitons: Confusion Matrix'
        req[2] = np.max(cuda.to_cpu(prob.data))>0.857
        title[2] = 'Confident POS Tagger Predicitons: Confusion Matrix'
        req[3] = 1
        title[3] = 'POS Tagger Predictions: Confusion Matrix'
        req[4] = all_words.count(words[j])<=3
        title[4] = 'POS Tagger Predictions for Rare Words'

        if req[num]:
            y_true.append(ivocab[tag].decode())
            y_pred.append(ivocab[prev_tag].decode())

    cm = ConfusionMatrix(y_true, y_pred)
    cm.plot(normalized=True)
    plt.title(title[num])
    plt.show()

for i in range(2,3):
    draw_confusion_matrix(i)
