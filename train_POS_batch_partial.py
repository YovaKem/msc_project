

from __future__ import division

import time
import math
import sys
import argparse
import _pickle as pickle
import copy
import os
import codecs
from collections import defaultdict

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers, serializers, links
import chainer.functions as F
from POStagger import POStagger as POStaggerBasic

from CharRNN import CharRNN

def load_data(args, vocab, subset):
    words = codecs.open('data/raw_data_pos/raw_english_{}.txt'.format(subset), 'rb', 'utf-8').read()
    tags_raw = open('data/raw_data_pos/tags_english_{}.txt'.format(subset),'rb').read().split()
    words = list(words)
    dataset_train = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        try:
            dataset_train[i] = vocab[word]
        except:
            dataset_train[i] = vocab['UNK']

    tags = np.ndarray((len(tags_raw),), dtype=np.int32)
    for i, tag in enumerate(tags_raw):
        tags[i] = tag_vocab[tag]


    return dataset_train, tags, tag_vocab



parser = argparse.ArgumentParser()
parser.add_argument('--init_from',                  type=str,   default='charrnn_1.59.chainermodelnew')
parser.add_argument('--checkpoint_dir',             type=str,   default='cv')
parser.add_argument('--n_units',                    type=int,   default=256)
parser.add_argument('--n_layers',                   type=int,   default=1)
parser.add_argument('--vocab',                      type=str,   default='data/en/vocab.bin')
parser.add_argument('--train_emb',                  type=int,   default=0)
parser.add_argument('--setting',                    type=str,   default='train')
parser.add_argument('--type',                       type=str,   default='basic')

args = parser.parse_args()

vocab = pickle.load(open(args.vocab, 'rb'))
print(len(vocab))

tag_vocab = pickle.load(open('data/en/tag_vocab.bin', 'rb'))
model_char = CharRNN(len(vocab), args.n_units, args.n_layers)
serializers.load_npz(args.init_from, model_char)

def generate_embeddings(subset):
    data, tags, tag_vocab   = load_data(args, vocab,subset)
    whole_len    = data.shape[0]

    batch_data = np.array(data, dtype=np.int32)
    batch_data = batch_data.reshape((whole_len,1))
    word_emb = {}
    for i in range(whole_len):
        if batch_data[i][0]==vocab[' ']:
            word_emb[len(word_emb)] = (state['h{0:d}'.format(args.n_layers-1)].data,tags[len(word_emb)])

        state, y = model_char.forward_one_step(batch_data[i], batch_data[i], train_dev=False, train = False)


    t = open('data/en/final_word_emb_Partial_{}'.format(subset),'wb')
    print('CHECK 2',len(word_emb)==len(tags))
    pickle.dump(word_emb,t)
    t.close()
    print('embeddings trained and saved')

subsets = 10
for i in range(subsets):
    print('training subset {} out of {}'.format(i, subsets))
    generate_embeddings(i)

def train_tagger(n_units, n_layers, learning_rate):

    data, all_tags, tag_vocab   = load_data(args, vocab, 0)
    log_name                = 'final_basic_model_partial_emb_{}_{}_{}.log'.format(n_layers, n_units, learning_rate)
    model                   = POStaggerBasic(len(tag_vocab),n_units,n_layers)
    # load model
    #serializers.load_npz('cv/partialbasic_128_2_0.005_31.chainermodelnew', model)
    log                     = open('%s/%s'%(args.checkpoint_dir,log_name), 'w')


    optimizer               = optimizers.MomentumSGD(learning_rate)
    optimizer.setup(model)
    curr_dev_loss           = 0
    prev_dev_loss           = 0

    ivocab = {}
    for c, i in tag_vocab.items():
        ivocab[i] = c

    '''try:
        word_emb_dict = pickle.load(open('data/en/final_word_emb_{}'.format(len(tags)), 'rb'))
        print('loaded embeddings')
    except:'''
    #print('training embeddings...')


    batchsize               = 100
    bprop_len               = 1
    whole_len               = 18358
    jump                    = whole_len // batchsize
    epoch                   = 0
    accum_loss              = Variable(np.zeros((), dtype=np.float32))
    total_loss, train_accuracy = 0.0,0.0
    count_train_accuracy = 0
    count_total_loss = 0
    for i in range(jump * 100):

        for k in range(subsets):

            word_emb_dict = pickle.load(open('data/en/final_word_emb_Partial_{}'.format(k),'rb'))
            tags = [word_emb_dict[i][1] for i in sorted(list(word_emb_dict.keys()))]
            word_emb = [word_emb_dict[i][0] for i in sorted(list(word_emb_dict.keys()))]
            word_emb = np.array(word_emb)

            x_batch1 = np.array([word_emb[(jump * j + i+1) % whole_len]
                                for j in range(batchsize)])
            x_batch2 = np.array([tags[(jump * j + i) % whole_len]
                                for j in range(batchsize)])
            y_batch = np.array([tags[(jump * j + i + 1) % whole_len]
                                for j in range(batchsize)])


            prob, loss_i, accu = model.forward_one_step(x_batch1,x_batch2, y_batch)
            train_accuracy += sum([1 for item in accu if item])
            count_train_accuracy += sum([1 for item in accu])
            accum_loss   += loss_i

            if (i + 1) % bprop_len == 0:  # Run truncated BPTT
                now = time.time()
                #print ('{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)//bprop_len, jump, accum_loss.data / bprop_len, now-cur_at))
                cur_at = now

                optimizer.zero_grads()
                accum_loss.backward()
                accum_loss.unchain_backward()  # truncate
                total_loss += accum_loss.data/ bprop_len
                count_total_loss +=1

                accum_loss = Variable(np.zeros((), dtype=np.float32))

                optimizer.clip_grads(5)
                optimizer.update()

        if (i + 1) % jump == 0:
            accum_dev_loss, dev_accuracy = 0.0,0.0
            word_emb_dict = pickle.load(open('data/en/final_word_emb_Partial_{}'.format(subsets),'rb'))

            tags = [word_emb_dict[i][1] for i in sorted(list(word_emb_dict.keys()))]
            len_dev = len(tags)
            word_emb = [word_emb_dict[i][0] for i in sorted(list(word_emb_dict.keys()))]
            word_emb = np.array(word_emb)
            prev_tag = len(tag_vocab)-1
            for j,tag in enumerate(tags):
                prob, loss,accu = model.forward_one_step(word_emb[j],np.array([prev_tag],dtype=np.int32),np.array([tag], dtype=np.int32),train_dev=True)
                if accu: dev_accuracy+=1
                accum_dev_loss += loss
                prev_tag = (np.argmax(cuda.to_cpu(prob.data)))
                #print('dev',ivocab[prev_tag])


            print('Training loss {} on epoch {}'.format(total_loss/count_total_loss,epoch))
            log.write('Training loss {} on epoch {}\n'.format(total_loss/count_total_loss,epoch))
            print('Training accuracy', train_accuracy/count_train_accuracy)
            log.write('Training accuracy: {}\n'.format(train_accuracy/count_train_accuracy))
            print('Dev loss',accum_dev_loss.data/len_dev)
            log.write('Dev loss: {}\n'.format(accum_dev_loss.data/len_dev))
            f = 'cv/partial{}_{}_{}_{}_{}.chainermodelnew'.format(args.type,n_units, n_layers,learning_rate,epoch)
            serializers.save_npz(f, model)
            prev_prev_dev_loss = prev_dev_loss
            prev_dev_loss = curr_dev_loss
            curr_dev_loss = accum_dev_loss.data/len_dev
            print('Dev accuracy',dev_accuracy/len_dev)
            log.write('Dev accuracy: {}\n'.format(dev_accuracy/len_dev ))

            print('_____________________________________________')
            epoch += 1
            total_loss = 0.0
            train_accuracy = 0.0
            count_train_accuracy = 0
            count_total_loss = 0
            if epoch >= 10:
                optimizer.lr *= 0.97

            if epoch>10 and prev_dev_loss != 0 and curr_dev_loss >= prev_dev_loss and prev_dev_loss >= prev_prev_dev_loss:
                break
    print ('Done')

def compute_dev_pplx():
    tag_vocab = pickle.load(open('data/en/tag_vocab.bin', 'rb'))
    tag_ivocab = {}
    log = open('%s/%s'%(args.checkpoint_dir,'dev_stats_partial_25_31'), 'w')
    for c, i in tag_vocab.items():
        tag_ivocab[i] = c
    for i in range(25,32):
        POSmodel = POStaggerBasic(len(tag_vocab),128,2)
        # load model
        serializers.load_npz('cv/partialbasic_128_2_0.005_{}.chainermodelnew'.format(i), POSmodel)

        accum_dev_loss, dev_accuracy = 0.0,0.0
        word_emb_dict = pickle.load(open('data/en/final_word_emb_Partial_{}'.format(subsets),'rb'))

        tags = [word_emb_dict[i][1] for i in sorted(list(word_emb_dict.keys()))]
        len_dev = len(tags)
        word_emb = [word_emb_dict[i][0] for i in sorted(list(word_emb_dict.keys()))]
        word_emb = np.array(word_emb)
        prev_tag = len(tag_vocab)-1
        for j,tag in enumerate(tags):
            prob, loss,accu = POSmodel.forward_one_step(word_emb[j],np.array([prev_tag],dtype=np.int32),np.array([tag], dtype=np.int32),train_dev=True)
            if accu: dev_accuracy+=1
            accum_dev_loss += loss
            prev_tag = (np.argmax(cuda.to_cpu(prob.data)))
        print('EPOCH',i)
        print('Dev loss',accum_dev_loss.data/len_dev)
        log.write('Dev loss: {}\n'.format(accum_dev_loss.data/len_dev))
        print('Dev accuracy',dev_accuracy/len_dev)
        log.write('Dev accuracy: {}\n'.format(dev_accuracy/len_dev ))
        print('_____________________________________________')
train_tagger(256,2,0.005)
#compute_dev_pplx()
