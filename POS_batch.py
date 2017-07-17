

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
from POStagger_buggy import POStagger as POStaggerBasic
from POStagger_buggy_lstm import POStagger as POStaggerLSTM
from CharRNN import CharRNN

def load_data(args, vocab):
    tag_vocab = {}
    #print ('%s/input_train.txt'% args.data_dir)
    if args.setting == 'debug':
        words = codecs.open('data/en/raw_english1000.txt', 'rb', 'utf-8').read()
        tags_raw = open('data/en/tags_english1000.txt','rb').read().split()
    elif args.setting=='optimize':
        words = codecs.open('data/en/raw_english10000.txt', 'rb', 'utf-8').read()
        tags_raw = open('data/en/tags_english10000.txt','rb').read().split()
    else:
        words = codecs.open('data/en/raw_english100000.txt', 'rb', 'utf-8').read()
        tags_raw = open('data/en/tags_english100000.txt','rb').read().split()
    words = list(words)
    dataset_train = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        try:
            dataset_train[i] = vocab[word]
        except:
            dataset_train[i] = vocab['UNK']

    tags = np.ndarray((len(tags_raw),), dtype=np.int32)
    for i, tag in enumerate(tags_raw):
        if tag not in tag_vocab:
            tag_vocab[tag] = len(tag_vocab)
        tags[i] = tag_vocab[tag]
    tag_vocab['GO']=len(tag_vocab)
    '''words_dev = codecs.open('%s/input_dev.txt' % args.data_dir, 'rb', 'utf-8').read()
    words_dev = list(words_dev)
    dataset_dev = np.ndarray((len(words_dev),), dtype=np.int32)
    for i, word in enumerate(words_dev):
        if word in vocab:
            dataset_dev[i] = vocab[word]
        else:
            dataset_dev[i] = vocab['UNK']

    print ('corpus length:', len(words))
    print('dev set len:', len(words_dev))
    print ('vocab size:', len(vocab))'''
    return dataset_train, tags, tag_vocab



parser = argparse.ArgumentParser()
parser.add_argument('--init_from',                  type=str,   default='current_best_1.59.chainermodelnew')
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
data, tags, tag_vocab = load_data(args, vocab)
pickle.dump(tag_vocab, open('data/en/tag_vocab.bin', 'wb'))
model_char = CharRNN(len(vocab), args.n_units, args.n_layers)
serializers.load_npz(args.init_from, model_char)

def generate_embeddings(data, length):
    whole_len    = data.shape[0]
    batch_data = np.array(data, dtype=np.int32)
    batch_data = batch_data.reshape((whole_len,1))
    word_emb = {}
    for i in range(whole_len-1):
        if batch_data[i][0]==vocab[' ']:
            word_emb[len(word_emb)] = state['c{0:d}'.format(args.n_layers-1)].data

        state, y = model_char.forward_one_step(batch_data[i], batch_data[i], train_dev=False, train = False)

    print('Length word_emb {}, lenght tags {}'.format(len(word_emb),length))
    t = open('data/en/final_word_emb_{}'.format(length),'wb')
    pickle.dump(word_emb,t)
    t.close()
    print('embeddings trained and saved')
    return word_emb



def train_tagger(n_units, n_layers, learning_rate):
    data, tags, tag_vocab   = load_data(args, vocab)


    if args.type == 'lstm':
        log_name                = 'lstm_model_{}_{}_{}.log'.format(n_layers, n_units, learning_rate)
        model                   = POStaggerLSTM(len(set(list(tag_vocab.keys()))),n_units,n_layers)
    else:
        log_name                = 'final_basic_model_{}_{}_{}.log'.format(n_layers, n_units, learning_rate)
        model                   = POStaggerBasic(len(set(list(tag_vocab.keys()))),n_units,n_layers)
    log                     = open('%s/%s'%(args.checkpoint_dir,log_name), 'w')


    optimizer               = optimizers.MomentumSGD(learning_rate)
    optimizer.setup(model)
    curr_dev_loss           = 0
    prev_dev_loss           = 0

    ivocab = {}
    for c, i in tag_vocab.items():
        ivocab[i] = c

    try:
        word_emb_dict = pickle.load(open('data/en/final_word_emb_{}'.format(len(tags)), 'rb'))
        print('loaded embeddings')
    except:
        print('training embeddings...')
        word_emb_dict = generate_embeddings(data, (len(tags)))

    word_emb = [word_emb_dict[i] for i in range(len(word_emb_dict)) if i<(len(tags))]
    word_emb = np.array(word_emb)

    tags                    = tags[:-1]
    batchsize               = 10
    bprop_len               = 5
    whole_len               = word_emb[:int(len(tags)*0.9)].shape[0]
    jump                    = whole_len // batchsize
    epoch                   = 0
    accum_loss              = Variable(np.zeros((), dtype=np.float32))
    total_loss, train_accuracy = 0.0,0.0

    for i in range(jump * 1000):
        x_batch1 = np.array([word_emb[:int(len(tags)*0.9)][(jump * j + i+1) % whole_len]
                            for j in range(batchsize)])
        x_batch2 = np.array([tags[:int(len(tags)*0.9)][(jump * j + i) % whole_len]
                            for j in range(batchsize)])
        y_batch = np.array([tags[:int(len(tags)*0.9)][(jump * j + i + 1) % whole_len]
                            for j in range(batchsize)])


        prob, loss_i, accu = model.forward_one_step(x_batch1,x_batch2, y_batch)
        train_accuracy += sum([1 for item in accu if item])
        accum_loss   += loss_i

        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            now = time.time()
            #print ('{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)//bprop_len, jump, accum_loss.data / bprop_len, now-cur_at))
            cur_at = now

            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            total_loss += accum_loss.data/ bprop_len

            accum_loss = Variable(np.zeros((), dtype=np.float32))

            optimizer.clip_grads(5)
            optimizer.update()

        if (i + 1) % jump == 0:
            accum_dev_loss, dev_accuracy = 0.0,0.0

            prev_tag = len(tag_vocab)-1
            for j,tag in enumerate(tags[int(len(tags)*0.9):],int(len(tags)*0.9)):
                prob, loss,accu = model.forward_one_step(word_emb[j],np.array([prev_tag],dtype=np.int32),np.array([tag], dtype=np.int32),train_dev=True)
                if accu: dev_accuracy+=1
                accum_dev_loss += loss
                prev_tag = (np.argmax(cuda.to_cpu(prob.data)))
                #print('dev',ivocab[prev_tag])


            print('Training loss {} on epoch {}'.format(total_loss/len(tags[:int(len(tags)*0.9)]),epoch))
            log.write('Training loss {} on epoch {}\n'.format(total_loss/len(tags[:int(len(tags)*0.9)]),epoch))
            print('Training accuracy', train_accuracy/len(tags[:int(len(tags)*0.9)]))
            log.write('Training accuracy: {}\n'.format(train_accuracy/len(tags[:int(len(tags)*0.9)])))
            print('Dev loss',accum_dev_loss.data/len(tags[int(len(tags)*0.9):]))
            log.write('Dev loss: {}\n'.format(accum_dev_loss.data/len(tags[int(len(tags)*0.9):])))
            f = 'cv/{}_{}_{}_{}_{}.chainermodelnew'.format(args.type,n_units, n_layers,learning_rate,epoch)
            serializers.save_npz(f, model)
            prev_prev_dev_loss = prev_dev_loss
            prev_dev_loss = curr_dev_loss
            curr_dev_loss = accum_dev_loss.data/len(tags[int(len(tags)*0.9):])
            print('Dev accuracy',dev_accuracy/len(tags[int(len(tags)*0.9):]) )
            log.write('Dev accuracy: {}\n'.format(dev_accuracy/len(tags[int(len(tags)*0.9):]) ))

            print('_____________________________________________')
            epoch += 1
            total_loss = 0.0
            train_accuracy = 0.0
            if epoch >= 10:
                optimizer.lr *= 0.97

            if prev_dev_loss != 0 and curr_dev_loss >= prev_dev_loss and prev_dev_loss >= prev_prev_dev_loss:
                break
    print ('Done')



train_tagger(256,3,0.001)
