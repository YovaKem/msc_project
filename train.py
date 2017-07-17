#%%
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
from chainer import cuda, Variable, FunctionSet, optimizers, serializers
import chainer.functions as F
from CharRNN import CharRNN

# input data
def load_data(args):
    vocab = {}
    print ('%s/train_input.txt'% args.data_dir)
    words = codecs.open('%s/train_input.txt' % args.data_dir, 'rb', 'utf-8').read()
    #check if this doesn't split my data on spaces

    if args.param_opt:
        length = len(list(words))
        #take 20% of the words
        words = list(words)[length//1000:length//500]+list(words)[length//200:length//200+length//1000]
    else:
        words = list(words)
    dataset_train = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset_train[i] = vocab[word]
    vocab['UNK'] = len(vocab)

    words_dev = codecs.open('%s/dev_input.txt' % args.data_dir, 'rb', 'utf-8').read()
    words_dev = list(words_dev)
    dataset_dev = np.ndarray((len(words_dev),), dtype=np.int32)
    for i, word in enumerate(words_dev):
        if word in vocab:
            dataset_dev[i] = vocab[word]
        else:
            dataset_dev[i] = vocab['UNK']

    print ('corpus length:', len(words))
    print('dev set len:', len(words_dev))
    print ('vocab size:', len(vocab))

    return dataset_train, dataset_dev, words, vocab

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',                   type=str,   default='data/en')
parser.add_argument('--checkpoint_dir',             type=str,   default='cv_chnaged_x_data')
parser.add_argument('--gpu',                        type=int,   default=-1)
#parser.add_argument('--rnn_size',                   type=int,   default=128)
#parser.add_argument('--learning_rate',              type=float, default=2e-3)
parser.add_argument('--learning_rate_decay',        type=float, default=0.97)
parser.add_argument('--learning_rate_decay_after',  type=int,   default=10)
parser.add_argument('--decay_rate',                 type=float, default=0.95)
#parser.add_argument('--dropout',                    type=float, default=0.0)
#parser.add_argument('--seq_length',                 type=int,   default=30)
#parser.add_argument('--batchsize',                  type=int,   default=50)
#parser.add_argument('--epochs',                     type=int,   default=50)
#parser.add_argument('--grad_clip',                  type=int,   default=5)
#if you want to continue training a pre-trained model
parser.add_argument('--init_from',                  type=str,   default='')
parser.add_argument('--param_opt',                  type=int,   default=1)

args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

#n_epochs    = args.epochs
#n_units     = args.rnn_size
#batchsize   = args.batchsize
#bprop_len   = args.seq_length
#grad_clip   = args.grad_clip
#dropout     = args.dropout
#learning_rate = args.learning_rate
decay_rate = args.decay_rate
learning_rate_decay_after = args.learning_rate_decay_after
learning_rate_decay = args.learning_rate_decay

train_data, dev_data, words, vocab = load_data(args)
pickle.dump(vocab, open('%s/vocab.bin'%args.data_dir, 'wb'))

def compute_loss(train, n_epochs,n_layers, n_units, batchsize,grad_clip, dropout,learning_rate, bprop_len):
    if train:
        data = train_data
    else:
        data = dev_data
    latest_string = 'latest'
    if len(args.init_from)>0:
        fn_latest1 =  ('%s/%s'%(args.checkpoint_dir,args.init_from))
    else:
        for item in [n_epochs,n_layers, n_units, batchsize,grad_clip, dropout,learning_rate, bprop_len]:
            latest_string = latest_string + '_'+str(item)
    #fn_latest = ('%s/%s.chainermodel'%(args.checkpoint_dir,latest_string))
        fn_latest1 = ('%s/%s.chainermodelnew'%(args.checkpoint_dir,latest_string))
    model = CharRNN(len(vocab), n_units, n_layers)
    serializers.load_npz(fn_latest1, model)

    accum_loss   = Variable(np.zeros((), dtype=np.float32))
    whole_len    = data.shape[0]
    jump         = whole_len // batchsize
    for i in range(jump):
        x_batch = np.array([data[(jump * j + i) % whole_len]
                            for j in range(batchsize)])
        y_batch = np.array([data[(jump * j + i + 1) % whole_len]
                            for j in range(batchsize)])
        state, loss_i = model.forward_one_step(x_batch, y_batch, dropout_ratio=0)
        accum_loss   += loss_i

    return accum_loss.data / jump

def train_LM(n_epochs,n_layers, n_units, batchsize,grad_clip, dropout,learning_rate, bprop_len):
    log_name = 'train{}_{}_{}_{}_{}_{}_{}.log'.format(n_layers, n_units, batchsize,grad_clip,dropout, learning_rate, bprop_len)
    log = open('%s/%s'%(args.checkpoint_dir,log_name), 'w')

    if len(args.init_from) > 0:
        model = CharRNN(len(vocab), n_units, n_layers)
        serializers.load_npz(args.init_from, model)
    else:
        model = CharRNN(len(vocab), n_units, n_layers)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.RMSprop(lr=learning_rate, alpha = decay_rate, eps=1e-8)
    optimizer.setup(model)
    #not_changed = 1
    #new_training_loss = 7

    loss_log     = defaultdict()
    whole_len    = train_data.shape[0]
    jump         = whole_len // batchsize
    epoch        = 0
    start_at     = time.time()
    cur_at       = start_at
    #state        = make_initial_state(n_units, batchsize=batchsize)
    if args.gpu >= 0:
        accum_loss   = Variable(cuda.zeros(()))
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)
    else:
        accum_loss   = Variable(np.zeros((), dtype=np.float32))
    total_loss = 0
    loss_update = 0
    curr_dev_loss = 0
    prev_dev_loss = 0
    print ('going to train {} iterations'.format(jump * n_epochs))
    for i in range(jump * n_epochs):
        x_batch = np.array([train_data[(jump * j + i) % whole_len]
                            for j in range(batchsize)])
        y_batch = np.array([train_data[(jump * j + i + 1) % whole_len]
                            for j in range(batchsize)])

        if args.gpu >=0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        state, loss_i = model.forward_one_step(x_batch, y_batch, dropout_ratio=dropout)
        accum_loss   += loss_i

        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            now = time.time()
            #print ('{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)//bprop_len, jump, accum_loss.data / bprop_len, now-cur_at))
            cur_at = now

            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            total_loss += accum_loss.data/ bprop_len
            loss_update+=1
            if args.gpu >= 0:
                accum_loss = Variable(cuda.zeros(()))
            else:
                accum_loss = Variable(np.zeros((), dtype=np.float32))

            optimizer.clip_grads(grad_clip)
            optimizer.update()

        #if (i + 1) % 10000 == 0:
            #pass
            #fn = ('%s/charrnn_epoch_%.2f.chainermodel' % (args.checkpoint_dir, float(i)//jump))

            #pickle.dump(copy.deepcopy(model).to_cpu(), open('%s/%s.chainermodel'%(args.checkpoint_dir,latest_string), 'wb'))


        if (i + 1) % jump == 0:
            #fn1= ('%s/charrnn_epoch_%.2f_%d.chainermodelnew' % (args.checkpoint_dir, float(i)//jump,epoch))
            #serializers.save_npz(fn1, model)
            #pickle.dump(copy.deepcopy(model).to_cpu(), open(fn, 'wb'))
            latest_string = 'latest'
            for item in [n_epochs,n_layers, n_units, batchsize,grad_clip, dropout,learning_rate, bprop_len]:
                latest_string = latest_string + '_'+str(item)

            #fn_latest = ('%s/%s.chainermodel'%(args.checkpoint_dir,latest_string))
            fn_latest1 = ('%s/%s.chainermodelnew'%(args.checkpoint_dir,latest_string))
            serializers.save_npz(fn_latest1, model)

            #prev_training_loss = new_training_loss
            print('Training loss:{} on epoch {}\n'.format(total_loss/loss_update,epoch))
            #new_training_loss = total_loss/loss_update
            log.write('Training loss:{} on epoch {}\n'.format(total_loss/loss_update,epoch))
            prev_prev_dev_loss = prev_dev_loss
            prev_dev_loss = curr_dev_loss
            curr_dev_loss= compute_loss(0,n_epochs,n_layers, n_units, batchsize,grad_clip, dropout,learning_rate, bprop_len)
            print('Development loss:{} on epoch {}\n'.format(curr_dev_loss,epoch))
            log.write('Development loss:{} on epoch {}\n'.format(curr_dev_loss,epoch))
            loss_log[(epoch)] = (total_loss/loss_update)
            total_loss = 0
            loss_update = 0
            epoch += 1

            if curr_dev_loss>prev_dev_loss and prev_dev_loss>prev_prev_dev_loss:
                break

            if epoch >= learning_rate_decay_after:
                optimizer.lr *= learning_rate_decay
                print ('decayed learning rate by a factor {} to {}'.format(learning_rate_decay, optimizer.lr))

        sys.stdout.flush()
    log.write(str(loss_log).split(',',1)[1])

train_LM(100, 1, 256, 50, 5, 0.2, 0.003, 30)
#print(compute_loss(0,50, 2, 128, 64, 5, 0.2, 0.005, 30))
