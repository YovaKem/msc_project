import numpy as np
from chainer import Variable, Chain
import chainer.functions as F
import chainer.links as L
from collections import defaultdict

class CharRNN(Chain):

    def __init__(self, n_vocab, n_units, nlayers_enc):
        super(CharRNN, self).__init__()
        #embed = F.EmbedID(n_vocab, n_units),
        self.add_link("embed_enc", L.EmbedID(n_vocab, n_units))
        self.lstm_enc = ["L{0:d}_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))
        self.add_link("out", L.Linear(n_units, n_vocab))
        self.nlayers_enc = nlayers_enc

    def reset_state(self):
        for lstm_name in self.lstm_enc:
            self[lstm_name].reset_state()
        self.loss = 0


    def forward_one_step(self, x_data, y_data, train=True, train_dev = True,dropout_ratio=0):
        #x = Variable(x_data, volatile=not train)
        #t = Variable(y_data, volatile=not train)
        x=x_data
        t=y_data
        state = {}

        embed_id = self.embed_enc(x_data)
        embed_id = F.dropout(embed_id, ratio = dropout_ratio, train = train_dev)
        # feed into first LSTM layer
        hs = self[self.lstm_enc[0]](embed_id)
        hs = F.dropout(hs, ratio = dropout_ratio, train = train_dev)
        # feed into remaining LSTM layers
        for lstm_layer in self.lstm_enc[1:]:
            hs = self[lstm_layer](hs)
            #hs = F.dropout(hs, ratio = dropout_ratio, train = train)

        y = self.out(F.dropout(self[self.lstm_enc[-1]].h, ratio = dropout_ratio, train = train_dev))

        for i,lstm_name in zip(range(self.nlayers_enc),self.lstm_enc):
            state['c{0:d}'.format(i)] = self[lstm_name].c
            state['h{0:d}'.format(i)] = self[lstm_name].h

        if train:
            return state, F.softmax_cross_entropy(y, t)
        else:
            return state, F.softmax(y)

#def make_initial_state(n_units, batchsize=50, train=True):
#    return {name: Variable(np.zeros((batchsize, n_units), dtype=np.float32),
#            volatile=not train)
#            for name in ('c', 'h')}
