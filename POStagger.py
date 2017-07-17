import numpy as np
from chainer import Variable, Chain
import chainer.functions as F
import chainer.links as L
from collections import defaultdict

class POStagger(Chain):

    def __init__(self, n_vocab, n_units, nlayers_enc):
        super(POStagger, self).__init__()
        #embed = F.EmbedID(n_vocab, n_units),
        self.layers = ["L{0:d}_enc".format(i) for i in range(nlayers_enc)]
        self.add_link(self.layers[0], L.Linear(256,n_units))
        for name in self.layers[1:]:
            self.add_link(name, L.Linear(n_units, n_units))

        self.add_link("out", L.Linear(n_units, n_vocab))
        self.nlayers_enc = nlayers_enc


    def forward_one_step(self, x_data1,x_data2, y_data, train=True, sample=False, train_dev = False):
        x1 = Variable(x_data1, volatile = not train)
        #x2 = Variable(x_data2, volatile = not train)
        t = Variable(y_data, volatile = not train)
        hs_lstm = self[self.layers[0]](x1)
        # feed into remaining LSTM layers
        for layer in self.layers[1:]:
            hs_lstm = self[layer](F.tanh(hs_lstm))
        #hs_lin = self.linear(x1)
        y = self.out(F.tanh(hs_lstm))
        #y = self.out(F.concat((F.tanh(hs_lin),F.dropout(hs_lstm))))
        if train and not train_dev:
            return F.softmax(y),F.softmax_cross_entropy(y, t), [np.argmax(m) == n for (m,n) in zip(F.softmax(y).data,y_data)]
        if train and train_dev:
            return F.softmax(y),F.softmax_cross_entropy(y, t), np.argmax(F.softmax(y).data) == y_data[0]
        else:
            return F.softmax(y)
