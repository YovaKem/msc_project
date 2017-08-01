import operator
import _pickle as pickle
import numpy as np

top_contexts = pickle.load(open('results_dict/pseudowords_new','rb'))
#new = open('top_10_context.log','w')
for top in top_contexts:
    print(top,top_contexts[top][4])
