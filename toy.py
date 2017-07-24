import operator
import _pickle as pickle
import numpy as np

real_pref = pickle.load(open('results_dict/words_match_Real_pref_Real_base','rb'))
pseudo_pref = pickle.load(open('results_dict/words_match_Pseudo_pref_Real_base','rb'))
overlap = {}
overlapping_words = {}
non_overlapping_words = {}
exclusively_matching = {}
counts = {'NOUN':6,'VERB':9,'ADJ':3}
for i in pseudo_pref.keys():
    #overlap[i] = [1 for word in set(pseudo_pref[i]) if word in real_pref[i]]
    overlapping_words[i]  = [word for word in set(pseudo_pref[i]) if word in real_pref[i]]
    #non_overlapping_words[i] = [word for word in set(real_pref[i]) if word not in pseudo_pref[i]]
    exclusively_matching[i] = [1 for word in set(pseudo_pref[i]) if pseudo_pref[i].count(word)==counts[i]-1 and real_pref[i].count(word)==counts[i]-1]
    #print(i)
    #print(sum(overlap[i])/len(set(pseudo_pref[i])))
    print(sum(exclusively_matching[i])/len(set(pseudo_pref[i])))
    #print(exclusively_matching[i])
print(overlapping_words['VERB'])
#print(np.mean([len(word) for cat in overlapping_words.keys() for word in overlapping_words[cat] ]))

#print(non_overlapping_words)
#print(np.mean([len(word) for cat in non_overlapping_words.keys() for word in non_overlapping_words[cat] ]))
