import collections
import pandas as pd
from nltk import FreqDist

def word_counter(corpuses, num, idx=None, pos=None):

    top_ten = {}

    for i in corpuses.keys():

        oh_no = collections.Counter([x for x in corpuses[i].split()])
    
        words = []
        counts = []
    
        if idx == None:
        
            for word, count in oh_no.most_common(num):
                words.append(word)
                counts.append(count)

            top_ten['words {}'.format(i)] = words
            top_ten['counts {}'.format(i)] = counts


    else:

        if pos == 1:
            
            for word, count in oh_no.most_common(num)[idx:]:
                words.append(word)
                counts.append(count)

            top_ten['words {}'.format(i)] = words
            top_ten['counts {}'.format(i)] = counts
        
        else:
            
            for word, count in oh_no.most_common(num)[:idx]:
                words.append(word)
                counts.append(count)

            top_ten['words {}'.format(i)] = words
            top_ten['counts {}'.format(i)] = counts



    return top_ten


def word