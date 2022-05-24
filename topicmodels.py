from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
import itertools
import pattern
from pattern.en import lemma, lexeme

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import WordNetLemmatizer

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models





class Modeling(BaseEstimator, TransformerMixin):
    
    def __init__(self, topics):
        
        #define attributes to store if text preprocessing requires fitting from data
        self.topics = topics
    
    def fit(self, data, y = 0):
        # this is where you would fit things like corpus specific stopwords
        # fit probable bigrams with bigram model in here
        
        # save as parameters of Text preprocessor
        return self

    def clean_sentences(self, doc):
        
        sentences_cleaned = doc.split()
        return sentences_cleaned
    
    def transform(self, data, y=0):
        
        sentences_cleaned = data.apply(self.clean_sentences)
        id2word = corpora.Dictionary(sentences_cleaned)
        texts = list(sentences_cleaned)
        corpus = [id2word.doc2bow(x) for x in texts]
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=self.topics, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

        dct = {}

        for i in range(self.topics):
            dct[i] = []

        for i in range(len(corpus)):
            topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
            for x, y in enumerate(topics):
                dct[x].append(y[1])


        prop_df = pd.DataFrame(dct, index=data.index)
        return prop_df
