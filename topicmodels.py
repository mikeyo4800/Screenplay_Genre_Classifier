from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models






class Modeling(BaseEstimator, TransformerMixin):
    
    def __init__(self, topics, chunk, passes):
        
        #define attributes to store if text preprocessing requires fitting from data
        self.topics = topics
        self.chunk = chunk
        self.passes = passes
        self.corpus = None
        self.dictionary = None
        self.texts = None
        self.lda_model = None
        self.cv_score = None
        self.vis = None
        self.words_per_doc = None
    
    def fit(self, data, y = 0):
        # this is where you would fit things like corpus specific stopwords
        # fit probable bigrams with bigram model in here
        
        # save as parameters of Text preprocessor
        return self

    def id2word_creation(self, doc, mod= False):

        if mod == False:
            sentence_split = self.clean_sentences(doc)
            self.dictionary = corpora.Dictionary([sentence_split])
        elif mod == True:
            self.dictionary = corpora.Dictionary(doc)

        return self.dictionary

    def clean_sentences(self, doc):
        
        sentences_cleaned = doc.split()
        return sentences_cleaned

    def get_texts(self, doc, mod=False):

        if mod == False:
            sentence_split = self.clean_sentences(doc)
            self.texts = list(sentence_split)
        elif mod == True:
            self.texts = list(doc)

        return self.texts

    def corpus_creation(self):

        self.corpus = [self.dictionary.doc2bow(x) for x in self.texts]

        return self.corpus

    def create_model(self, data):
        
        sentences_cleaned = data.apply(self.clean_sentences)
        id2word = self.id2word_creation(sentences_cleaned, mod=True)
        texts = self.get_texts(sentences_cleaned, mod = True)
        corpus = self.corpus_creation()
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=self.topics, update_every=1, chunksize=self.chunk, passes=self.passes, alpha='auto', per_word_topics=True)

        return self.lda_model

    def visualize_model(self):
        
        pyLDAvis.enable_notebook()
        self.vis = pyLDAvis.gensim_models.prepare(self.lda_model, self.corpus, self.dictionary)
        
        return self.vis

    
    def transform(self, data, y=0):
        
        lda_model = self.create_model(data)

        #self.words_per_doc = lda_model.get_document_topics(bow=)

        dct = {}

        for i in range(self.topics):
            dct[i] = []

        for i in range(len(self.corpus)):
            topics = lda_model.get_document_topics(self.corpus[i], minimum_probability=0.0)
            for x, y in enumerate(topics):
                dct[x].append(y[1])

        coherence_model = CoherenceModel(model=lda_model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
        self.cv = coherence_model.get_coherence()

        prop_df = pd.DataFrame(dct, index=data.index)



        print("CV Score: {}".format(self.cv) )
        return prop_df
