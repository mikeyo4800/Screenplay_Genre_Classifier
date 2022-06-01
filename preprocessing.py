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

from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer




class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, activator_type = None, lem_or_stem = None, stop_words = None):
        
        self.activator_type = activator_type
        self.lem_or_stem = lem_or_stem
        self.stop_words = stop_words
    
    def fit(self, data, y = 0):
        return self
    
    def transform(self, data, y = 0):
        
        if self.lem_or_stem == 'lem':
            fully_normalized_corpus = data.apply(self.lem_process_doc)
            return fully_normalized_corpus
        
        elif self.lem_or_stem == 'stem':
            fully_normalized_corpus = data.apply(self.stem_process_doc)
            return fully_normalized_corpus

    def the_cleaner(self, text):
        
        text_replace = text.replace('\\r', ' ').replace('\\n', ' ').replace('\\', '').split()
        text_strip = [re.sub(r"\([^()]*\)", "", i) for i in text_replace]
        perfect = " ".join([x for x in text_strip if x.isalpha() and len(x) > 1])

        return perfect
    
    def lem_process_doc(self, doc):


        stop_words = stopwords.words('english')
        movie_stop_words = ['b', 'fade', 'in', 'cut', 'to', 'int', 'ext', 'v', 'o', '\b', 'out', 'transition', 'to', 'angle', 'pan', 'word', 'title', 'description', 'screenplay']
        stop_words_modified = list(itertools.chain(movie_stop_words, stop_words))
        doc_lower = self.the_cleaner(doc).lower()


        if self.stop_words != None:
            stop_words_modified = list(itertools.chain(stop_words_modified, self.stop_words))
        
        if self.activator_type == 'wb':
            
            sent = TextBlob(doc_lower)
            tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
            words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
            lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
            return " ".join(lemmatized_list)


        elif self.activator_type == 'pt':
            
            doc_norm = [tok for tok in word_tokenize(doc_lower) if ((tok.isalpha()) & (tok not in stop_words_modified))]
            return " ".join([lemma(tok) for tok in doc_norm])

        elif self.activator_type == 'wnl':
            wnl = WordNetLemmatizer()

            def pos_tagger(nltk_tag):
                if nltk_tag.startswith('J'):
                    return wordnet.ADJ
                elif nltk_tag.startswith('V'):
                    return wordnet.VERB
                elif nltk_tag.startswith('N'):
                    return wordnet.NOUN
                elif nltk_tag.startswith('R'):
                    return wordnet.ADV
                else:         
                    return None


            doc_norm = [tok for tok in word_tokenize(doc_lower) if ((tok.isalpha()) & (tok not in stop_words_modified)) ]


            wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tag(doc_norm))) 
            doc_norm = [wnl.lemmatize(token, pos) for token, pos in wordnet_tagged if pos is not None]

            return " ".join(doc_norm)

    def stem_process_doc(self, doc):

        stop_words = stopwords.words('english')
        movie_stop_words = ['b', 'fade', 'in', 'cut', 'to', 'int', 'ext', 'v', 'o', '\b', 'out', 'transition', 'to', 'angle', 'pan', 'word', 'title', 'description', 'screenplay']
        stop_words_modified = list(itertools.chain(movie_stop_words, stop_words))
        doc_lower = self.the_cleaner(doc).lower()
        
        if self.stop_words != None:
            stop_words_modified = list(itertools.chain(stop_words_modified, self.stop_words))
        
        doc_norm = [tok for tok in word_tokenize(doc_lower) if ((tok.isalpha()) & (tok not in stop_words_modified))]

        if self.activator_type == 'ps':

            ps = PorterStemmer()
            stem_doc = [ps.stem(x) for x in doc_norm]
            return " ".join(stem_doc)

        elif self.activator_type == 'ls':

            ls = LancasterStemmer()
            stem_doc = [ls.stem(x) for x in doc_norm]
            return " ".join(stem_doc)

        elif self.activator_type == 'ss':
            sn = SnowballStemmer('english')
            stem_doc = [sn.stem(x) for x in doc_norm]
            return " ".join(stem_doc)


