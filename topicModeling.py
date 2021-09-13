import preprocess
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import gc
import seaborn as sb
import math
import re
import glob
import os
import logging
import warnings
import gensim
import numpy as np
from gensim.models import CoherenceModel, LdaModel
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary

"""Config Setup"""
category_folder = '/path/to/store/categorized/news/'
chunksize = 500 
passes = 40
iterations = 400
eval_every = 1  

def buildTopicModel(data, numTopic):
    
    # create dictionary
    dictionary = Dictionary(data)
    dictionary.filter_extremes(no_below=10, no_above=0.2)
    #Create dictionary and corpus required for Topic Modeling
    corpus = [dictionary.doc2bow(doc) for doc in data]
    temp = dictionary[0]  # only to "load" the dictionary.
    id2word = dictionary.id2token
    
    lda = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=numTopic, \
                       passes=passes, eval_every=eval_every)
    # Compute Coherence Score using c_v
    coherenceModel = CoherenceModel(model=lda, texts=data, dictionary=dictionary, coherence='c_v')

    return lda, corpus, id2word, coherenceModel

if __name__ == "__main__":
    preprocessed_dataDF = preprocess.preprocess()
    postList = preprocessed_dataDF.tolist()
    wordList = [data.split() for data in postList]

    with open(category_folder+'wordList.json', 'w', encoding='utf-8') as f:
        json.dump(wordList, f, ensure_ascii=False)