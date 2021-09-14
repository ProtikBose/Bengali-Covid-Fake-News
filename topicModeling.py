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
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter

"""Config Setup"""
category_folder = '/path/to/store/categorized/news/'
chunksize = 500 
passes = 40
iterations = 400
eval_every = 1  

"""Functions of interest"""
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

def findBestModel(dataset, topicNumList):
    modelList = []
    coherenceList = []

    for topicNum in topicNumList:
        lda, corpus, id2word, coherenceModel = buildTopicModel(dataset, topicNum)
        # print(lda.print_topics())
        coherence = coherenceModel.get_coherence()
        coherenceList.append(coherence)
        modelList.append(lda)
        print("Topic Number:", topicNum)
        print("coherence score: ", coherence)
    
    # Show graph
    plt.plot(topicNumList, coherenceList)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    #find best model
    bestIdx = coherenceList.index(max(coherenceList))
    bestCoherence = max(coherenceList)
    print(bestIdx)
    print(bestCoherence)
    print(topicNumList[bestIdx])

    pickle.dump(modelList[bestIdx],open(category_folder + 'topicNum_' + str(topicNumList[bestIdx]) + '.pkl', 'wb'))
    return modelList[bestIdx]

def predict(model, wordList, all_news_df):
    
    dictionary = Dictionary(wordList)
    dictionary.filter_extremes(no_below=10, no_above=0.2)
    #Create dictionary and corpus required for Topic Modeling
    corpus = [dictionary.doc2bow(doc) for doc in wordList]

    topicId = []
    topicProb = []
    for i in range(0,len(corpus)):
        probs = model.get_document_topics(corpus[i])
        topicId.append(max(probs,key=itemgetter(1))[0])
        topicProb.append(max(probs,key=itemgetter(1))[1])
    
    for i in range(len(all_news_df)):
        all_news_df.loc[i, 'Topic Id'] = topicId[i]
        all_news_df.loc[i, 'Topic Prob'] = topicProb[i]
    all_news_df.to_csv(category_folder+ 'CategorizedWithProb.csv', encoding='utf-8')

if __name__ == "__main__":
    preprocessed_dataDF = preprocess.preprocess()
    postList = preprocessed_dataDF.tolist()
    wordList = [data.split() for data in postList]

    with open(category_folder+'wordList.json', 'w', encoding='utf-8') as f:
        json.dump(wordList, f, ensure_ascii=False)

    topicNumList = [5, 8, 10, 12, 15, 17, 20, 23 ,25, 30, 35, 40, 45, 50]
    bestModel = findBestModel(wordList, topicNumList)
    predict(bestModel, wordList, preprocessed_dataDF)