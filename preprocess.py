"""Preprocess dataset and make ready for feeding to models"""

#clone the following stemmer from git : git clone https://github.com/banglakit/bengali-stemmer.git
from bengali_stemmer.rafikamal2014 import RafiStemmer
#pip install bnlp toolkit : pip install bnlp_toolkit
from bnlp.corpus import stopwords, punctuations
from bnlp.corpus.util import remove_stopwords
from bnlp import NLTKTokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import pickle
import gc
import seaborn as sb
import math
import os
import re
import glob

"""Setup constants"""
all_news_path = '/path/to/all_news/'

"""Preprocessing Functions"""
def removeForeign(word):
  a = "".join(i for i in word if 2432 <= ord(i) <= 2559)
  return a

def makeRemoveHyperLink(text):
  result = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
  return result

def MakeHTMLremove(text):
  '''
  result = re.compile('(<.*?>)')
  result = result.sub(r'', text) 
  '''
  cleancode = re.compile('<code>.*?</code>')
  cleanr = re.compile('<.*?>')
  cleanentity = re.compile('&.*;')
  cleantext = re.sub(cleancode, '', text)
  cleantext = re.sub(cleanr, ' ', cleantext)
  cleantext = re.sub(cleanentity, ' ', cleantext)
  
  return cleantext

def cleaning(updated):

    stopwordsBNLP = stopwords
    # print(len(stopwordsBNLP))
    # print(punctuations)
    stemmer = RafiStemmer()
  
    #html remove
    updated= updated.apply(lambda x: MakeHTMLremove(x))
    #hyperlink remove
    updated = updated.apply(lambda x: makeRemoveHyperLink(x))
    # tokenizer
    bnltk = NLTKTokenizer()
    updated = updated.apply(lambda x: bnltk.word_tokenize(x))
    # remove punctuations
    updated = updated.apply(lambda x: [item for item in x if item not in punctuations])
    # remove stop words
    updated = updated.apply(lambda x: [item for item in x if item not in stopwordsBNLP])
    # remove foreign words
    updated = updated.apply(lambda x: [ removeForeign(item) for item in x ])
    # stripping
    updated = updated.apply(lambda x: [item.strip() for item in x ])
    # remove numbers
    updated = updated.apply(lambda x: [re.sub(r'[০১২৩৪৫৬৭৮৯\.]+', '', item) for item in x ])
    # stemming
    updated = updated.apply(lambda x: [stemmer.stem_word(item) for item in x ])
    # stripping
    updated = updated.apply(lambda x: [item.strip() for item in x ])
    
    
    return updated

def concatenateDF(all_news_files):
    all_news_df = pd.read_csv(all_news_files[0],encoding='utf-8')
    for i in range(1, len(all_news_files)):
        tempDf = pd.read_csv(all_news_files[i],encoding='utf-8')
        all_news_df = pd.concat([all_news_df, tempDf], ignore_index=True) 
    return all_news_df

def memoryUsage(df, dfName):
  df.info(memory_usage="deep")
  print(f"\nNon null rows count of {dfName}\n")
  df.isnull().sum()


def makeDropColumnsList(all_news_df, columnsToDrop):
    for column in all_news_df.columns:
        if column != 'Message' and column != 'Description' and column != 'Link Text':
            columnsToDrop.append(column)
    return columnsToDrop

def DropCols(df, columnsToDrop):
  df.drop(columnsToDrop, axis=1, inplace=True)
  df.head(10)

def forcingType(df, cols):
  for col in cols:
    df[col] = df[col].astype(str)


def mergingCols(df):
  df['Description'] =  df['Message'] +" "+df['Link Text']+" "+ df['Description']


def removeDuplicateRows(df):
  print("Unique Content :",len(list(set(df['Description']))))
  print("Total rows :",len(df)) 
  df.drop_duplicates(subset ="Description", 
                      keep = "first", inplace = False)

def preprocess():

    all_news_files = glob.glob(all_news_path + "/*.csv")
    all_news_df = concatenateDF(all_news_files)

    memoryUsage(all_news_df, "dataset")

    columnsToDrop = []
    columnsToDrop = makeDropColumnsList(all_news_df, columnsToDrop)
    DropCols(all_news_df, columnsToDrop)
    forcingType(all_news_df, ['Message','Description', 'Link Text'])
    mergingCols(all_news_df)
    removeDuplicateRows(all_news_df)

    preprocessed = cleaning(all_news_df.copy())
    preprocessed = preprocessed.apply(lambda x: "".join(" "+item for item in x))
    return preprocessed

