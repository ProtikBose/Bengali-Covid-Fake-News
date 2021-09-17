""" EDIT THE FILE PATHS ACCORDINGLY """

"""Imports"""
from keras import callbacks
from keras.models import Sequential
from keras.layers import Activation,Flatten,Dense,Dropout,Embedding,Bidirectional,LSTM
from keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,Bidirectional,SpatialDropout1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import KFold
import gc
import numpy as np
import keras.backend as K
import numpy as np
import pandas as pd
import seaborn as sb
import math
import os
import re

"""Preprocessing Functions"""

git clone https://github.com/banglakit/bengali-stemmer.git

cd "/content/bengali-stemmer"

from bengali_stemmer.rafikamal2014 import RafiStemmer
stemmer = RafiStemmer()

pip install bnlp_toolkit

from bnlp.corpus import stopwords, punctuations
stopwordsBNLP = stopwords


from bnlp.corpus.util import remove_stopwords
from bnlp import NLTKTokenizer

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

""" Data Loading and preparation"""

dataset= pd.read_csv('path/to/whole_dataset_shuffled.csv')
print("df1.shape: ",dataset.shape)

def memoryUsage(df, dfName):
  df.info(memory_usage="deep")
  print(f"\nNon null rows count of {dfName}\n")
  df.isnull().sum()

memoryUsage(dataset, "datset")


columnsToDrop = []
for column in dataset.columns:
  if column != 'Message' and column != 'Description' and column != 'Label' and column != 'Link Text':
    columnsToDrop.append(column)


def DropCols(df):
  df.drop(columnsToDrop, axis=1, inplace=True)
  

DropCols(dataset)



def forcingType(df, cols):
  for col in cols:
    df[col] = df[col].astype(str)

forcingType(dataset, ['Message','Description', 'Link Text'])



"""
prepend message to Description
"""
def mergingCols(df):
  print("message: ",df['Message'][20]," description: ",df['Description'][20], "link text: ", df['Link Text'][20])
  df['Description'] =  df['Message'] +" "+df['Link Text']+" "+ df['Description']
  print("mixed: ",df['Description'][20])

mergingCols(dataset)

"""
Removing duplicate rows
"""
def removeDuplicateRows(df):
  print("Unique Content :",len(list(set(df['Description']))))
  print("Total rows :",len(df)) 
  df.drop_duplicates(subset ="Description", 
                      keep = "first", inplace = False)

removeDuplicateRows(dataset)

def preprocess(df):
  preprocessed=cleaning(df.copy())
  preprocessed = preprocessed.apply(lambda x: "".join(" "+item for item in x))
  return preprocessed

preprocessed_dataset = preprocess(dataset['Description'])


def makeDatasetDFs(text, labels):
  ll=[]
  for i in range(len(text)):
    ll.append([text[i],labels[i]])
  return pd.DataFrame(ll,columns=['text','labels'])

df_dataset = makeDatasetDFs(preprocessed_dataset, dataset['Label'])



df_dataset.to_csv("path/to/whole_dataset_shuffled_preprocesed.csv", 
                  index=False, encoding='utf-8')


X = df_dataset.text
y = df_dataset.labels

""" StratifiedKFold """

from sklearn.model_selection import StratifiedKFold, KFold

train_path = r'to/path/Stratified K Fold/Training Set/'
test_path = r'to/path/Stratified K Fold/Test Set/'

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

df_train = []
df_test = []

i=1
for train_index, test_index in skf.split(X, y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    train = pd.DataFrame()
    test = pd.DataFrame()
    test['Description'] = x_test_fold
    test['Label'] = y_test_fold
    train['Description'] = x_train_fold
    train['Label'] = y_train_fold
    df_train.append(train)
    df_test.append(test)
    train.to_csv(train_path + 'train'+ str(i) + '.csv', index=False, encoding='utf-8')
    test.to_csv(test_path + 'test'+ str(i) + '.csv', index=False, encoding='utf-8')
    i=i+1
    del train, test


