
"""Imports"""
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold
import gc
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sb
import math
import os
import re
import glob
from tqdm.notebook import tqdm
import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import config


"""Loading test and training files from the folders (train and test separately) 
where datafiles for 10 Fold CV are stored"""
train_path = r'/path/to/10_fold/train/folder'
test_path = r'/path/to/10_fold/test/folder'
all_train_files = glob.glob(train_path + "/*.csv")
all_test_files = glob.glob(test_path + "/*.csv")

train_texts=[]
train_labels= []
test_texts=[]
test_labels = []
for i in range(0, len(all_train_files)):
  train_file = pd.read_csv(all_train_files[i],encoding='utf-8')
  test_file = pd.read_csv(all_test_files[i],encoding='utf-8')
  train_texts.append(train_file['text'].tolist())
  train_labels.append(train_file['labels'].tolist())
  test_texts.append(test_file['Description'].tolist())
  test_labels.append(test_file['Label'].tolist())

"""Constant setup"""
K = 10 #K=10 for 10-Fold CV
rootOutputDir = "/root/output/dir/" #change here for each run, i.e. Run1, Run2
outputDir = rootOutputDir+config.model_type

"""Functions of interest"""
def makeDatasetDFs(textList, labelList):
  ll=[]
  for i in range(len(textList)):
    ll.append([textList[i],labelList[i]])
  return pd.DataFrame(ll,columns=['text','labels'])

def modelEval(FoldIndex,predictions):
    df_test = makeDatasetDFs(test_texts[FoldIndex], test_labels[FoldIndex])
    acc = accuracy_score(df_test['labels'],predictions)
    recall = recall_score(df_test['labels'],predictions)
    f1 = f1_score(df_test['labels'],predictions)
    precision = precision_score(df_test['labels'],predictions)
    tn, fp, fn, tp = confusion_matrix(df_test['labels'],predictions).reshape(-1)
    
    if FoldIndex == 0:
        resultFile = pd.DataFrame([],columns=['Fold No', 'Accuracy','Recall','Precision','F1_Score','TP','TN','FP','FN'])
    else:
       resultFile = pd.read_csv(rootOutputDir+'Results.csv',encoding='utf-8')
    
    data = {'Fold No': str(FoldIndex), 'Accuracy':acc, 'Recall':recall, 'Precision':precision, 'F1_Score':f1,
          'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn }
    resultFile=resultFile.append(data,ignore_index=True,sort=False)
    resultFile.to_csv(rootOutputDir+'Results.csv', index=False, encoding='utf-8')

def trainModel(modelType, modelName):
    cuda_available = torch.cuda.is_available()
    

    for FoldIndex in range(0,K):
        modelFileName = rootOutputDir+"model_fold"+str(FoldIndex)+'.pkl'
        df_train = makeDatasetDFs(train_texts[FoldIndex], train_labels[FoldIndex])
        
        train_args= config.train_args
        train_args["logging_steps"] = math.ceil(len(train_texts[FoldIndex])/8)
        train_args["best_model_dir"] = modelFileName
        train_args["output_dir"] = outputDir

        model = ClassificationModel(modelType, modelName, args=train_args, use_cuda=cuda_available)
        history = model.train_model(df_train)
        print(history)
        pickle.dump(model, open(modelFileName, 'wb'))
        predictions, raw_outputs = model.predict(df_train['text'].tolist())
        modelEval(FoldIndex, predictions)
    
    with open(rootOutputDir+'hyperparameters.txt', 'w') as file:
        file.write(json.dumps(train_args)) 

if __name__ == "__main__":
    trainModel(modelType=config.model_type, modelName=config.model_name)












