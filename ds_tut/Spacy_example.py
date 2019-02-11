#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:05:12 2018


@author: travisbarton
"""

import spacy
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

nlp = spacy.load('en')


Data = pd.read_csv("/Users/emersonsjsu/GitHub/movie_classification/ds_tut/VDifferentData.csv")
# Data contains everything
Data = Data.iloc[:, 1:]

vectors = np.empty([Data.shape[0], 385])
# sents is list of word embeddings for post titles
sents = []
for i in range(Data.shape[0]):
    sents.append(nlp(Data.iloc[i,0]).vector)

# Convert list of lists into 2d matrix
for i in range(Data.shape[0]):
    for j in range(384):
        vectors[i, j] = sents[i][j]

vectors[:,384] = Data.iloc[:,1]
# vectors is same format as sents, but now a np array


X_train, X_test, y_train, y_test = train_test_split(
        vectors[:,:384], vectors[:,384], test_size=0.33, random_state=42)


# gamma=scale decreases gamma, which makes line more jagged and prone to overfitting
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

preds.astype(int)

print(sum(preds.astype(int) == y_test)/len(preds)*100)

