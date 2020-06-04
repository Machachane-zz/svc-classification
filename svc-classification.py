# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 02:56:27 2020

@author: Machachane
"""

from sklearn.datasets import fetch_20newsgroups
import numpy as np
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import timeit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import timeit
from sklearn.pipeline import Pipeline


groups = fetch_20newsgroups()

data_train = fetch_20newsgroups(subset='train', random_state=21)
train_label = data_train.target
data_test = fetch_20newsgroups(subset='test', random_state=21)
test_label = data_test.target
print('\nTrain data length, Test data length, Test label lenth:\n',len(data_train.data), len(data_test.data), len(test_label))

print('\nNp unique:\n',np.unique(test_label))


all_names = names.words()
WNL = WordNetLemmatizer()
def clean(data):
    cleaned = defaultdict(list)
    count = 0
    for group in data:
        for words in group.split():
            if words.isalpha() and words not in all_names:
                cleaned[count].append(WNL.lemmatize(words.lower()))
        cleaned[count] = ' '.join(cleaned[count])
        count+=1
    return(list(cleaned.values()))

x_train = clean(data_train.data)

print('\nX train:\n', x_train[0])

print('\nSVC ------------------------------------------------------------------------')

svc_lib = SVC(kernel = 'linear')
parameters = {'C' : (0.5,1.0,10,100)}


grid_search1 = GridSearchCV(svc_lib, parameters, n_jobs = -1, cv = 3)
start_time = timeit.default_timer()
grid_search1.fit(X_train, train_label)
final = timeit.default_timer()-start_time
print('\nExecution Time:\n ', final)

print(grid_search1.best_params_)
print(grid_search1.best_score_)

grid_search_best1 = grid_search1-best_estimator_
accur1 = grid_search_best1.score(X_test, test_label)
print('\nAccur:\n', accur1)

print('\nLinear SVC ------------------------------------------------------------------------')

linear_svc = LinearSVC()
parameters = {'C': (0.5, 1, 10, 100)}

grid_search2 = GridSearchCV(linear_svc, parameters, n_jobs = -1, cv = 3)
start_time = timeit.default_timer()
grid_search2.fit(X_train, train_label)
final = timeit.default_timer()-start_time
print('\nExecution Time:\n', final)

print('\nGrid search2 best params:\n',grid_search2.best_params_)
print('1nGrid search2 best score:\n', grid_search2.best_score_)

grid_search_best2 = grid_search2.best_estimator_
accur2 = grid_search_best2.score(X_test, test_label)

print('\nAccur2:\n', accur2)


print('\nModel Tuning -> Linear SVC ------------------------------------------------------------------------')

pipeline = Pipeline([('tf_id', TfidfVectorizer(stop_words = 'english')), ('svm_im', LinearSVC())])
pipeline

parameter = {'tf_id__max_features' : (100,1000,2000,8000), 'tf_id__max_df':(0.25, 0.5), 'tf_id__smooth_idf': (True, False), 'tf_id__sublinear_tf': (True, False)}

grid_search = GridSearchCV(pipeline, parameter, cv = 3)
print('\nGrid search fit:\n ', grid_seach.fit(c_train, train_label)
      
print('\nGrid search best params:\n', grid_search.best_params_)
print('\nGrid search best score:\n', grid_search.best_score_)
