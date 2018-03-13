# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import opts
import dataHelper
#refer to "https://zhuanlan.zhihu.com/p/26729228"
opt = opts.parse_opt()
import dataHelper as helper
train_iter, test_iter = dataHelper.loadData(opt,embedding=False)
#categories = ['good', 'bad', 'mid']
x_train,y_train=train_iter
x_test,y_test = test_iter

#opt.model ="haha"
if opt.model == "bayes":
    """ Naive Bayes classifier """
    # sklearn有一套很成熟的管道流程Pipeline，快速搭建机器学习模型神器
    bayes_clf = Pipeline([('vect', CountVectorizer()), 
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultinomialNB()) 
                          ])
    bayes_clf.fit(x_train, y_train)
    """ Predict the test dataset using Naive Bayes"""
    predicted = bayes_clf.predict(x_test)
    print('Naive Bayes correct prediction: {:4.4f}'.format(np.mean(predicted == y_test)))
    # 输出f1分数，准确率，召回率等指标
#    print(metrics.classification_report(y_test, predicted, target_names=categories))
elif opt.model == "svm":
    
    """ Support Vector Machine (SVM) classifier"""
    svm_clf = Pipeline([('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)),
    ])
    svm_clf.fit(x_train, y_train)
    predicted = svm_clf.predict(x_test)
    print('SVM correct prediction: {:4.4f}'.format(np.mean(predicted == y_test)))
#    print(metrics.classification_report(y_test, predicted, target_names=categories))
    
else:
    """ 10-折交叉验证 """
    clf_b = make_pipeline(CountVectorizer(), TfidfTransformer(), MultinomialNB())
    clf_s= make_pipeline(CountVectorizer(), TfidfTransformer(), SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter= 5, random_state=42))
    
    bayes_10_fold = cross_val_score(clf_b, x_test, y_test, cv=10)
    svm_10_fold = cross_val_score(clf_s, x_test, y_test, cv=10)
    
    print('Naives Bayes 10-fold correct prediction: {:4.4f}'.format(np.mean(bayes_10_fold)))
    print('SVM 10-fold correct prediction: {:4.4f}'.format(np.mean(svm_10_fold)))
# 输出混淆矩阵
#print("Confusion Matrix:")
#print(metrics.confusion_matrix(y_test, predicted))
#print('\n')




