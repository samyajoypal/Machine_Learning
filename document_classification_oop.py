# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 02:04:42 2021

@author: Samyajoy
"""

import pandas as pd
import numpy as np
import os, re, sys
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from langdetect import detect
from nltk.stem.cistem import Cistem
from string import punctuation
import matplotlib.pyplot as plt
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

def umlauts(word):
    """
    Replace umlauts for a given text
    
    :param word: text as string
    :returns manipulated text as str
    """
    
    tempVar = word # local variable
    
    # Using str.replace() 
    
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('Ä', 'Ae')
    tempVar = tempVar.replace('Ö', 'Oe')
    tempVar = tempVar.replace('Ü', 'Ue')
    tempVar = tempVar.replace('ß', 'ss')
    
    return tempVar



def NLP(data):
    """
Standard Natual Language Processing

:Detects Language (Currently Supports Only German and English)
:Removes Punctuations
:Removes numbers
:removes stop words depending on language
:Does Stemming based on language
:Returns Colums of Text After Standard Natual Language Processing

"""
    lang=detect(data[0])
    remove_pun = str.maketrans('', '', punctuation)
    if (lang=='de'):
        stemmer = Cistem()
        stop_words = stopwords.words('german')
        Stemmed_Content=[]
        for row in data:
            temp1=[]
            row=row.translate(remove_pun)
            row=row.translate({ord(ch): None for ch in '0123456789'})
            for i in sent_tokenize(row):
                temp = [] 
                for j in word_tokenize(i):
                    if j not in stop_words:
                        tm=umlauts(j)
                        temp.append(stemmer.stem(tm))
                s=" ".join(temp)
                temp1.append(s)
            temp2=''.join([str(elem) for elem in temp1])
            Stemmed_Content.append(temp2)
            
    else:
        stop_words=nltk.corpus.stopwords.words('english')
        ps=nltk.PorterStemmer()
        Stemmed_Content=[]
        for row in data:
            temp1=[]
            row=row.translate(remove_pun)
            row=row.translate({ord(ch): None for ch in '0123456789'})
            for i in sent_tokenize(row): 
                temp = [] 
                for j in word_tokenize(i): 
                    temp.append(ps.stem(j))
                s=" ".join(temp)
                temp1.append(s)
            temp2=''.join([str(elem) for elem in temp1])
            Stemmed_Content.append(temp2)
            
    print("Standard NLP Techniques Are Suffessfully Done")
    return Stemmed_Content

class doc_classification:
    def __init__(self, X, Y): 
        """
    Takes input of covariates and response variable
    
    :param X: Covariates (Column of Texts)
    :param Y: Response Variable (Column of Classes)
    """
        self.X=X
        self.Y=Y
        self.Stemmed_Content=self.X
        
    def preprocess(self,data):
        self.Stemmed_Content=NLP(data)


        
    def supervised_model_fit(self):
        """
    Supervised Learning Model Fitting
    
    :Vectorizes each rows by TFIDF vectorizer
    :Splits Data randomely into training and test data
    :Uses five classifiers viz. Naive Bayes, Multinomial Naive Bayes, Random Forest,
     Linear Classifier (SVM) and SVM with non-linear kernel
    :For each classifier model is fitted on train data and predition is done on 
     training data to check accuracy.
    :training time, prediction time, accuracy is noted
    :Display of other metrices like confusion matrix can be enabled in each section if required
    :Cross validation and F1 scores are calculated and noted
    :Returns all the noted results
    
    """
        
        ##Use TFIDF to determine the covariates and theirr values
        corpus=self.Stemmed_Content
        vectorizer = TfidfVectorizer()
        tfidf_matrix=vectorizer.fit_transform(corpus).todense()
        tfidf_names=vectorizer.get_feature_names()
        print("Number of TFIDF Features: %d"%len(tfidf_names)) #same info can be gathered by using tfidf_matrix.shape
        ##Classifiers to be used 

        training_time_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}
        prediction_time_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}
        accuracy_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}
        variables = tfidf_matrix
        labels = self.Y
        self.variables_train, self.variables_test, self.labels_train, self.labels_test  =   train_test_split(variables, labels, test_size=.3)
        #splitting the data into random training and test sets for both independent variables and labels.
        #analyzing the shape of the training and test data-set:
        print('Shape of Training Data: '+str(self.variables_train.shape))
        print('Shape of Test Data: '+str(self.variables_test.shape))
        
        ##Applying Naive Bayes

        training_time_container.keys()
        
        #loading Gaussian Naive Bayes from the sklearn library:
        self.bnb_classifier=BernoulliNB()
        #initializing the object
        t0=time()
        self.bnb_classifier=self.bnb_classifier.fit(self.variables_train,self.labels_train)
        training_time_container['b_naive_bayes']=time()-t0
        #fitting the classifier or training the classifier on the training data
        
        #after the model has been trained, we proceed to test its performance on the test data:
        t0=time()
        self.bnb_predictions=self.bnb_classifier.predict(self.variables_test)
        prediction_time_container['b_naive_bayes']=time()-t0
        
        prediction_time_container['b_naive_bayes']
        
        #the trained classifier has been used to make predictions on the test data-set. To evaluate the performance of the model,
        #there are a number of metrics that can be used as follows:
        nb_ascore=sklearn.metrics.accuracy_score(self.labels_test, self.bnb_predictions)
        accuracy_container['b_naive_bayes']=nb_ascore
        
#        print("Bernoulli Naive Bayes Accuracy Score: %f"%accuracy_container['b_naive_bayes'])
#        print("Training Time: %f"%training_time_container['b_naive_bayes'])
#        print("Prediction Time: %f"%prediction_time_container['b_naive_bayes'])
#        
#        print("Confusion Matrix of Bernoulli Naive Bayes Classifier output: ")
#        sklearn.metrics.confusion_matrix(labels_test,bnb_predictions)
#        
#        print("Classification Metrics: ")
#        print(sklearn.metrics.classification_report(labels_test,bnb_predictions))
        print("Naive Bayes Model Fitting Done Sucessfully")
        
        
        ##Applying Multinomial Naive Bayes:

        self.mn_bayes=MultinomialNB()
        t0=time()
        self.mn_bayes_fit=self.mn_bayes.fit(self.variables_train,self.labels_train)
        training_time_container['mn_naive_bayes']=time()-t0
        t0=time()
        self.prediction_mn=self.mn_bayes_fit.predict(self.variables_test)
        prediction_time_container['mn_naive_bayes']=time()-t0
        mn_ascore=sklearn.metrics.accuracy_score(self.labels_test, self.prediction_mn) 
        accuracy_container['mn_naive_bayes']=mn_ascore
        
#        print("Accuracy Score of Multi-Nomial Naive Bayes: %f" %(mn_ascore))
#        #and its training and prediction time are:
#        print("Training Time: %fs"%training_time_container['mn_naive_bayes'])
#        print("Prediction Time: %fs"%prediction_time_container['mn_naive_bayes'])
        
        print("Multinomial Naive Bayes Model Fitting Done Sucessfully")
        
        ##Applying Random Forest Classifier:

        self.rf_classifier=RandomForestClassifier(n_estimators=50)
        t0=time()
        self.rf_classifier=self.rf_classifier.fit(self.variables_train,self.labels_train)
        
        training_time_container['random_forest']=time()-t0
#        print("Training Time: %fs"%training_time_container['random_forest'])
        
        t0=time()
        self.rf_predictions=self.rf_classifier.predict(self.variables_test)
        prediction_time_container['random_forest']=time()-t0
#        print("Prediction Time: %fs"%prediction_time_container['random_forest'])
        
        accuracy_container['random_forest']=sklearn.metrics.accuracy_score(self.labels_test, self.rf_predictions)
#        print ("Accuracy Score of Random Forests Classifier: ")
#        print(accuracy_container['random_forest'])
#        print(sklearn.metrics.confusion_matrix(labels_test,rf_predictions))
        
        print("Random Forest Model Fitting Done Sucessfully")
         
         ##Applying Linear Classifier (SVM) using Stochastic Gradient Descent:


        self.svm_classifier=linear_model.SGDClassifier(loss='hinge',alpha=0.0001)
        
        t0=time()
        self.svm_classifier=self.svm_classifier.fit(self.variables_train, self.labels_train)
        training_time_container['linear_svm']=time()-t0
#        print("Training Time: %fs"%training_time_container['linear_svm'])
        
        t0=time()
        self.svm_predictions=self.svm_classifier.predict(self.variables_test)
        prediction_time_container['linear_svm']=time()-t0
#        print("Prediction Time: %fs"%prediction_time_container['linear_svm'])
        
        accuracy_container['linear_svm']=sklearn.metrics.accuracy_score(self.labels_test, self.svm_predictions)
#        print ("Accuracy Score of Linear SVM Classifier: %f"%accuracy_container['linear_svm'])
#        print(sklearn.metrics.confusion_matrix(labels_test,svm_predictions))
        
        self.svm_classifier_enet=linear_model.SGDClassifier(loss='hinge',alpha=0.0001,penalty='elasticnet')
        self.svm_classifier_enet=self.svm_classifier_enet.fit(self.variables_train, self.labels_train)
        
        self.svm_enet_predictions=self.svm_classifier_enet.predict(self.variables_test)
        
#        print ("Accuracy Score of Linear SVM Classifier: %f"%sklearn.metrics.accuracy_score(labels_test,svm_enet_predictions))
        print("Linear Classifier (SVM) using Stochastic Gradient Descent Model Fitting Done Sucessfully")
        
        
        ##Applying SVM with non-linear kernel

        self.nl_svm_classifier=SVC(C=1000000.0, gamma= 'auto', kernel='rbf')
        
        t0=time()
        self.nl_svm_classifier=self.nl_svm_classifier.fit(self.variables_train,self.labels_train)
        training_time_container['non_linear_svm']=time()-t0
        
        t0=time()
        self.nl_svm_predictions=self.nl_svm_classifier.predict(self.variables_test)
        prediction_time_container['non_linear_svm']=time()-t0
        
        accuracy_container['non_linear_svm']=sklearn.metrics.accuracy_score(self.labels_test,self.nl_svm_predictions)
        
#        print("Accuracy score of Non-Linear SVM: %f"%accuracy_container['linear_svm'])
        print("SVM with non-linear kernel Model Fitting Done Sucessfully")
        
        
        
        self.clslabels=['b_naive_bayes', 'mn_naive_bayes', 'random_forest', 'linear_svm', 'non_linear_svm']
        self.classifiers=[self.bnb_classifier,self.mn_bayes,self.rf_classifier,self.svm_classifier,self.nl_svm_classifier]
        
        #Results:
        
        self.Training_Time=pd.DataFrame(list(zip(self.clslabels, training_time_container.values())), columns =['Classifier', 'Training Time'])
        
        self.Prediction_Time=pd.DataFrame(list(zip(self.clslabels, prediction_time_container.values())), columns =['Classifier', 'Prediction Time'])
        
        self.Accuracy=pd.DataFrame(list(zip(self.clslabels, accuracy_container.values())), columns =['Classifier', 'Accuracy'])
        
        
        
    def train_time(self):
        return self.Training_Time
    
    def pred_time(self):
        return self.Prediction_Time
    
    def accuracy(self):
        self.k=self.Accuracy['Accuracy'].index(max(self.Accuracy['Accuracy']))
        return self.Accuracy
    
    def cross_val(self):
        cv_scores=[]
        cv_scores_f1=[]
        for i in range(len(self.classifiers)):
            t1=self.classifiers[i].fit(self.variables_train,self.labels_train)
            cv_scores_temp = sklearn.model_selection.cross_val_score(t1, self.variables_train, self.labels_train, cv=5).mean()
            cv_scores.append(cv_scores_temp)
            cv_scores_f1_temp=sklearn.model_selection.cross_val_score(t1, self.variables_train,self.labels_train,cv=5,scoring='f1_weighted').mean()
            cv_scores_f1.append(cv_scores_f1_temp)
            
        self.Cross_Validation=pd.DataFrame({'Classifier': self.clslabels, 'Accuracy': cv_scores, 'F1 Scores': cv_scores_f1})
        
        return self.Cross_Validation
    
    def predict(self,new_X, k):
        '''
        Predicts the classes for new data by fitted model
        
        :param new_X: column of a data frame, where each row is string
        :param k: takes input of any of 0,1,2,3,4 for
         bnb_classifier,mn_bayes,rf_classifier,svm_classifier,nl_svm_classifier respectively
        :Alternatively pram k can be removed from argument and self.k can be used directly to use
         the best classifier
        
        '''
        pr_data=NLP(new_X)
        vectorizer = TfidfVectorizer()
        tfidf_matrix=vectorizer.fit_transform(pr_data).todense()
        variables = tfidf_matrix
        return self.classifiers[k].predict(variables)
        
        