# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:03:12 2021

@author: Samyajoy
"""
# import all the required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline


class autosap:
    def __init__(self, X, Y):
        """
    Takes input of covariates and response variable
    
    :param X: Covariates (Column of DataFrame)
    :param Y: Response Variable (Column of Response Variable in Data Frame)
    """
            
        self.X=X
        self.Y=Y


    def fselect_fit(self):
        """
    Recursive Feature Elimination with Cross Validation
    
    :selects optimal number of features and also the best model depending on accuracy
    :It uses Logistic Regression, Decision Tree, Random Forest and XBG Classifier
    """
        # create pipeline of differennt base algorithms to be used in RFECV (no. of features will be auto-selected based on cv in RFECV)
        self.models = {}
        # logistic regression
        rfecv = RFECV(estimator = LogisticRegression(), cv = 10, scoring = 'accuracy')
        model = DecisionTreeClassifier()
        self.models['LR'] = Pipeline(steps = [('features', rfecv), ('model', model)])
        # decision tree
        rfecv = RFECV(estimator = DecisionTreeClassifier(), cv = 10, scoring = 'accuracy')
        model = DecisionTreeClassifier()
        self.models['DT'] = Pipeline(steps = [('features', rfecv), ('model', model)])
        # random forest
        rfecv = RFECV(estimator = RandomForestClassifier(), cv = 10, scoring = 'accuracy')
        model = DecisionTreeClassifier()
        self.models['RF'] = Pipeline(steps = [('features', rfecv), ('model', model)])
        # XGBoost Classifier
        rfecv = RFECV(estimator=XGBClassifier(), cv = 10, scoring = 'accuracy')
        model = DecisionTreeClassifier()
        self.models['XGB'] = Pipeline(steps = [('features', rfecv), ('model', model)])
        
        # evaluate all the models
        self.results = []
        self.names = []
        for name, model in self.models.items():
            cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
            scores = cross_val_score(model, self.X, self.Y, scoring = 'accuracy', cv = cv, n_jobs = -1)
            self.results.append(np.mean(scores))
            self.names.append(name)
            print('>%s: %.3f' % (name, np.mean(scores)))
        self.models_df=pd.DataFrame(self.models)
        
        self.pred_models=[LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),XGBClassifier()]
        
    def accuracy(self):
        self.acc= pd.DataFrame(list(zip(self.names, self.results)), columns =['Model', 'Accuracy'])
        self.k=self.acc['Accuracy'].index(max(self.acc['Accuracy']))
        
        return self.acc
    def predict(self,new_X):
        '''
        Predicts the classes for new data by fitted model
        
        :param new_X: column of a data frame, where each row is string
        :It automatically uses the best model to do prediction
        
        '''
        # create pipeline
        rfecv = RFECV(estimator = self.pred_models[self.k], cv = 10, scoring = 'accuracy')
        model = DecisionTreeClassifier()
        pipeline = Pipeline(steps=[('features', rfecv), ('model', model)])
        # fit the model on all available data
        pipeline.fit(self.X, self.Y)
        # make a prediction for one example
        yhat = pipeline.predict(new_X)
        return yhat