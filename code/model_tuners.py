#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:09:24 2022

@author: diego
"""
import os
import numpy as np
from sklearn.model_selection import cross_val_score # K-fold cross validation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR   
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
rootpath="/workspace/1/wsp1/"
outpath = os.path.join(rootpath, "out/2_end2end/")

#%% Random tuner:
def random_tuner (model_name, X, Y, params):
    skmodel = model_name + "()"
    model = eval(skmodel)
    #model.fit(X, Y)
    rand_search = RandomizedSearchCV(model, param_distributions=params, 
                                     n_iter=7, cv=5, 
                                     scoring='neg_mean_squared_error',
                                     return_train_score=True, refit=True)
    rand_search.fit(X, Y)
    # print cv scores:
    cvres = rand_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    # print best estimator:
    print("\n\n Best parameters for ", model_name, ":\n")
    print(rand_search.best_params_)
    # print best estimator:
    print("\n\n Best estimator for ", model_name, ":\n")
    print(rand_search.best_estimator_)
    # Export model to file:
    fname = outpath + model_name +"_best.pkl"
    joblib.dump(rand_search.best_estimator_, fname)
    return rand_search.best_estimator_
#%%
