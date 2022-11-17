#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 23:20:10 2022

@author: diego
"""

import os
import pandas as pd
import numpy as np

#%%

rootpath="/home/diego/Documents/EDUCATION/LEARNING/IT/MACHINE_LEARNING/ML_Handson_2022/workspace/1/wsp1/"
datapath = os.path.join(rootpath, "datasets/housing/")
outpath = os.path.join(rootpath, "out/2_end2end/")
codepath = os.getcwd

#%% load dataset

def load_housing_data(datapath):
    csv_path = os.path.join(datapath, "housing.csv")
    return pd.read_csv(csv_path)

#%% display model score parameters

def display_scores(scores):
    print("\nScores:", scores)
    print("Mean:", scores.mean())
    print("Std. dev.:", scores.std())
    
    #%%  define function for model testing  
# model_name are strings; examples: # "LinearRegression"  "RandomForestRegressor"


def run_model(model_name, X, Y):
    from sklearn.model_selection import cross_val_score # K-fold cross validation
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR   
    import joblib
    
    print("\n RUNNING", model_name, "WITH CROSS VALIDATION 10 ON HOUSING DATASET")
    skmodel = model_name + "()"
    print(skmodel)
    model = eval(skmodel)
    print(model)
    model.fit(X, Y)
    scores = cross_val_score(model, X, Y, 
                                scoring="neg_mean_squared_error", cv=10)
    rmse = np.sqrt(-scores)
    print("\n RMSE of cross-validations:")
    display_scores(rmse)
    #print("Labels description:")
    #print(Y_train.describe())
    means_ratio = rmse.mean()/Y.mean()
    print("\n Mean RMSE / Labels mean = ",means_ratio, "\n")
    # Export model to file:
    fname = outpath + model_name +".pkl"
    joblib.dump(model, fname)
    return [rmse.mean(), rmse.std(), means_ratio]

      