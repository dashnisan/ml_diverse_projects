#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 00:21:19 2022

@author: diego
"""
import functions
import model_tuners
import model_classes
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
print("Scikit-Learn version:", sklearn.__version__ )

#%%
rootpath="/workspace/1/wsp1/"
datapath = os.path.join(rootpath, "datasets/housing/")
outpath = os.path.join(rootpath, "out/2_end2end/")
codepath = os.getcwd
#%% generate  sets:
full_set = functions.load_housing_data(datapath)

#define the strati as a new attribute in the dataset (must be in the dataset):
full_set["income_cat"] = pd.cut(full_set["median_income"],
                     bins=[0.,1.5,3.0,4.5,6.,np.inf],
                     labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(full_set, full_set["income_cat"]):
    train_set = full_set.loc[train_index] 
    test_set = full_set.loc[test_index]

# export summary of train set data to file:
train_set_summary = train_set.describe().transpose()
file = outpath + "train_data_summary.csv"
train_set_summary.to_csv(file, header=True, index=True, encoding='utf-8')
print("\n Wrote file", file, "to disk")

# check result of stratified spliting: get the ration of counts of data popints with given category to total amount:
ratios_test_set = test_set["income_cat"].value_counts() / len(test_set)
ratios_full_set = full_set["income_cat"].value_counts() / len(full_set)

print("Stratification ratios using median_income:")
print("Test set:")
print(ratios_test_set)
print("Full set:")
print(ratios_full_set)

# remove income_cat attribute from housing for training:
for set_ in (train_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)
#%% DATA PREPROCESSING:
# Separate predictors (attributes) from labels:
X_train = train_set.drop("median_house_value", axis=1)
y_train = train_set["median_house_value"].copy()

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()
print("Success: Separate predictors from labels: training and test sets")

# Get list of numerical and categorical attributes:
all_att = list(X_train)
num_att = all_att
num_att.remove("ocean_proximity")
cat_att=["ocean_proximity"]
#%% Construct transformation pipeline for preparation of numerical data:
num_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("feat_scale", StandardScaler()),        
    ])
#%%
# Get the categories including th ones of the OneHotEncoder:    
cat_encoder = OneHotEncoder()
X_train_cat = X_train[["ocean_proximity"]]
X_train_cat_enc = cat_encoder.fit_transform(X_train_cat)
enco_cats = cat_encoder.categories_
a = enco_cats[0]
#allplus_att = num_att + a
allplus_att = [y for x in [num_att, a] for y in x]

#%% Transformation pipeline for numerical and categorical data:
numcat_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_att),
        ("cat", OneHotEncoder(), cat_att),
        ])

# Transform train-set attributes:
X_train_trf = numcat_pipeline.fit_transform(X_train)

# Copy numpy array to pandas Dataframe for visualizing transformation:
X_train_trf_df = pd.DataFrame(X_train_trf, columns=allplus_att, index=X_train.index)

# write dataframe to file:
stats_X_train_trf = X_train_trf_df.describe()

file = outpath + "stats_X_train_trf.csv"
stats_X_train_trf.to_csv(file, header=True, index=True, encoding='utf-8')
print("\n Wrote file", file, "to disk")

#%% run selected models on training dataset with standard parameters:
#try_models = []
try_models = ['LinearRegression', 'RandomForestRegressor', 'SVR']
nrows, ncols = (len(try_models),3) 
model_compare = [[None]*ncols]*nrows
i = 0
for model in try_models:
    print(" \n RUNNING MODEL ", model, "WITH STANDARD PARAMETERS")
    outpars = functions.run_model(model, X_train_trf, y_train)
    model_compare[i] = outpars
    i += 1

#%% parse model results into pandas dataframe
model_compare_cols = ['CROSS VAL RMSE MEAN','CROSS VAL RMSE STD','RMSE MEAN / Y MEAN']
model_results = pd.DataFrame(model_compare, columns=model_compare_cols, index=try_models)

#%% print model scores comparison to file
model_results.insert = (0, "Model", try_models)
file = outpath + "model_comparison.csv"
model_results.to_csv(file, header=True, index=True, encoding='utf-8')
print("\n Wrote file", file, "to disk")

#%% Models Fine Tuning: 
tune_models = ['RandomForestRegressor', 'SVR']
# parameter list must be named: <modelname>_pars:
RandomForestRegressor_pars = [{'n_estimators': [3, 30, 300, 1000],
                   'max_features': [2, 3, 4, 5, 6, 7, 8, 9]
                   }]
    
SVR_pars = [{'kernel': ['linear', 'poly', 'rbf'],
             'C': [0.6, 1, 2], # The strength of the regularization is inversely proportional to C
             'gamma': ['scale', 'auto'] # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
             }]

# create instances of models to tune and run the models:
for model in tune_models:
    print("\n\n Running model tuner for ", model)
    dyn_varname_obj = model+"_obj"
    dyn_varname_pars = model+"_pars"
    vars()[dyn_varname_obj] = model_classes.model(model, eval(dyn_varname_pars), None)
    vars()[dyn_varname_obj].best_estimator = model_tuners.random_tuner(model,
                              X_train_trf, y_train,
                              eval(dyn_varname_obj+".tune_pars"))
#%% Select manually best model from fine tuned ones and predict on test set:
best_model = RandomForestRegressor_obj.best_estimator
X_test_trf = numcat_pipeline.fit_transform(X_test)
test_pred = best_model.predict(X_test_trf)
test_mse = mean_squared_error(y_test, test_pred)
test_rmse = np.sqrt(test_mse)
confidence = 0.95
squared_errors = (test_pred - y_test)**2
test_rmse_conf = np.sqrt(stats.t.interval(confidence,
                         len(squared_errors)-1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
print("\n RESULTS SELECTED BEST MODEL ON TEST SET:\n")
print("\n Selected model: ", best_model)
print("\n RMSE = ", test_rmse)
print("\n RMSE within confidence interval ", confidence, " = ", test_rmse_conf)

#%% Export prediction results on test set to files:
header = ['RMSE 95% CONFIDENCE '+ RandomForestRegressor_obj.name + 
          ' Best Estimator']
test_results = pd.DataFrame(test_rmse_conf, columns=header, index=None)
file = outpath + "best_model_rmse_test.csv"
test_results.to_csv(file, header=True, index=False, encoding='utf-8')
#%% features' importance:
header = ['Importance']
matrix = np.matrix(best_model.feature_importances_)
test_results = pd.DataFrame(matrix, columns=allplus_att)
file = outpath + "best_model_feat_RelImportance.csv"
test_results.to_csv(file, header=True, index=False, encoding='utf-8')
