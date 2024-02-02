README.txt

This is a sample code written by me for training the "California Housing Data Set". The main directions for the pipelines are taken from [2] but implemented in scripts rather than in jupyter notebooks.  

Currently the code does:
1. Generate training and data sets using stratified shuffling on the attribute "median_income". A summary of the training data can be found in ml_housing/out/2_end2end/train_data_summary.csv and an overview in the jupyter notebook ml_housing/code/visualize_nb.ipynb.

2. Data preprocessing using scikit-learn pipelines: 
    a. Fill missing numerical data with "median" strategy
    b. Feature scaling numerical data
    c. Generate numerical attributes from the only categorical one ("ocean_proximity") using OneHotEncoder
No PCA or similar algorithm is applied.

3. Training on training set:
try_models = ['LinearRegression', 'RandomForestRegressor', 'SVR']
SVR stands for Support Vector Regressor
These 3 models are trained using the standard hyperparameters of scikit-learn and using cross validation with 10 subsets.
The comparison of the training results can be seen under:
ml_housing/out/2_end2end/model_comparison.csv

4. Hyperparameter fine tuning is performed for selected models: in this case 'RandomForestRegressor' and 'SVR'. The tuned parameters are:
RandomForestRegressor_pars = [{'n_estimators': [3, 30, 300, 1000],
                   'max_features': [2, 3, 4, 5, 6, 7, 8, 9]
                   }]
    
SVR_pars = [{'kernel': ['linear', 'poly', 'rbf'],
             'C': [0.6, 1, 2], # The strength of the regularization is inversely proportional to C
             'gamma': ['scale', 'auto'] # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
             }]

5. One of these models is used for making predictions on the test set. The scores are found in: 
ml_housing/out/2_end2end/best_model_rmse_test.csv

All the models are exported as .pkl objects in order to enable further analysis and improvement; however those files are not available in github
because of the large storage requirements (magnitudes in the order 100MB-1G).

ToDo:
The same data set will be processed with tensorflow using a single GPU. This work is ongoing and will be published as soon as it is done.

References:
[1] https://developers.google.com/machine-learning/crash-course/california-housing-data-description
[2] Géron, Aurélien. Hands-On Machine Lerarning with Scikit-Learn, Keras and TensorFlow. Second Edition, 2019, O'Reilly Media Inc.
[3] Ng. Andrew. Machine Learning. Course offered by Stanford University on coursera.org. This course is archived and has been replaced.See https://www.coursera.org/specializations/machine-learning-introduction
