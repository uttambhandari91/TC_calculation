

import pandas as pd

dataset = pd.read_csv('/content/alloy composition TC.csv')

dataset.head()

dataset. columns

dataset.describe()

#Import required modules
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import sklearn as sk
import numpy as np
from decimal import *
import math
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import sklearn.metrics
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 0)

rand_state = 1    # Shuffle the dataset
train_dataset = dataset.sample(frac=0.9,random_state=rand_state)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("TC")
train_stats = train_stats.transpose()
# Stats and distribution of testing data
test_stats = test_dataset.describe()
test_stats.pop("TC")
test_stats = test_stats.transpose()
# Define target properties in training set
y_train_TC = train_dataset.pop("TC")      # thermalConductivity

# Define target properties in testing set
y_test_TC = test_dataset.pop("TC")      # C11

# Normalization
def norm_train(x):
  return (x - train_stats['mean']) / train_stats['std']
X_train_norm = norm_train(train_dataset)
X_test_norm = norm_train(test_dataset)

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#setting up the initial models
rf_reg = RandomForestRegressor()
gb_reg = GradientBoostingRegressor()
xgb_reg = XGBRegressor()
La_reg=Lasso()
ridge_reg = Ridge()

#fitting the initial models
for model in [rf_reg, gb_reg,xgb_reg,La_reg,ridge_reg]:
    model.fit(X_train_norm, y_train_TC)

import sklearn.metrics
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#running predictions on Train set
rf_reg_pred_train = rf_reg.predict(X_train_norm)
gb_reg_pred_train = gb_reg.predict(X_train_norm)
xgb_reg_pred_train = xgb_reg.predict(X_train_norm)
La_reg_pred_train = La_reg.predict(X_train_norm)
ridge_reg_pred_train = ridge_reg.predict(X_train_norm)

#running predictions on Training set
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
mae_train_rf = mean_absolute_error(rf_reg_pred_train, y_train_TC, )
mae_train_gb = mean_absolute_error(gb_reg_pred_train, y_train_TC)
mae_train_xgb = mean_absolute_error(xgb_reg_pred_train, y_train_TC)
mae_train_La = mean_absolute_error(La_reg_pred_train, y_train_TC)
mae_train_ridge = mean_absolute_error(ridge_reg_pred_train, y_train_TC)

mse_train_rf = sklearn.metrics.mean_squared_error(rf_reg_pred_train, y_train_TC, multioutput='uniform_average', squared=True)
mse_train_gb = sklearn.metrics.mean_absolute_error(gb_reg_pred_train, y_train_TC, multioutput='uniform_average')
mse_train_xgb = sklearn.metrics.mean_absolute_error(xgb_reg_pred_train, y_train_TC, multioutput='uniform_average')
mse_train_La = sklearn.metrics.mean_absolute_error(La_reg_pred_train, y_train_TC, multioutput='uniform_average')
mse_train_ridge = sklearn.metrics.mean_absolute_error(ridge_reg_pred_train, y_train_TC, multioutput='uniform_average')

from sklearn.metrics import r2_score
R2_train_rf = r2_score(y_train_TC,rf_reg_pred_train)
R2_train_gb = r2_score(y_train_TC,gb_reg_pred_train)
R2_train_xgb = r2_score(y_train_TC,xgb_reg_pred_train)
R2_train_La = r2_score(y_train_TC,La_reg_pred_train)
R2_train_ridge = r2_score(y_train_TC,ridge_reg_pred_train)

rmse_train_rf = math.sqrt(mse_train_rf)
rmse_train_gb = math.sqrt(mse_train_gb)
rmse_train_xgb = math.sqrt(mse_train_xgb)
rmse_train_La = math.sqrt(mse_train_La)
rmse_train_ridge = math.sqrt(mse_train_ridge)

scoring_results = [('Random Forest Regression', rmse_train_rf, R2_train_rf,mae_train_rf),
                   ('Gradient Boosting Regression', rmse_train_gb,R2_train_gb, mae_train_gb),
                   ('XGBRegressor', rmse_train_xgb,R2_train_xgb, mae_train_xgb),
                   ('Lasso', rmse_train_La,R2_train_La, mae_train_La),
                   ('ridge', rmse_train_ridge,R2_train_ridge, mae_train_ridge),
]

scoring_df = pd.DataFrame(data = scoring_results,
                       columns=['Model Name', 'RMSE', 'R2 Score(Train)', 'MAE(Train)'])
scoring_df

#running predictions on Train set
rf_reg_pred_test = rf_reg.predict(X_test_norm)
gb_reg_pred_test = gb_reg.predict(X_test_norm)
xgb_reg_pred_test = xgb_reg.predict(X_test_norm)
La_reg_pred_test = La_reg.predict(X_test_norm)
ridge_reg_pred_test = ridge_reg.predict(X_test_norm)

#running predictions on Training set
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
mae_test_rf = mean_absolute_error(rf_reg_pred_test, y_test_TC, )
mae_test_gb = mean_absolute_error(gb_reg_pred_test, y_test_TC)
mae_test_xgb = mean_absolute_error(xgb_reg_pred_test, y_test_TC)
mae_test_La = mean_absolute_error(La_reg_pred_test, y_test_TC)
mae_test_ridge = mean_absolute_error(ridge_reg_pred_test, y_test_TC)

mse_test_rf = sklearn.metrics.mean_squared_error(rf_reg_pred_test, y_test_TC, multioutput='uniform_average')
mse_test_gb = sklearn.metrics.mean_absolute_error(gb_reg_pred_test, y_test_TC, multioutput='uniform_average')
mse_test_xgb = sklearn.metrics.mean_absolute_error(xgb_reg_pred_test, y_test_TC, multioutput='uniform_average')
mse_test_La = sklearn.metrics.mean_absolute_error(La_reg_pred_test, y_test_TC, multioutput='uniform_average')
mse_test_ridge = sklearn.metrics.mean_absolute_error(ridge_reg_pred_test, y_test_TC, multioutput='uniform_average')

from sklearn.metrics import r2_score
R2_test_rf = r2_score(y_test_TC,rf_reg_pred_test)
R2_test_gb = r2_score(y_test_TC,gb_reg_pred_test)
R2_test_xgb = r2_score(y_test_TC,xgb_reg_pred_test)
R2_test_La = r2_score(y_test_TC,La_reg_pred_test)
R2_test_ridge = r2_score(y_test_TC,ridge_reg_pred_test)

rmse_test_rf = math.sqrt(mse_test_rf)
rmse_test_gb = math.sqrt(mse_test_gb)
rmse_test_xgb = math.sqrt(mse_test_xgb)
rmse_test_La = math.sqrt(mse_test_La)
rmse_test_ridge = math.sqrt(mse_test_ridge)

scoring_results = [('Random Forest Regression', rmse_test_rf, R2_test_rf,mae_test_rf),
                   ('Gradient Boosting Regression', rmse_test_gb,R2_test_gb, mae_test_gb),
                   ('XGBRegressor', rmse_test_xgb,R2_test_xgb, mae_test_xgb),
                   ('Lasso', rmse_test_La,R2_test_La, mae_test_La),
                   ('ridge', rmse_test_ridge,R2_test_ridge, mae_test_ridge),
]

scoring_df = pd.DataFrame(data = scoring_results,
                       columns=['Model Name', 'RMSE', 'R2 Score(Test)', 'MAE(Test)'])
scoring_df

from sklearn.model_selection import GridSearchCV

# Defining the grid values

parameters = {'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18],
                  'subsample'    : [0.4,],
                   'max_depth': [3,4,5,6,8],
                 'n_estimators' : [100,200,300,400,600,800,1000],
              'min_samples_leaf': [1,2,3],
              'min_samples_split': [2,3,4]
                 }
# Redefining the regressor
GBR = GradientBoostingRegressor(random_state = 1)

# Gridsearch

grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = 5, scoring = 'neg_mean_absolute_error')
grid_GBR.fit(X_train_norm, y_train_TC)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_GBR.best_estimator_)
print("\n The best parameters across ALL searched params:\n",grid_GBR.best_params_)