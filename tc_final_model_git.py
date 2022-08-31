
import pandas as pd

dataset = pd.read_csv('/content/alloy composition TC.csv')

dataset.head()

dataset. columns

dataset.info()

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
from sklearn.ensemble import GradientBoostingRegressor
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

GBR=GradientBoostingRegressor(learning_rate= 0.03, max_depth= 3, n_estimators= 1000, subsample=0.4, min_samples_split=2, min_samples_leaf=1)
# Fit regression model
model_TC = GBR.fit(X_train_norm, y_train_TC)

#Choose metrics. Check the accuracy of the prediction

y_predict_train_TC = model_TC.predict(X_train_norm).flatten()

mae_train_TC = sklearn.metrics.mean_absolute_error(y_predict_train_TC, y_train_TC, multioutput='uniform_average')
mse_train_TC = sklearn.metrics.mean_squared_error(y_predict_train_TC, y_train_TC, multioutput='uniform_average', squared=True)
R2_train_TC = sklearn.metrics.r2_score(y_predict_train_TC, y_train_TC)
rmse_train_TC = math.sqrt(mse_train_TC)

print('Metrics for C44 calculation')
print('\n')
print('MAE_train = ', round(mae_train_TC,3))
print('MSE_train = ', round(mse_train_TC,3))
print('R^2_train = ', round(R2_train_TC,3))
print('RMSE_train = ', round(rmse_train_TC,3))
print('\n')

y_predict_TC = model_TC.predict(X_test_norm).flatten()

#Choose metrics. Check the accuracy of the prediction

mae_test_TC = sklearn.metrics.mean_absolute_error(y_predict_TC, y_test_TC, multioutput='uniform_average')
mse_test_TC = sklearn.metrics.mean_squared_error(y_predict_TC, y_test_TC, multioutput='uniform_average', squared=True)
R2_test_TC = sklearn.metrics.r2_score(y_predict_TC, y_test_TC)
rmse_test_TC = math.sqrt(mse_test_TC)

print('MAE_test = ', round(mae_test_TC,3))
print('MSE_test = ', round(mse_test_TC,3))
print('R^2_test = ', round(R2_test_TC,3))
print('RMSE_test = ', round(rmse_test_TC,3))

#Compare predicted and true values
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.gcf()

# Change seaborn plot size
fig.set_size_inches(8,8)
a = plt.axes(aspect='equal')
#plt.scatter(y_test, y_predict, marker="s", label="test", s = 100, facecolors='aqua', edgecolors='indigo', linewidth = 1.7)
plt.scatter(y_train_TC, y_predict_train_TC, marker="o", label="train", s = 75, facecolors='aqua', edgecolors='black', linewidth = 1.7)
plt.scatter(y_test_TC, y_predict_TC, marker="s", label="test", s = 100, facecolors='red', edgecolors='indigo', linewidth = 1.7)
plt.legend(fontsize=14)
plt.xticks(fontsize=20,fontweight ='bold')
plt.yticks(fontsize=20,fontweight ='bold')
plt.xlabel('experimental TC', fontsize=25,fontweight ='bold')
plt.ylabel('ML Calculation', fontsize=25,fontweight ='bold')
lims = [8,50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
fontsize = 30
plt.legend(loc="upper left", frameon=True, fontsize=fontsize)
plt.tight_layout()
#plt.savefig('E_form.png', dpi=300)
plt.show()

dataset1 = pd.read_csv('/content/new_ICONEL.csv')

df_TC = dataset1[['%Ni', '%Cu', '%Fe', '%Cr', '%Mo', '%Nb', '%Ta', '%Mn', '%Si', '%Co',
       '%Al', '%Ti', '%Zr', '%W', '%V', '%C', '%B', '%P', 'S', '%La', 'TT']]

# Normalization of features for calculating TC
X_TC = norm_train(df_TC)

E_pred = model_TC.predict(X_TC)

df_TC['TC'] = E_pred

print(df_TC)

array = dataset.values
X = array[:,0:20]
Y = array[:,21]

from sklearn.model_selection import RepeatedKFold
GBR2 = GradientBoostingRegressor(learning_rate= 0.03, max_depth= 3, n_estimators= 1000, subsample=0.4, min_samples_split=2, min_samples_leaf=1)
cv = RepeatedKFold(n_splits=10, random_state=1)
np.mean(cross_val_score(GBR2, X, Y, cv=cv))

