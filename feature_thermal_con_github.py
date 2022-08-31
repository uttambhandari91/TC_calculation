
import pandas as pd

#Import required libraries
import pandas as pd
import numpy as np
import pandas

import matplotlib.pyplot as plt
import seaborn as sns

#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Remove the limit from the number of displayed columns and rows. It helps to see the entire dataframe while printing it
pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
pd.set_option("display.max_rows", 200)

dataset = pd.read_csv('/content/alloy composition TC#1.csv')

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

pip install shap==0.41.0

import shap

import matplotlib.pyplot as plt
figure = plt.gcf() # get current figure
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 12
figure.set_size_inches(20, 10)

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
GBR=GradientBoostingRegressor(learning_rate= 0.07, max_depth= 4, n_estimators= 1000, subsample=0.4, min_samples_split=2, min_samples_leaf=2)
model = GBR.fit(X_train_norm, y_train_TC)
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X_train_norm)

# visualize the first prediction's explanation
shap.plots.bar(shap_values)
figure = plt.gcf() # get current figure
axes = plt.figure(figsize=(30, 30))

shap.plots.bar(shap_values,show=False)
figure = plt.gcf() # get current figure
axes = plt.figure(figsize=(30, 30))

# summarize the effects of all the features
shap.plots.beeswarm(shap_values,plot_size=(10,10))
figure = plt.gcf() # get current figure
figure.set_size_inches(20, 10)

shap.summary_plot(shap_values, plot_type='dot', plot_size=(7, 7), cmap='hsv',max_display=5)

import matplotlib.pyplot as plt

# Define colormap
my_cmap = plt.get_cmap('viridis')

# Plot the summary without showing it
plt.figure()
shap.plots.bar(shap_values,
                  show=False
                  )

# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap(my_cmap)