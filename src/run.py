#!/usr/bin/python

import numpy as np
import pandas as pd # Parse CSV file

from matplotlib import pyplot as plt # Basic graph plotting library
import seaborn as sns   # advanced graph plotting library

from sklearn.preprocessing import MinMaxScaler # Scaler to normalize the data

from sklearn.model_selection import train_test_split # Construct a trainer out of 4 data sets

from DecitionTree import DecisionTree

# Open csv file
data = pd.read_csv('../pulsar_stars.csv')

# Get values from the "target_class" column
labels = data.target_class.values

# Drop the "target_class" column to only keep the data fed to the algorithms
data.drop(["target_class"], axis=1, inplace=True)
features = data.values

# Scaling all data to the range {0, 1}
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# Return 20% of the data as test data, the rest is used to train our models
x_train, x_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2)

# Printing current result of the DecisionTree
print(DecisionTree(x_train, y_train, x_test, y_test))