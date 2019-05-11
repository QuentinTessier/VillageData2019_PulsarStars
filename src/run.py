#!/usr/bin/python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../pulsar_stars.csv')

data.dropna()
data.head()

scaler = StandardScaler()

X_data = data.iloc[:,0:-1].values

Y_data = data.iloc[:,-1].values

scaler.fit(X_data)
X_scaled = scaler.transform(X_data)

trainer = train_test_split(X_scaled, Y_data)

print(X_scaled[0:5])