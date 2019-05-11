#!/usr/bin/python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

DataFrame = pd.read_csv("../pulsar_stars.csv")  
DataFrame.head()
DataFrame.info()
DataFrame.describe()
DataFrame.corr()
labels = DataFrame.target_class.values

DataFrame.drop(["target_class"],axis=1,inplace=True)

features = DataFrame.values

scaler = MinMaxScaler(feature_range=(0,1))

features_scaled = scaler.fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(features_scaled,labels,test_size=0.2)

knn_model = KNeighborsClassifier(n_neighbors=7,weights="distance")

knn_model.fit(x_train,y_train)

y_head_knn = knn_model.predict(x_test)

knn_score = knn_model.score(x_test,y_test)
print(knn_score)