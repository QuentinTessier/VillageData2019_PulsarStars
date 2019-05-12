#!/usr/bin/python

import numpy as np
import pandas as pd # Parse CSV file

from matplotlib import pyplot as plt # Basic graph plotting library
import seaborn as sns   # advanced graph plotting library

from sklearn.preprocessing import MinMaxScaler # Scaler to normalize the data

from sklearn.model_selection import train_test_split # Construct a trainer out of 4 data sets

from DecitionTree import DecisionTree
from KNN import KNN
from Support_Vector_Machine import SupportVectorMachine
from RandomForestClassifier import RandomForest
from Naive_Bayes_Classifier import NaiveBayes
from LogisticRegression import LogisticRegression

# Open csv file
data = pd.read_csv('../pulsar_stars.csv')

# Display Correlation of the Different object with one another
plt.figure(figsize=(16,12))
sns.heatmap(data=data.corr(),annot=True,cmap="bone",linewidths=1,fmt=".2f",linecolor="gray")
plt.title("Correlation Map",fontsize=20)
plt.tight_layout()
plt.show()

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

# Training and Testing

dc_score, dc_head = DecisionTree(x_train, y_train, x_test, y_test)
knn_score, knn_head = KNN(x_train, y_train, x_test, y_test)
#svm_score, svm_head = SupportVectorMachine(x_train, y_train, x_test, y_test)
rfc_score, rfc_head = RandomForest(x_train, y_train, x_test, y_test)
nb_score, nb_head = NaiveBayes(x_train, y_train, x_test, y_test)
#lr_score, lr_head = LogisticRegression(x_train, y_train, x_test, y_test)

# ~~~~~~~~~~~~~~~~~~~~

# Add scores to the array
scores = (dc_score, knn_score, rfc_score, nb_score)

# Add heads to the array
heads = [dc_head, knn_head, rfc_head, nb_head]

# Name of the Algorithm
algorithms = ("Decision Tree", "K Nearest Neighbor", "Random Forest", "Naive Bayes")

# Range of elements
h_pos = np.arange(1, 4)

# Color of the bar
colors = ["red", "red", "red", "red"]

# Display bar graph to compare every algorithms results

# plt.figure(figsize=(24, 12))
# plt.xticks(h_pos, algorithms, fontsize=18)
# plt.yticks(np.arange(0.00, 1.01, step=0.01))
# plt.ylim(0.90, 1.00)
# plt.bar(h_pos, scores, color=colors)
# plt.grid()
# plt.suptitle("Models Comparison", fontsize=24)
# plt.show()