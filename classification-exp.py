import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
# by default 100 samples

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# Write the code for Q2 a) and b) below. Show your results.

print("2 (a) : using the Decision Tree")
#Splitting the data
indices = np.arange(len(X))
np.random.shuffle(indices)
split_index = int(0.7 * len(indices))
train_indices, test_indices = indices[:split_index], indices[split_index:]
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Training Set')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Testing Set')
plt.legend()
# plt.show()

X_train, y_train = pd.DataFrame(X[train_indices], columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y[train_indices], name='target')
X_test, y_test = pd.DataFrame(X[test_indices], columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y[test_indices], name='target')
print(X.shape)
print(X_train.shape)
print(y_train.shape)
# Using the tree
tree = DecisionTree(criterion="information_gain", max_depth=np.inf)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Calculate accuracy
accuracy_value = (y_pred == y_test).mean()
print("Accuracy:", accuracy_value)

# Calculate per-class precision and recall
unique_classes = pd.Series(y).unique()

precision_recall_dict = {"Class": [], "Precision": [], "Recall": []}

for class_label in unique_classes:
    true_positive = np.sum((y_pred == class_label) & (y_test == class_label))
    false_positive = np.sum((y_pred == class_label) & (y_test != class_label))
    false_negative = np.sum((y_pred != class_label) & (y_test == class_label))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    precision_recall_dict["Class"].append(class_label)
    precision_recall_dict["Precision"].append(precision)
    precision_recall_dict["Recall"].append(recall)

precision_recall_df = pd.DataFrame(precision_recall_dict)
print("\nPer-class Precision and Recall:")
print(precision_recall_df)
print()
print("=====================================")


# Q2b
# K-Fold Cross Validation
# Define the number of folds (k)

print("2 (b) : 5 fold cross-validation") 
print()

k = 5
begin_test = 0
data = {"Fold": [], "Accuracy" : []}
for i in range(k):
    X_test_kf = pd.DataFrame(X[begin_test:begin_test + int(0.2*X.shape[0])] )
    y_test_kf = pd.Series(y[begin_test: begin_test + int(0.2*y.shape[0])] , dtype= "category")
    X_train_kf = pd.DataFrame(np.concatenate((X[0: begin_test] , X[begin_test + int(0.2*X.shape[0]) : ])))
    y_train_kf = pd.Series(np.concatenate((y[0:begin_test] , y[begin_test + int(0.2*y.shape[0]):] ) ), dtype= "category")
    
    tree = DecisionTree(criterion="information_gain")
    tree.fit(X_train_kf, y_train_kf)
    y_hat = tree.predict(X_test_kf)
    accuracy_value = accuracy(y_hat, y_test_kf)
    data["Fold"].append(i + 1)
    data["Accuracy"].append(accuracy_value)

    begin_test += int(X.shape[0]*0.2)   #Each fold/part will be 20% of the whole dataset

print(pd.DataFrame(data)) 

# Nested cross-validation to find the optimum depth of the decision tree.
print()
begin_valid = 0
max_depth_list = [1,2,3,4,5,6,7,8,9,10]

summary = {0:[], 1:[], 2:[], 3:[], 4:[]} #stores accuracy for each depth
for i in range(5):
    X_valid = pd.DataFrame(X[begin_valid:begin_valid + int(0.2*X.shape[0])] )
    y_valid = pd.Series(y[begin_valid: begin_valid+ int(0.2*y.shape[0])] , dtype= "category")
    X_train_kf = pd.DataFrame(np.concatenate((X[0: begin_valid] , X[begin_valid + int(0.2*X.shape[0]) : ])))
    y_train_kf = pd.Series(np.concatenate((y[0:begin_valid] , y[begin_valid + int(0.2*y.shape[0]):] ) ) , dtype= "category")

    for depth in max_depth_list: 
        tree = DecisionTree(criterion= "information_gain", max_depth = depth) 
        tree.fit(X_train_kf, y_train_kf) 
        y_hat_valid = tree.predict(X_valid) 
        summary[i].append(accuracy(y_hat_valid, y_valid))      #storing the accuracy for each depth
    begin_valid += int(X.shape[0]*0.2)

summary = pd.DataFrame(summary)
summary["Average Score"] = summary.mean(axis=1) 
print(summary)
print()

#The optimal depth corrosponds to that at which the avg of accuracy scores of all folds is maximum.
print("Nested Cross Validation using 5 folds:")
print("The optimal depth is {} and it has accuracy = {}".format(summary["Average Score"].idxmax(), round(summary["Average Score"].max(), 2)))
