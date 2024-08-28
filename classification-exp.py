import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
import sys

from collections import Counter
np.random.seed(42)




# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
class_counts = Counter(y)
print("Class composition using Counter:")
print(class_counts)
'''# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()
dataset = pd.DataFrame(X)
dataset['Y'] = pd.Series(y, dtype='category')
X = dataset.iloc[:, :-1]
y = dataset['Y']
frac = 0.7
split_val = int(frac*len(dataset))

train_data, test_data = dataset.iloc[:split_val, :], dataset.iloc[split_val+1:, :]
X_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]
X_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]



tree = DecisionTree(criterion="information_gain")  # Split based on Inf. Gain
tree.fit(X_train,y_train)
y_hat = tree.predict(X_test)
print(len(y_hat))
y_hat = pd.Series([y_hat])
print(y_hat)
print(f"y_hat size: {len(y_hat)}, y size: {len(y_test)}")
y_test = pd.Series([y_test])
print(f"y_hat size: {len(y_hat)}, y size: {len(y_test)}")
tree.plot()
print('Accuracy: ', accuracy(y_hat,  y_test))
for cls in test_data.iloc[:, -1].unique():
    print('Precision: ', precision(y_hat, test_data.iloc[:, -1], cls))
    print('Recall: ', recall(y_hat, test_data.iloc[:, -1], cls))


# Optimizing for depth
def find_best_depth(X, y, folds=5, depths=[1]):
    assert(len(X) == len(y))
    assert(len(X) > 0)

    max_depth = max(depths)
    trees = {}
    accuracies = {}
    # similar to 100/4 = 25, size of data in each fold
    sub_size = int(len(X)//folds)

    for fold in range(folds):
        # Training a seperate model for each fold
        # getting the first set of data
        sub_data_inddexes = range(fold*sub_size, (fold+1)*sub_size)
        c_fold = []
        for i in range(len(X)):
            if(i in sub_data_inddexes):
                c_fold.append(True)
            else:
                c_fold.append(False)
        c_fold = pd.Series(c_fold)
            
        X_train = X[~c_fold].reset_index(
            drop=True)
        y_train = y[~c_fold].reset_index(drop=True)
        X_test = X[c_fold].reset_index(drop=True)
        y_test = y[c_fold].reset_index(drop=True)

        tree = DecisionTree(criterion='information_gain',max_depth=max_depth)
        tree.fit(X_train, y_train)
        trees[fold+1] = tree

        for depth in depths:
            print("Depth is "+str(depth))
            tree = DecisionTree('information_gain',max_depth=depth)
            tree.fit(X_train, y_train)
            y_hat = tree.predict(X_test)
            if fold+1 in accuracies:
                accuracies[fold+1][depth] = accuracy(y_hat, y_test)
            else:
                accuracies[fold+1] = {depth: accuracy(y_hat, y_test)}
                


    accuracies = pd.DataFrame(accuracies).transpose()
    accuracies.index.name = "Fold ID"
    accuracies.loc["mean"] = accuracies.mean(axis = 'rows')
    best_mean_acc = accuracies.loc["mean"].max()
    best_depth = accuracies.loc["mean"].idxmax()
    print(accuracies)
    print("Best Mean Accuracy ===>" + str(best_mean_acc))
    print("Optimum Depth ===>"+str(best_depth))


find_best_depth(X, y, folds=5, depths=list(range(1, 11)))

'''
# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()
dataset = pd.DataFrame(X)
dataset['Y'] = pd.Series(y, dtype='category')
X = dataset.iloc[:, :-1]
y = dataset['Y']
frac = 0.7
split_val = int(frac*len(dataset))

train_data, test_data = dataset.iloc[:split_val, :], dataset.iloc[split_val+1:, :]
X_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]
X_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]



tree = DecisionTree(criterion="gini_index",max_depth=10)  # Split based on Inf. Gain
tree.fit(X_train,y_train)
y_hat = tree.predict(X_test)

print(len(y_hat))

y_hat_list=y_hat.tolist()
print(y_hat_list)
#print(f"y_hat size: {}, y size: {len(y_test)}")
y_test_list= y_test.tolist()
print(f"y_hat size: {len(y_hat_list)}, y size: {len(y_test_list)}")
#y_hat = pd.Series([y_hat])
#y_test = pd.Series([y_test])



tree.plot()
print('Accuracy: ', accuracy(y_hat_list,  y_test_list))
for cls in test_data.iloc[:, -1].unique():
    print('Precision: ', precision(y_hat, test_data.iloc[:, -1], cls))
    print('Recall: ', recall(y_hat, test_data.iloc[:, -1], cls))

from sklearn.metrics import f1_score


y_pred = tree.predict(X_test)
f1 = f1_score(y_test, y_pred, average='binary')
print(f"F1 Score: {f1:.4f}")


# Optimizing for depth
'''def find_best_depth(X, y, folds=5, depths=[1]):
    assert(len(X) == len(y))
    assert(len(X) > 0)

    max_depth = max(depths)
    trees = {}
    accuracies = {}
    # similar to 100/4 = 25, size of data in each fold
    sub_size = int(len(X)//folds)

    for fold in range(folds):
        # Training a seperate model for each fold
        # getting the first set of data
        sub_data_inddexes = range(fold*sub_size, (fold+1)*sub_size)
        c_fold = []
        for i in range(len(X)):
            if(i in sub_data_inddexes):
                c_fold.append(True)
            else:
                c_fold.append(False)
        c_fold = pd.Series(c_fold)
            
        X_train = X[~c_fold].reset_index(
            drop=True)
        y_train = y[~c_fold].reset_index(drop=True)
        X_test = X[c_fold].reset_index(drop=True)
        y_test = y[c_fold].reset_index(drop=True)

        tree = DecisionTree(criterion='information_gain',max_depth=max_depth)
        tree.fit(X_train, y_train)
        trees[fold+1] = tree

        for depth in depths:
            print("Depth is "+str(depth))
            tree = DecisionTree('information_gain',max_depth=depth)
            tree.fit(X_train, y_train)
            y_hat = tree.predict(X_test)
            if isinstance(y_hat, int):  # If it's a single integer (scalar prediction)
                y_hat_list = [y_hat] * len(X_test)  # Repeat the scalar prediction for all test samples
            else:
                y_hat_list = y_hat.tolist()  # Convert to a list if it's not already one
    
            print("Length of y_hat: ", len(y_hat_list))
            print("Predictions: ", y_hat_list)
            y_test_list=y_test.tolist()
            if fold+1 in accuracies:
                accuracies[fold+1][depth] = accuracy(y_hat_list, y_test_list)
            else:
                accuracies[fold+1] = {depth: accuracy(y_hat_list, y_test_list)}'''

from sklearn.model_selection import KFold

def find_best_depth(X, y, folds=5, depths=list(range(1, 11))):
    assert len(X) == len(y)
    assert len(X) > 0

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)  # Ensures randomness and prevents leakage
    accuracies = {}

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        for depth in depths:
            print(f"Fold {fold}, Depth {depth}")
            
            # Train the decision tree with the given depth
            tree = DecisionTree(criterion='gini_index', max_depth=depth)
            tree.fit(X_train, y_train)
            
            # Predict on the test set
            y_hat = tree.predict(X_test)
            
            # Convert predictions and true values to lists for the accuracy function
            if isinstance(y_hat, int):  # If a single value is predicted
                y_hat_list = [y_hat] * len(X_test)
            else:
                y_hat_list = y_hat.tolist()
            
            y_test_list = y_test.tolist()
            
            # Calculate accuracy for this fold and depth
            acc = accuracy(y_hat_list, y_test_list)
            
            # Store accuracies in the dictionary
            if fold in accuracies:
                accuracies[fold][depth] = acc
            else:
                accuracies[fold] = {depth: acc}

    # Convert accuracies dictionary to a DataFrame for easier analysis
    accuracies_df = pd.DataFrame(accuracies).transpose()
    accuracies_df.index.name = "Fold ID"
    accuracies_df.loc["mean"] = accuracies_df.mean(axis='rows')

    # Find the best mean accuracy and the corresponding depth
    best_mean_acc = accuracies_df.loc["mean"].max()
    best_depth = accuracies_df.loc["mean"].idxmax()

    # Display the accuracies table and best depth information
    print(accuracies_df)
    print("Best Mean Accuracy ===> {:.2f}".format(best_mean_acc))
    print("Optimum Depth ===> {}".format(best_depth))

# Example usage
find_best_depth(X, y, folds=5, depths=list(range(1, 11)))


