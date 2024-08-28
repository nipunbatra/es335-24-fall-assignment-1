import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
from sklearn.tree import DecisionTreeRegressor

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn
print(data.head(10))
#Cleaning
data.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]

#mpg - continous variable - target variable
data["horsepower"]  = data["horsepower"].str.replace("?", "NaN", regex= False).astype(float)
data["horsepower"].fillna(value= data["horsepower"].mean(), inplace=True) 
# only horsepower had missing values

#Preparing datasets  
data = data.drop(["cylinders", "model year", "origin", "car name"],  axis = 1)  #dropping the columns with categorical attributes

X = data.iloc[:, 1:] 
y = pd.Series(data.iloc[:, 0])
print(X.head())
print(y.head())
#Spliting X and y in train and test datasets in a ratio 7:3
X_train, X_test = np.split(X, [int(X.shape[0] * 0.7)])
X_train = pd.DataFrame(X_train) 
X_test = pd.DataFrame(X_test)

y_train, y_test = np.split(y, [int(y.shape[0] * 0.7)])
y_train = pd.Series(y_train, dtype= "float64") 
y_test = pd.Series(y_test, dtype= "float64")
y_test = y_test.reset_index(drop=True)

# Using our decision tree
tree = DecisionTree(criterion= "information_gain")
tree.fit(X_train, y_train) 
y_hat = tree.predict(X_test)  
tree.plot()

# Using sklearn's decison tree
tree_sk = DecisionTreeRegressor(random_state=0)
tree_sk.fit(X_train, y_train)
y_hat_sk = pd.Series(tree_sk.predict(X_test))

# The metrics used for comparision are RMSE and MAE
summary = {"Metric": ["RMSE", "MAE"], "Our_model":[], "Scikit":[]} 
summary["Our_model"].append(rmse(y_hat, y_test)) 
summary["Our_model"].append(mae(y_hat, y_test)) 
summary["Scikit"].append(rmse(y_hat_sk, y_test)) 
summary["Scikit"].append(mae(y_hat_sk,  y_test)) 

print(pd.DataFrame(summary))
