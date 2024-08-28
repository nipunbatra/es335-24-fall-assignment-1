import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from tree.base import DecisionTree
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])
data.drop(data.index[data['horsepower'] == "?"], inplace=True)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
# removing unclear columns
data.drop('car name', axis=1, inplace=True)

#print(data)


label_encoder = LabelEncoder()

data['cylinders'] = label_encoder.fit_transform(data['cylinders'])
data['model year'] = label_encoder.fit_transform(data['model year'])
data['origin'] = label_encoder.fit_transform(data['origin'])
X = data.drop(columns=['mpg'])
print(X)

y = data['mpg']
print(y)
print(data.index)
data.reset_index(drop=True, inplace=True)
print(data.iloc[1:3])

# Remove rows where 'horsepower' is "?"
data.drop(data.index[data['horsepower'] == "?"], inplace=True)

# Convert horsepower to numeric
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# Drop any rows where 'horsepower' is NaN
data.dropna(subset=['horsepower'], inplace=True)

# Reset the index
data.reset_index(drop=True, inplace=True)

# View the rows at index 1 and 2
print(data.iloc[1:3])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

'''tree0 = DecisionTree(criterion='mse')  # Split based on Inf. Gain

tree0.fit(X_train, y_train)
y_hat = tree0.predict(X_test)
tree0.plot()
#print(f"X_test type: {type(X_test)}")
#print(f"X_test shape: {X_test.shape}")

# Predict using the custom decision tree
#try:
    #y_hat = tree0.predict(X_test)
    #print("Custom Decision Tree Predictions:", y_hat)
#except Exception as e:
      # print(f"Error during prediction: {e}")
print('RMSE: ', rmse(y_hat, y_test))
print('MAE: ', mae(y_hat, y_test))
print('---------------------')'''

tree1 = DecisionTreeRegressor(max_depth=5)
tree1.fit(X_train, y_train)
text_representation = tree.export_text(tree1)
print(text_representation)
yy_hat = tree1.predict(X_test)
yy_hat = pd.Series(yy_hat)
print('RMSE: ', rmse(yy_hat, y_test))
print('MAE: ', mae(yy_hat, y_test))

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn
