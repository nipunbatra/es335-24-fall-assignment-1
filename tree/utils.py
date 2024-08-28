"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, drop_first=False)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_float_dtype(y): 
        return True
    else :
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    value, counts = np.unique(Y, return_counts=True)
    
    prob = counts/counts.sum()
    entropy = 0
    for p in prob : 
        entropy -= p*np.log2(p)

    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    value, counts = np.unique(Y, return_counts = True) 
    prob = counts/counts.sum()
    gini = 1
    gini -= np.sum(prob**2)

    return gini

def mse(Y : pd.Series)->float:
    mean_val = Y.mean()
    mse_value = np.mean((Y-mean_val)**2)
    return mse_value

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
     # Initial impurity based on the chosen criterion
    if criterion == 'entropy':
        initial_impurity = entropy(Y)
    elif criterion == 'gini':
        initial_impurity = gini_index(Y)
    elif criterion == 'mse':
        initial_impurity = mse(Y)
    else:
        raise ValueError("Invalid criterion. Choose from 'entropy', 'gini', or 'mse'.")

    # Weighted impurity after splitting based on attribute
    unique_values = np.unique(attr)
    weighted_impurity = 0
    
    for value in unique_values:
        subset_Y = Y[attr == value]
        subset_weight = len(subset_Y) / len(Y)
        
        if criterion == 'entropy':
            subset_impurity = entropy(subset_Y)
        elif criterion == 'gini':
            subset_impurity = gini_index(subset_Y)
        elif criterion == 'mse':
            subset_impurity = mse(subset_Y)
        
        weighted_impurity += subset_weight * subset_impurity

    # Information gain is the reduction in impurity
    info_gain = initial_impurity - weighted_impurity
    return info_gain


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_gain = -np.inf
    best_feature = None

    for feature in features:
        attr = X[feature]
        gain = information_gain(y, attr, criterion)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if pd.api.types.is_numeric_dtype(X[attribute]):
        # For real-valued features, use a threshold to split the data
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
    else:
        # For discrete features, split based on exact match
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value

    # Split the data based on the mask
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    return X_left, y_left, X_right, y_right