import pandas as pd
from typing import Union
import pandas as pd
import math
import numpy as np
from itertools import count
from functools import reduce

'''def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """

    assert isinstance(y_hat, pd.Series), f"Expected y_hat to be a pandas Series, got {type(y_hat)}"
    assert isinstance(y, pd.Series), f"Expected y to be a pandas Series, got {type(y)}"
    
    assert y_hat.size == y.size
    # TODO: Write here
   
    if (isinstance(y_hat,pd.Series)):
        y_hat = y_hat.tolist()
    if (isinstance(y,pd.Series)):
        y = y.tolist()
    
    l = len(y_hat)
    count = 0
    for i in range(l):
        if (y_hat[i] == y[i]):
            count+=1
    
    ans = (count/l)*100
    return ans
    pass
'''


def accuracy(y_hat:pd.Series , y:pd.Series):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    if len(y_hat) != len(y):
        raise ValueError("Length of predicted labels and true labels must be the same.")
    
    # Calculate the number of correct predictions
    if not isinstance(y_hat, pd.Series):
        y_hat = pd.Series(y_hat)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    #correct_predictions =reduce(lambda x,m:x+y_hat.count(m),y,0)
    correct_predictions = sum(y_hat == y)
    
    # Calculate accuracy
    accuracy = correct_predictions / len(y)
    
    return accuracy



def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """

    if (isinstance(y_hat,pd.Series)):
        y_hat = y_hat.tolist()
    if (isinstance(y,pd.Series)):
        y = y.tolist()
    
    chosen_class = cls
    total_samples = len(y)

    pred_class_total = y_hat.count(chosen_class)
    
    correct_pred_count = 0
    for i in range(total_samples):
        if (y_hat[i] == chosen_class):
            if (y_hat[i] == y[i]):
                correct_pred_count+=1
    if (pred_class_total == 0):
        return None
    ans = (correct_pred_count/pred_class_total)*100
    return ans
    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """

    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()

    chosen_class = cls
    total_samples = len(y)

    total_samples_chosen = y.count(chosen_class)
    recall_count = 0

    for i in range(total_samples):
        if (y[i]==chosen_class):
            if (y_hat[i]==y[i]):
                recall_count+=1

    ans = (recall_count/total_samples_chosen)*100
    return ans
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    diff = [0]*len(y)
    for i in range (len(y)):
        diff[i]=((y_hat[i])-(y[i]))**2
    ans = sum(diff)/len(y)
    ans = math.sqrt(ans)
    return ans



    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """

    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    diff = [0]*len(y)
    for i in range (len(diff)):
        diff[i]=abs(y_hat[i]-y[i])
    return sum(diff)/len(y)
    pass
