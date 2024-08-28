from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    same_value = (y==y_hat).sum()
    return float(same_value)/y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size

    TP = (y[y_hat==cls]==cls).sum()
    yhat_true = (y_hat==cls).sum()
    if yhat_true ==0 : 
        return -1
    return float(TP)/yhat_true


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size

    TP = (y[y_hat==cls]==cls).sum()
    y_true = (y==cls).sum()

    return float(TP)/y_true


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size

    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return np.sqrt(np.mean((y_hat-y)**2))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size

    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return (y_hat-y).abs().mean()