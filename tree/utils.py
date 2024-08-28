"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""


import math


import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input DataFrame.

    Parameters:
    - X (pd.DataFrame): The input DataFrame with categorical features.

    Returns:
    - pd.DataFrame: A DataFrame with one-hot encoded features.
    """
    # Perform one-hot encoding
    encoded_df = pd.get_dummies(X, drop_first=False)
    
    return encoded_df
    pass
def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real (continuous) values or discrete (categorical) values.

    Parameters:
    - y (pd.Series): The input series to check.

    Returns:
    - bool: True if the series has real (continuous) values, False if it has discrete (categorical) values.
    """
    # Check if all values are numeric
    if pd.api.types.is_numeric_dtype(y):
        # Check if the number of unique values is significantly smaller than the total number of values
        num_unique_values = len(y.unique())
        num_total_values = len(y)
        
        # If the number of unique values is high compared to the total number of values, consider it continuous
        threshold = 0.05
        if num_unique_values / num_total_values > threshold:
            return True
        else:
            return False
    else:
        return False
    pass

def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """     
    # find all the different outputs
    total = len(Y)
    diff_outputs_dict = {}
    for i,elem in Y.items():
        # print("entropy test 1")
        # print(elem)
        if str(elem) not in diff_outputs_dict.keys():
            diff_outputs_dict[str(elem)] = 1 
        else:
            diff_outputs_dict[str(elem)] = diff_outputs_dict[str(elem)]+1
    ent = 0.0
    for key in diff_outputs_dict.keys():
        val = diff_outputs_dict[str(key)]
        p_i = (val+0.0)/total
        term = -1*p_i*math.log2(p_i)
        ent+=term
    return float(ent)    


def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """

    if (isinstance(Y, list) == False):
        temp_Y = Y.tolist()
    else:
        temp_Y = Y
    total_samples = len(temp_Y)

    temp = np.unique(Y, return_counts=True)

    Y_count = list(temp[1])
    Y_unique = list(temp[0])

    ans = 1

    for attr in Y_unique:
        g = Y_count[Y_unique.index(attr)] / total_samples
        ans -= (g**2)

    return ans
    pass
def mse(Y: pd.Series) -> float:
    """Function to calculate the Mean Squared Error"""
    mean_y = Y.mean()
    mse = ((Y - mean_y) ** 2).mean()
    return mse

def information_gain(Y, attr,criterion):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    assert criterion in ['information_gain', 'gini_index','mse']
    if(criterion=='information_gain'):
        diff_vals = [] 
        for i, elem in attr.items():
            if(str(elem) not in diff_vals):
                diff_vals.append(str(elem))
        redu_entropy = 0.0
        for val in diff_vals:
            storer = []
            sv_len = 0
            for i, elem in attr.items():
                if(str(elem)==val):
                    # get the corresponding label
                    sv_len+=1
                    y_label = Y[i]
                    storer.append(y_label)
            storer_ = pd.Series(storer)
            ent = entropy(storer_)
            redu_entropy += ((sv_len)/len(attr))*ent
        return float(entropy(Y))-float(redu_entropy)
    elif(criterion=='gini_index'):
        diff_vals = [] 
        for i, elem in attr.items():
            if(str(elem) not in diff_vals):
                diff_vals.append(str(elem))
        redu_gini = 0.0
        for val in diff_vals:
            storer = []
            sv_len = 0
            for i, elem in attr.items():
                if(str(elem)==val):
                    # get the corresponding label
                    sv_len+=1
                    y_label = Y[i]
                    storer.append(y_label)
            storer_ = pd.Series(storer)
            ent = gini_index(storer_)
            redu_gini += ((sv_len)/len(attr))*ent
        return float(gini_index(Y))-float(redu_gini)
    else:
        diff_vals = attr.unique()
        redu_mse = 0.0
        for val in diff_vals:
            subset = Y[attr == val]
            redu_mse += (len(subset) / len(attr)) * mse(subset)
        return float(mse(Y)) - float(redu_mse)
        pass



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series) -> str:
    """
    Function to find the optimal attribute to split upon.
    
    Inputs:
    > X: pd.DataFrame of features
    > y: pd.Series of target labels
    > criterion: str, the criterion to use ('information_gain', 'gini_index', 'mse')
    > features: pd.Series of features to consider for splitting
    
    Outputs:
    > Return the best attribute to split upon
    """
    best_gain = -float('inf')
    best_attribute = None
    
    for feature in features:
        attr = X[feature]
        
        if check_ifreal(attr):
            # Continuous feature: sort and consider midpoints between unique values as split points
            unique_vals = np.sort(attr.unique())
            split_points = (unique_vals[:-1] + unique_vals[1:]) / 2  # midpoints between consecutive values
            for split_point in split_points:
                left_split = y[attr <= split_point]
                right_split = y[attr > split_point]
                gain = information_gain(y, pd.Series(np.where(attr <= split_point, 0, 1)), criterion='mse')
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = (feature, split_point)  # tuple to store the feature and the split point
        else:
            # Discrete feature: calculate information gain for the categorical feature
            gain = information_gain(y, attr, criterion='information_gain, gini_index' )
            if gain > best_gain:
                best_gain = gain
                best_attribute = feature
    
    return best_attribute
    pass
def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value):
    """
    Function to split the data according to an attribute.
    
    Inputs:
    - X: pd.DataFrame containing the features.
    - y: pd.Series of the target variable.
    - attribute: The feature to split upon.
    - value: The value of the feature to split upon.
    
    Returns:
    - X_left, y_left: The subset of data where the attribute's value is less than or equal to the split value (for continuous) or equal to the split value (for discrete).
    - X_right, y_right: The subset of data where the attribute's value is greater than the split value (for continuous) or not equal to the split value (for discrete).
    """
    
    # Check if the attribute is continuous (real-valued) or discrete
    is_real = check_ifreal(X[attribute])
    
    if is_real:
        # For continuous attributes, split based on whether the value is <= or >
        mask_left = X[attribute] <= value
        mask_right = X[attribute] > value
    else:
        # For discrete attributes, split based on whether the value is == or !=
        mask_left = X[attribute] == value
        mask_right = X[attribute] != value
    
    # Apply masks to split the data
    X_left = X[mask_left]
    y_left = y[mask_left]
    X_right = X[mask_right]
    y_right = y[mask_right]
    
    return X_left, y_left, X_right, y_right


def reduction_in_variance(Y,attr):
    total_var = Y.var()
    total_len = len(Y)
    diff_vals = [] # diff vals that attr takes like sunny and shit
    for i, elem in attr.iteritems():
        if(str(elem) not in diff_vals):
            diff_vals.append(str(elem))
    redu_var = 0.0
    for val in diff_vals:
        storer = []
        sv_len = 0
        for i,elem in attr.iteritems():
            if(str(elem)==val):
                sv_len+=1
                y_label = Y[i]
                storer.append(y_label)
        storer_ = pd.Series(storer)
        var = storer_.var()
        redu_var+= (((sv_len+0.0)/total_len)*var)
    return float(total_var-redu_var)

def variance(Y):
    # squares mean - mean's square

    if (isinstance(Y, pd.Series)):
        Y = Y.tolist()
    
    total_samples = len(Y)

    Y_squares = [i*i for i in Y]

    sq_mean = sum(Y_squares)/total_samples
    mean_sq = (sum(Y)/total_samples)**2

    return(sq_mean-mean_sq)

def variance_gain(Y, attr):
    """
    Input:
    Output:
    """
    initial_gain = variance(Y)
    
    Y = Y.tolist()
    
    attr = attr.tolist()
    attr_set = set(attr)
    attr_set = list(attr_set)
    
    for i in attr_set:
        l = []
        for j in range (len(attr)):
            if attr[j] == i:
                l.append(Y[j])
        initial_gain = initial_gain-(len(l)/len(Y))*variance(l)
    return initial_gain

def gini_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    Y = Y.tolist()
    attr = attr.tolist()
    attr_set = set(attr)
    attr_set = list(attr_set)
    initial_gain=0
    for i in attr_set:
        l = []
        for j in range (len(attr)):
            if attr[j] == i:
                l.append(Y[j])
        initial_gain = initial_gain+(len(l)/len(Y))*gini_index(l)
    return initial_gain

def loss(Y, split_index):

    if (isinstance(Y, list) == False):
        y = Y.tolist()
    
    total_samples = len(y)

    c1 = 0
    c2 = 0
    for i in range(total_samples):
        if (i <= split_index):
            c1 += y[i]
        else:
            c2 += y[i]
    c1 /= total_samples
    c2 /= total_samples

    loss = 0
    for i in range(total_samples):
        loss += ((y[i] - c1)**2 + (y[i]-c2)**2)
    # print("losss...")
    # print(loss)
    return loss




    

            




        

