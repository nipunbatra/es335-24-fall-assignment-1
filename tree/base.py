"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal,Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *
from collections import Counter
np.random.seed(42)


@dataclass
class Node:
    def __init__(self):
        # pred attribute for the node
        self.feature_label = None
        # pred value of pred attribute for the node
        self.pred_value = None
        # all the children nodes
        self.children = dict()
        # for real input cases left and right nodes
        self.left = None
        self.right = None
        # partition value storage
        self.split_value = None



class DecisionTree:
    criterion: Literal["information_gain", "gini_index","mse"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

        self.in_type = None
        self.out_type = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 


        category = "category"

        lst = []
        print(X.columns)
        for i in list(X.columns):
            lst.append(X[i].dtype.name)

        self.in_type = lst[0]
        self.out_type = y.dtype.name

        # case: DIDO
      
        if ((category in lst) and y.dtype.name == "category"):
            features = list(X.columns)
            out = "out"
            if (out in features):
                features.remove(out)
            temp_X = X
            temp_X['out'] = y
            feature_mat = temp_X
            depth = 0
            self.tree = self.DIDO(feature_mat, None, features, 0)

        # case: DIRO
      
        elif ((category in lst) and y.dtype.name != "category"):
            features = list(X.columns)
            out = "out"
            if (out in features):
                features.remove(out)
            temp_X = X
            temp_X['out'] = y
            feature_mat = temp_X
            depth = 0
            self.tree = self.DIRO(feature_mat, None, features, 0)

        # case: RIDO
        elif ((category not in lst) and y.dtype.name == "category"):
            out = 'out'
            if (out in X.columns):
                X = X.drop(['out'], axis=1)
            self.X = X
            self.y = y
            self.no_of_attributes = len(list(X.columns))
            self.no_of_out_classes = len(set(y))
            feature_mat = X
            output_vec = y
            depth = 0
            self.tree = self.RIDO(feature_mat,output_vec, depth)

        # case: RIRO
        elif ((category not in lst) and y.dtype.name != "category"):
            out = 'out'
            if (out in X.columns):
                X = X.drop(['out'], axis=1)
            self.no_of_attributes = len(list(X.columns))
            feature_mat = X
            output_vec = y
            depth = 0
            self.tree = self.RIRO(feature_mat,output_vec, depth)
        

        '''feature_types = [X[col].dtype.name for col in X.columns]
        self.in_type = feature_types[0]
        self.out_type = y.dtype.name

        # Case 1: Discrete Features, Discrete Output (DIDO)
        if category in feature_types and self.out_type == "category":
            features = [col for col in X.columns if col != 'out']
            self.tree = self.DIDO(X, y, features, 0)

        # Case 2: Discrete Features, Real Output (DIRO)
        elif category in feature_types and self.out_type != "category":
            features = [col for col in X.columns if col != 'out']
            self.tree = self.DIRO(X, y, features, 0)

        # Case 3: Real Features, Discrete Output (RIDO)
        elif category not in feature_types and self.out_type == "category":
            self.tree = self.RIDO(X, y, 0)

        # Case 4: Real Features, Real Output (RIRO)
        elif category not in feature_types and self.out_type != "category":
            self.tree = self.RIRO(X, y, 0)'''


    def _split_riro(self, X, y):

        # if number of unique classes in output vec is <=1, return the unique class
        # and split will be first value in any column
        m = list(np.unique(y))
        if(len(m) <= 1):
            # split = X.loc[0,0]
            # print("returning on <=1 unique o/p class...")
            return m[0], None

        start_loss = 10**8
        best_feature, best_split_threshold = None, None

        for feature in list(X.columns):

            a = X[[feature]].copy()
            #a = pd.DataFrame(a)
            a['out'] = y.copy()
            # print(a)
            a = a.sort_values(by=feature, ascending=True).reset_index(drop=True)
            # print("after sorting")
            # print(a)
            #a = a.reset_index()
            # print("after resetting")
            # print(a)
            #a = a.drop(['index'], axis=1)
            # print("after dropping")
            # print(a)

            classes = a['out']
           # a = a.drop(['out'],axis=1)
            cutoff_values = a[[feature]].copy()

            # print("cutoff values:")
            # print(cutoff_values)

            # print("o/p")
            # print(classes)
            cutoff_values[feature] = pd.to_numeric(cutoff_values[feature], errors='coerce')
            for i in range(1, len(classes)):
                #c = classes[i-1]

                curr_loss = loss(classes, i-1)
                # print("feature:", feature, "curr_loss", curr_loss, "start_loss:", start_loss)

                if (curr_loss < start_loss):
                    start_loss = curr_loss
                    best_feature = feature

                prev_val = cutoff_values.loc[i-1, feature]
                curr_val = cutoff_values.loc[i, feature]
                #print(f"prev_val type: {type(prev_val)}, curr_val type: {type(curr_val)}")
                #print(f"prev_val: {prev_val}, curr_val: {curr_val}")

                if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                    best_split_threshold = round(((prev_val + curr_val) / 2), 6)
                else:
                    # Handle cases where values might be NaN due to conversion issues
                    print(f"Skipping invalid values at index {i-1} or {i}")   
                    #best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
        return best_feature, best_split_threshold

    def RIRO(self, samples, output_vec, depth=0, parent_node=None):

        # if depth limit is notreached ... proceed
        if depth < self.max_depth:

            # find the best split
            # recurse over each feature and each mid point of two consecutive samples
            # check where you get the optimised criteria return that feature and midpoint
            feature, split_value = self._split_riro(samples, output_vec)
            # print("feature:", feature, "split_value: ", split_value)

            if (feature is not None and split_value is not None):

                # split the dataset
                samples['out'] = output_vec
                samples = samples.sort_values(by=feature, ascending=True)

                # print(samples)

                samples = samples.reset_index()

                # print(samples)

                samples = samples.drop(['index'], axis=1)

                # print(samples)

                output_vec = samples['out']
                samples = samples.drop(['out'],axis=1)

                X_l = list()
                y_l = list()
                X_r = list()
                y_r = list()

                for index in range(len(samples)):
                    if (samples.loc[index, feature] <= split_value):
                        X_l.append(samples.loc[index])
                        y_l.append(output_vec[index])
                    else:
                        X_r.append(samples.loc[index])
                        y_r.append(output_vec[index])

                X_l = pd.DataFrame(X_l)
                X_r = pd.DataFrame(X_r)
                y_l = pd.Series(y_l)
                y_r = pd.Series(y_r)

                X_l = X_l.reset_index()
                X_r = X_r.reset_index()
                y_l = y_l.reset_index()
                y_r = y_r.reset_index()

                X_r = X_r.drop(['index'], axis=1)
                X_l = X_l.drop(['index'], axis=1)
                y_r = y_r.drop(['index'], axis=1)
                y_l = y_l.drop(['index'], axis=1)


                # print("printing X_l:")
                # print(X_l)
                # print("printing X_r:")
                # print(X_r)
                # print("printing y_l:")
                # print(y_l)
                # print("printing y_r:")
                # print(y_r)

                # create a new node for the pred feature
                node = Node()
                node.feature_label = feature
                node.split_value = split_value
                node.pred_value = round(float(output_vec.mean()),6)

                # if the new datasets are not empty, recusrse over them
                if (len(X_l)!=0 and len(y_l)!=0):
                    node.left = self.RIRO(X_l, y_l, depth+1, node)
                if (len(X_r)!=0 and len(y_r)!=0):
                    node.right = self.RIRO(X_r, y_r, depth+1, node)

            # if the output vec was having only one unique value
            # then create a leaf node and return
            elif (feature is not None and split_value is None):
                node = Node()
                node.pred_value = feature
                return node
            return node
        
        # if depth limit is reached 
        # create a leaf node with pred value of max occuring class of o/p vec
        else:
            node = Node()
            node.pred_value = round(float(output_vec.iloc[0].mean()),6)
            # print("depth limit reached:...", ans)
            return node

    def _best_split(self, X, y):

        m = list(np.unique(y))
        # if number of unique classes in output vec is <=1, return the unique class
        # and split will be first value in any column
        if(len(m) == 1):
            # split = X.loc[0,0]
            # print("returning on <=1 unique o/p class...")
            return m[0], None

        # if (self.criterion == "information_gain"):
        #     temp_inf_gain = entropy(y)
        #     start_gain = temp_inf_gain
        #     print("start_gain:", start_gain)
        # else:
        #     start_gini = gini_index(y)
        #     print("start_gini:", start_gini)

        best_feature, best_split_threshold = None, None

        if (self.criterion == "information_gain"):
            start_gain = -10**8
        else:
            start_gain = 10**8

        for feature in list(X.columns):

            a = X[feature]
            a = pd.DataFrame(a)
            a['out'] = y
            # print(a)
            a = a.sort_values(by=feature, ascending=True)
            # print("after sorting")
            # print(a)
            a = a.reset_index()
            # print("after resetting")
            # print(a)
            a = a.drop(['index'], axis=1)
            # print("after dropping")
            # print(a)

            classes = a['out']

            temp_c = np.unique(classes, return_counts=True)
            temp_classes = list(temp_c[0])
            classes_count = list(temp_c[1])

            # print("temp_classes:", temp_classes)
            # print("classes_count:", classes_count)

            self.no_of_out_classes = len(temp_classes)

            # print("no_of_out_classes:", self.no_of_out_classes)

            a = a.drop(['out'],axis=1)
            cutoff_values = a

            # print("cutoff values:")
            # print(cutoff_values)

            # print("o/p")
            # print(classes)

            labels_before = dict()
            for i in range(self.no_of_out_classes):
                labels_before[temp_classes[i]] = 0
            # labels_before = [0]*self.no_of_out_classes

            # print("labels_before:", labels_before)

            labels_after = dict()

            for elem in range(self.no_of_out_classes):
                labels_after[temp_classes[elem]] = classes_count[elem]  #.append(classes.tolist().count(elem))
                # print("labels_after:",labels_after)

            # print(labels_after)

            for i in range(1, len(classes)):
                
                c = classes[i-1]
                labels_before[c]+=1
                labels_after[c]-=1

                if (self.criterion == "information_gain"):
                    gain_left = entropy(pd.Series(list(labels_before.values())))
                    gain_right = entropy(pd.Series(list(labels_after.values())))
                    # print(gain_right, gain_left)
                    # print(gain_left+gain_right)
                    gain_temp = entropy(y) - (i * gain_left + (len(classes) - i) * gain_right) / len(classes)
                    
                    if (start_gain < gain_temp):
                        start_gain = gain_temp
                        best_feature = feature
                        best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
                else:
                    gini_left = gini_index(pd.Series(list(labels_before.values())))
                    gini_right = gini_index(pd.Series(list(labels_after.values())))

                    #print(f"i: {i}, gini_left: {gini_left}, gini_right: {gini_right}, len(classes): {len(classes)}")
                    if (gini_left or gini_right) is None or len(classes)==0: 
                    
                        return 0
                    else:
                        gini_index_temp = (i * gini_left + (len(classes) - i) * gini_right) / len(classes)

                    # print("gini_index_temp:", gini_index_temp)
                    if (gini_index_temp <= start_gain):
                        start_gain = gini_index_temp
                        best_feature = feature
                        best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
            # print("curr_feat:", feature ,"start_gain:", start_gain)
        # print("best_start_gain:", start_gain)
        return best_feature, best_split_threshold
    



    def RIDO(self, samples, output_vec, depth=0, parent_node=None):

        # if depth limit is notreached ... proceed
        if depth < self.max_depth:

            # find the best split
            # recurse over each feature and each mid point of two consecutive samples
            # check where you get the optimised criteria return that feature and midpoint
            feature, split_value = self._best_split(samples, output_vec)
            # print("feature:", feature, "split_value: ", split_value)

            # if feature splitvalue found
            if (feature is not None and split_value is not None):

                # split the dataset
                samples['out'] = output_vec
                samples = samples.sort_values(by=feature, ascending=True)

                # print(samples)

                samples = samples.reset_index()

                # print(samples)

                samples = samples.drop(['index'], axis=1)

                # print(samples)

                output_vec = samples['out']
                samples = samples.drop(['out'],axis=1)

                X_l = list()
                y_l = list()
                X_r = list()
                y_r = list()

                for index in range(len(samples)):
                    if (samples.loc[index, feature] <= split_value):
                        X_l.append(samples.loc[index])
                        y_l.append(output_vec[index])
                    else:
                        X_r.append(samples.loc[index])
                        y_r.append(output_vec[index])

                X_l = pd.DataFrame(X_l)
                X_r = pd.DataFrame(X_r)
                y_l = pd.Series(y_l)
                y_r = pd.Series(y_r)

                X_l = X_l.reset_index()
                X_r = X_r.reset_index()
                y_l = y_l.reset_index()
                y_r = y_r.reset_index()

                X_r = X_r.drop(['index'], axis=1)
                X_l = X_l.drop(['index'], axis=1)
                y_r = y_r.drop(['index'], axis=1)
                y_l = y_l.drop(['index'], axis=1)

                # print("printing X_l:")
                # print(X_l)
                # print("printing X_r:")
                # print(X_r)
                # print("printing y_l:")
                # print(y_l)
                # print("printing y_r:")
                # print(y_r)

                # create a new node for the pred feature
                node = Node()
                node.feature_label = feature
                node.split_value = split_value
                node.pred_value = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
                
                # if the new datasets are not empty, recusrse over them
                if (len(X_l)!=0 and len(y_l)!=0):
                    node.left = self.RIDO(X_l, y_l, depth+1, node)
                if (len(X_r)!=0 and len(y_r)!=0):
                    node.right = self.RIDO(X_r, y_r, depth+1, node)
            
            # if the output vec was having only one unique value
            # then create a leaf node and return
            elif (feature is not None and split_value is None):
                node = Node()
                node.pred_value = feature
                # node.split_value = X[]
                return node
            return node
        
        # if depth limit is reached 
        # create a leaf node with pred value of max occuring class of o/p vec
        else:
            if (len(output_vec) == 0):
                node = Node()
                node.pred_value = None
                # print("null output_vec returning ...")
                return node
            temp = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
            node = Node()
            node.pred_value = temp
            # print("depth limit reached returning ...", temp)
            return node


    def DIRO(self, samples, target_attr, attributes, depth):

        output_vec = samples['out']

        # if depth limit is notreached ... proceed
        if (depth < self.max_depth):
            
            # if # of unique classes in output vec <=1
            # return the unique class
            if (len(output_vec.unique()) <= 1):
                temp = output_vec.unique()[0]
                # print("output_vec returning...", temp)
                return temp
            
            # if all the attributes are used up then
            # return mean of output vec
            elif (len(attributes) == 0):
                temp = sum(output_vec)/len(output_vec)
                # print("attributes returning...", temp)
                return temp
            
            else:
                a = list()
                # print("attributes:")
                # print(attributes)

                # calculate criteria for each feature
                # check which feature gives you minimum confusion for prediction
                for x in attributes:
                    attr = samples[x]
                    var_gain = variance_gain(output_vec, attr)
                    a.append(var_gain)

                best_attr = attributes[a.index(max(a))]

                # print("best_attr", best_attr)

                # create a node for the predicted best feature
                root = Node()
                root.feature_label = best_attr

                # recurse over remainning features with new data
                for x in samples[best_attr].unique():
                    new_data = samples[samples[best_attr]==x]
                    new_data = new_data.reset_index()
                    new_data = new_data.drop(['index'], axis=1)
                    # print(new_data)

                    if (len(new_data) == 0):
                        root.children[x] = sum(output_vec)/len(output_vec)
                    else:
                        temp_attr = []
                        for y in attributes:
                            if (y!=best_attr):
                                temp_attr.append(y)

                        subtree = self.DIRO(new_data, best_attr, temp_attr, depth+1)

                        root.children[x] = subtree

                return root
        # if depth limit reached...
        # return meanof output vec
        else:
            temp = sum(output_vec)/len(output_vec)
            return temp

    def DIDO(self, samples, target_attr, attributes, depth):
        """For Discrete input, discrete output
            Output: a tree
        """
        output_vec = samples['out']

        # if depth limit is not reached, proceed ...
        if (depth < self.max_depth):

            # if # of unique classes in output vec <=1
            # return the unique class
            if (len(list(output_vec.unique())) <= 1):
                temp = list(output_vec.unique())[0]
                return temp

            # if all the attributes are used up then
            # return max occuring class in output vec
            elif (len(attributes) == 0):
                temp = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
                return temp

            else:
                a = list()

                # calculate criteria for each feature
                # check which feature gives you minimum confusion for prediction
                if (self.criterion == "information_gain"):
                    for x in attributes:
                        attr = samples[x]

                        inf_gain = information_gain(output_vec,attr,criterion='information_gain')

                        a.append(inf_gain)
                        # print("x", x, "info_gain", inf_gain)
                    best_attr = attributes[a.index(max(a))]

                elif (self.criterion == "gini_index"):
                    for x in attributes:
                        attr = samples[x]

                        g_gain = gini_gain(output_vec,attr)

                        a.append(g_gain)
                        # print("x", x, "info_gain", inf_gain)
                    best_attr = attributes[a.index(max(a))]

                # print("best_attr", best_attr)
                # create a node for the predicted best feature
                root = Node()
                root.feature_label = best_attr
                
                # recurse over remainning features with new data
                for x in samples[best_attr].unique():
                    new_data = samples[samples[best_attr]==x]
                    # print(new_data)

                    if (len(new_data) == 0):
                        root.children[x] = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
                    else:
                        temp_attr = []
                        for y in attributes:
                            if (y!=best_attr):
                                temp_attr.append(y)

                        subtree = self.DIDO(new_data, best_attr, temp_attr, depth+1)

                        root.children[x] = subtree

                return root
        # if depth limit reached...
        # return most occuring output class
        else:
            temp = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
            return temp
        



    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
     # case: DIDO
        if (self.in_type == "category" and self.out_type == "category"):
            y_hat = list()

            # if depth of tree is zero and
            # only a class is been predicted
            if (type(self.tree) != Node):
                y_hat.append(self.tree)
                return pd.Series(y_hat)

            attributes = list(X.columns)
            attributes.remove('out')
            # print(attributes)

            for i in range(len(X)):
                tree = self.tree
                # print("row...", i)
                data = list(X.loc[i])
                while(1):
                    curr_feat = tree.feature_label
                    # print("curr_feat:", curr_feat)
                    curr_val = data[curr_feat]
                    # print("curr_val:", curr_val)
                    if (type(tree.children[curr_val]) == Node):
                        tree = tree.children[curr_val]
                    else:
                        y_hat.append(tree.children[curr_val])
                        break

            y_hat = pd.Series(y_hat)
            return y_hat

        # case: DIRO
        elif (self.in_type == "category" and self.out_type != "category"):
            y_hat = list()

            attributes = list(X.columns)
            attributes.remove('out')
            # print(attributes)

            for i in range(len(X)):
                tree = self.tree
                # print("row...", i)
                data = list(X.loc[i])
                while(1):
                    curr_feat = tree.feature_label
                    # print("curr_feat:", curr_feat)
                    curr_val = data[curr_feat]
                    # print("curr_val:", curr_val)
                    if (type(tree.children[curr_val]) == Node):
                        tree = tree.children[curr_val]
                    else:
                        y_hat.append(tree.children[curr_val])
                        break

            y_hat = pd.Series(y_hat)
            return y_hat

        # case: RIDO
        elif (self.in_type != "category" and self.out_type == "category"):
            y_hat = list()

            attributes = list(X.columns)
            out = 'out'
            if (out in attributes):
                attributes.remove('out')
            # print(attributes)

            for i in range(len(X)):
                tree = self.tree
                while(1):
                    curr_node_feature = tree.feature_label
                    # print("curr_fe", curr_node_feature, type(curr_node_feature), tree.split_value)
                    if (curr_node_feature==None):
                        break
                    sample_val = X.iloc[i, curr_node_feature]
                    if  (sample_val  is  None or tree.split_value is None):
                        return 0
                    elif(sample_val or tree.split_value)is not None and  (sample_val<=tree.split_value):
                            if (tree.left != None):
                                tree = tree.left
                            else:
                                break
                    
                    else:
                        if (tree.right != None):
                            tree = tree.right
                        else:
                            break

                y_hat.append(tree.pred_value)

            y_hat = pd.Series(y_hat)
            return y_hat

        # case: RIRO
        elif (self.in_type != "category" and self.out_type != "category"):
            y_hat = list()

            # attributes = list(X.columns)
            # out = 'out'
            # if (out in attributes):
            #     attributes.remove('out')
            # # print(attributes)

            for i in range(len(X)):
                tree = self.tree
                while(1):
                    curr_node_feature = tree.feature_label
                    if (curr_node_feature==None):
                        break
                    sample_val = X.loc[i, curr_node_feature]
                    if (sample_val <= tree.split_value):
                        if (tree.left != None):
                            tree = tree.left
                        else:
                            break
                    else:
                        if (tree.right != None):
                            tree = tree.right
                        else:
                            break
                
                # print(tree.pred_value)
                y_hat.append(tree.pred_value)

            y_hat = pd.Series(y_hat)
            # print(y_hat)
            return y_hat
        pass


    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        if self.in_type == "category":

            tree = self.tree

            def printdict(d, indent=0):

                print("\t"*(indent-1) + "feature:" +str(d.feature_label))

                for key, value in d.children.items():

                    print('\t' * indent + "\t" + "feat_value:" +  str(key))

                    if isinstance(value, Node):
                        printdict(value, indent+1)
                    else:
                        print('\t' * (indent+1) + str(value))

            printdict(tree)

        elif (self.in_type != "category"):
            tree = self.tree
            def printdict(d, indent=0):

                if (isinstance(d.left, Node)):
                    print("\t"*(indent) + "feature:" +str(d.feature_label) + "\t" + "split value:" + str(d.split_value))
                    printdict(d.left, indent+1)

                    # print('\t' * (indent+1) + str(d.pred_value))

                if (isinstance(d.right, Node)):
                    print("\t"*(indent) + "feature:" +str(d.feature_label) + "\t" + "split value:" + str(d.split_value))
                    printdict(d.right, indent+1)

                if (isinstance(d.right, Node) == False and isinstance(d.left, Node) == False):
                    print('\t' * (indent+1) + str(d.pred_value))

            printdict(tree)
        pass
