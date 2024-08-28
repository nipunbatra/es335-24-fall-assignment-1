from dataclasses import dataclass
from typing import Literal, Union, Optional
import numpy as np
import pandas as pd
from tree.utils import entropy, gini_index

np.random.seed(42)

@dataclass
class TreeNode:
    """
    A class to represent a node in the decision tree
    """
    is_leaf: bool
    prediction: Optional[Union[float, int, str]] = None
    split_feature: Optional[int] = None
    split_value: Optional[Union[float, str]] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None

class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int
    root: Optional[TreeNode]

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def _calculate_criterion(self, y):
        """
        Calculate the chosen criterion for a given set of labels
        """
        if self.criterion == "information_gain":
            return entropy(y)
        elif self.criterion == "gini_index":
            return gini_index(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _find_best_split(self, X, y):
        """
        Find the best feature and value to split on based on the chosen criterion
        """
        best_feature = None
        best_value = None
        best_criterion_value = float('inf') if self.criterion == "gini_index" else -float('inf')
        best_splits = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            # Get the unique values of the feature column
            values = X.iloc[:, feature].unique()
            
            # Check if the feature is categorical
            if pd.api.types.is_categorical_dtype(X.iloc[:, feature]) or isinstance(values[0], str):
                # Handle categorical features by comparing equality
                for value in values:
                    left_indices = X.iloc[:, feature] == value
                    right_indices = ~left_indices
                    left_y, right_y = y[left_indices], y[right_indices]

                    if len(left_y) == 0 or len(right_y) == 0:
                        continue

                    if self.criterion == "information_gain":
                        criterion_value = (self._calculate_criterion(y) -
                                           len(left_y)/len(y) * self._calculate_criterion(left_y) -
                                           len(right_y)/len(y) * self._calculate_criterion(right_y))
                        is_better = criterion_value > best_criterion_value
                    else:  # gini_index
                        criterion_value = (len(left_y)/len(y) * self._calculate_criterion(left_y) +
                                           len(right_y)/len(y) * self._calculate_criterion(right_y))
                        is_better = criterion_value < best_criterion_value

                    if is_better:
                        best_criterion_value = criterion_value
                        best_feature = feature
                        best_value = value
                        best_splits = (left_indices, right_indices)
            else:
                # Handle numerical features by comparing inequality
                for value in values:
                    left_indices = X.iloc[:, feature] <= value
                    right_indices = ~left_indices
                    left_y, right_y = y[left_indices], y[right_indices]

                    if len(left_y) == 0 or len(right_y) == 0:
                        continue

                    if self.criterion == "information_gain":
                        criterion_value = (self._calculate_criterion(y) -
                                           len(left_y)/len(y) * self._calculate_criterion(left_y) -
                                           len(right_y)/len(y) * self._calculate_criterion(right_y))
                        is_better = criterion_value > best_criterion_value
                    else:  # gini_index
                        criterion_value = (len(left_y)/len(y) * self._calculate_criterion(left_y) +
                                           len(right_y)/len(y) * self._calculate_criterion(right_y))
                        is_better = criterion_value < best_criterion_value

                    if is_better:
                        best_criterion_value = criterion_value
                        best_feature = feature
                        best_value = value
                        best_splits = (left_indices, right_indices)

        return best_feature, best_value, best_splits

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree
        """
        if len(y.unique()) == 1:
            return TreeNode(is_leaf=True, prediction=y.iloc[0])

        if depth == self.max_depth:
            return TreeNode(is_leaf=True, prediction=y.mode()[0])

        best_feature, best_value, best_splits = self._find_best_split(X, y)

        if best_feature is None:
            return TreeNode(is_leaf=True, prediction=y.mode()[0])

        left_indices, right_indices = best_splits
        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(is_leaf=False, split_feature=best_feature, split_value=best_value, left=left_node, right=right_node)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.root = self._build_tree(X, y)

    def _predict_sample(self, x, node):
        """
        Predict the class for a single sample
        """
        if node.is_leaf:
            return node.prediction
        if isinstance(node.split_value, (int, float)):
            if x[node.split_feature] <= node.split_value:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
        else:
            if x[node.split_feature] == node.split_value:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(lambda x: self._predict_sample(x, self.root), axis=1)

    def _plot_tree(self, node, depth=0):
        if node.is_leaf:
            print("\t" * depth, f"Leaf: {node.prediction}")
        else:
            print("\t" * depth, f"?(X{node.split_feature} == {node.split_value})" if isinstance(node.split_value, str)
                  else f"?(X{node.split_feature} <= {node.split_value})")
            self._plot_tree(node.left, depth + 1)
            self._plot_tree(node.right, depth + 1)

    def plot(self) -> None:
        self._plot_tree(self.root)

