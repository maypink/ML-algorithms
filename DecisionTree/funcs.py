import numpy as np
import pandas as pd


def entropy(y: pd.Series):
    _, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()
    probabilities = list(i+0.001 for i in probabilities)
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def weighted_entropy(root_size: int, subset: pd.Series):
    return (len(subset)/root_size)*entropy(subset)


def info_gain(root: pd.Series, left_subset: pd.Series, right_subset: pd.Series):
    sum_weighted_entropy = weighted_entropy(len(root), left_subset) + weighted_entropy(len(root), right_subset)
    Q = entropy(root) - sum_weighted_entropy
    return Q