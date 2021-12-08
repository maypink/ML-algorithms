import numpy as np


def entropy(y):
    _, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def weighted_entropy(root_size, subset):
    return (len(subset)/root_size)*entropy(subset)


def info_gain(root, left_subset, right_subset):
    sum_weighted_entropy = weighted_entropy(len(root), left_subset) + weighted_entropy(len(root), right_subset)
    Q = entropy(root) - sum_weighted_entropy
    return Q