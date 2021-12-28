from tqdm import tqdm
import numpy as np
from typing import List


class KNeighborsClassifier:
    def __init__(self, k: int = 3):
        self.k = k

    def dist(self, sample: np.ndarray, cur_sample: np.ndarray) -> float:
        euclidian_dist = 0
        for i in range(len(sample) - 1):
            euclidian_dist += (sample[i] - cur_sample[i]) ** 2
        euclidian_dict = euclidian_dist ** 0.5
        return euclidian_dist

    def _predict(self, sample: np.ndarray, X: np.ndarray, y: np.ndarray) -> int:
        samples_dist = {}
        for i in range(len(X) - 1):
            cur_dist = self.dist(sample, X[i])
            samples_dist[cur_dist] = i
        sorted_dist = sorted(samples_dist.items(), key=lambda x: x[0])
        idxs = list(i[1] for i in sorted_dist[:(self.k)])
        neighbors = list(y[i] for i in idxs)
        return max(set(neighbors), key=neighbors.count)

    def predict(self, samples: np.ndarray, X: np.ndarray, y: np.ndarray) -> List[int]:
        preds = []
        for sample in tqdm(samples):
            preds.append(self._predict(sample, X, y))
        return preds
