from tqdm import tqdm
import numpy as np
from typing import List


class KNeighborsClassifier:
    def __init__(self, k: int = 3):
        self.k = k

    @staticmethod
    def calc_dist(sample: np.ndarray, cur_sample: np.ndarray) -> float:
        euclidian_dist = 0
        for i in range(len(sample) - 1):
            euclidian_dist += (sample[i] - cur_sample[i]) ** 2
        euclidian_dict = euclidian_dist ** 0.5
        return euclidian_dist

    def get_sorted_distances(self, sample: np.ndarray, x: np.ndarray):
        samples_dist = {}
        for i in range(len(x) - 1):
            cur_dist = self.calc_dist(sample, x[i])
            samples_dist[cur_dist] = i
        sorted_dist = sorted(samples_dist.items(), key=lambda q: q[0])
        return sorted_dist

    def _predict_for_one(self, sample: np.ndarray, x: np.ndarray, y: np.ndarray) -> int:
        sorted_dist = self.get_sorted_distances(sample, x)
        idxs = list(i[1] for i in sorted_dist[:self.k])
        neighbors = list(y[i] for i in idxs)
        return max(set(neighbors), key=neighbors.count)

    def predict(self, samples: np.ndarray, x: np.ndarray, y: np.ndarray) -> List[int]:
        preds = []
        for sample in tqdm(samples):
            preds.append(self._predict_for_one(sample, x, y))
        return preds
