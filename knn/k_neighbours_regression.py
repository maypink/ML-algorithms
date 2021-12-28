from tqdm import tqdm
import numpy as np
from typing import List


class KNeighborsRegression:
    def __init__(self, k: int = 3):
        self.k = k

    def dist(self, sample: np.ndarray, cur_sample: np.ndarray) -> float:
        euclidian_dist = 0
        for i in range(len(sample) - 1):
            euclidian_dist += (sample[i] - cur_sample[i]) ** 2
        euclidian_dict = euclidian_dist ** 0.5
        return euclidian_dist

    def weighted_average(self, sorted_dist: List[float], y: np.ndarray) -> float:
        num = 0
        for (dist, idx) in sorted_dist:
            num += dist ** (-1) * y[idx]
        denom = sum(list(dist ** (-1) for dist, _ in sorted_dist))
        return num / denom

    def _predict(self, sample: np.ndarray, X: np.ndarray, y: np.ndarray) -> int:
        samples_dist = {}
        for i in range(len(X) - 1):
            cur_dist = self.dist(sample, X[i])
            samples_dist[cur_dist] = i
        sorted_dist = sorted(samples_dist.items(), key=lambda x: x[0])
        pred = self.weighted_average(sorted_dist[:self.k], y)
        return pred

    def predict_proba(self, samples: np.ndarray, X: np.ndarray, y: np.ndarray) -> List[int]:
        preds = []
        for sample in samples:
            preds.append(self._predict(sample, X, y))
        return preds

    def predict(self, samples: np.ndarray, X: np.ndarray, y: np.ndarray, th: float = 0.5) -> List[int]:
        preds = []
        for sample in tqdm(samples):
            pred = (self._predict(sample, X, y) > th)
            preds.append(pred)
        return preds
