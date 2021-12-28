from tqdm import tqdm
import numpy as np
from typing import List
from k_neighbors_classifier import KNeighborsClassifier


class KNeighborsRegression(KNeighborsClassifier):

    @staticmethod
    def weighted_average(sorted_dist: List[float], y: np.ndarray) -> float:
        num = 0
        for (dist, idx) in sorted_dist:
            num += dist ** (-1) * y[idx]
        denom = sum(list(dist ** (-1) for dist, _ in sorted_dist))
        return num / denom

    def _predict_for_one(self, sample: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        sorted_dist = self.get_sorted_distances(sample, x)
        pred = self.weighted_average(sorted_dist[:self.k], y)
        return pred

    def predict_proba(self, samples: np.ndarray, x: np.ndarray, y: np.ndarray) -> List[float]:
        preds = []
        for sample in samples:
            preds.append(self._predict_for_one(sample, x, y))
        return preds

    def predict(self, samples: np.ndarray, x: np.ndarray, y: np.ndarray, th: float = 0.5) -> List[bool]:
        preds = []
        for sample in tqdm(samples):
            pred = (self._predict_for_one(sample, x, y) > th)
            preds.append(pred)
        return preds