from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class KNN(ABC):
    def __init__(self, k: int = 3):
        self.k = k

    @staticmethod
    def calc_dist(sample: np.ndarray, cur_sample: np.ndarray) -> float:
        euclidian_dist = 0
        for i in range(len(sample) - 1):
            euclidian_dist += (sample[i] - cur_sample[i]) ** 2
        euclidian_dist = euclidian_dist ** 0.5
        return euclidian_dist

    def get_sorted_distances(self, sample: np.ndarray, x: np.ndarray) -> List[Tuple]:
        samples_dist = {}
        for i in range(len(x) - 1):
            cur_dist = self.calc_dist(sample, x[i])
            samples_dist[cur_dist] = i
        sorted_dist = sorted(samples_dist.items(), key=lambda q: q[0])
        return sorted_dist

    @abstractmethod
    def _predict_for_one(self):
        pass

    @abstractmethod
    def predict(self):
        pass

