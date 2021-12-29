from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
from knn import KNN


class KNeighborsClassifier(KNN):

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
