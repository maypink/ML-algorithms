from pathlib import Path
import numpy as np
import pickle


class LinearRegression:
    def __init__(self, bias: bool = False, lr: float = 0.001, max_iter: int = 100, eps: float = 0.01):
        self.bias = bias
        self.lr = lr
        self.max_iter = max_iter
        self.weights = None
        self.losses = []
        self.eps = eps

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def losses(self):
        return self._losses

    @losses.setter
    def losses(self, losses):
        self._losses = losses

    def is_should_stop(self, old_weights: np.ndarray,
                            preds: np.ndarray, labels: np.ndarray):
        if (abs(preds - labels)).sum() < self.eps:
            return True
        if (abs(old_weights - self.weights)).sum() < self.eps:
            return True
        return False

    @staticmethod
    def _count_grad(features: np.ndarray, labels: np.ndarray, preds: np.ndarray):
        grad = 2 * (preds - labels).dot(features[:, 1])/(features.shape[0])
        return grad

    def fit(self, features: np.ndarray, labels: np.ndarray):
        if not self.bias:
            features = np.column_stack((features, np.ones(features.shape[0]).astype(float)))

        self.weights = np.random.rand(features.shape[1])

        for i in range(self.max_iter):
            old_weights = self.weights.copy()
            preds = self.predict(features)

            cur_loss = self.mse_loss(preds, labels)
            if cur_loss == float('inf'):
                self.weights = old_weights
                break

            self.losses.append(cur_loss)
            grad = self._count_grad(features, labels, preds)
            self.weights = self.weights - self.lr * grad

            if self.is_should_stop(old_weights, preds, labels):
                break

        return self

    @classmethod
    def from_pickle(cls, path: Path) -> 'LinearRegression':
        with open(path, 'rb') as f:
            params = pickle.load(f)
        model = cls(lr=params['lr'], max_iter=params['max_iter'],
                    eps=params['eps'])
        model.weights = np.array(params['weights'])
        model.losses = params['losses']
        return model

    def to_pickle(self, path: Path) -> None:
        params = {'max_iter': self.max_iter,
                  'lr': self.lr,
                  'eps': self.eps,
                  'weights': self.weights,
                  'losses': self.losses}
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def mse_loss(labels: np.ndarray, preds: np.ndarray) -> float:
        dif = np.power(preds - labels, 2)
        loss = dif.sum() / len(preds)
        return loss

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] != self.weights.shape[0]:
            features = np.column_stack((features, np.ones(features.shape[0]).astype(float)))
        return np.dot(features, self.weights)
