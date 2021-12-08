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

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_losses(self) -> np.ndarray:
        return self.losses

    def check_stop_criteria(self, old_weights: np.ndarray,
                            preds: np.ndarray, labels: np.ndarray):
        if (abs(preds - labels)).sum() < self.eps:
            return True
        if (abs(old_weights - self.weights)).sum() < self.eps:
            return True
        return False

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
            grad = -(labels - preds).dot(features) / (2.0 * features.shape[0])
            self.weights = self.weights - self.lr * grad

            if self.check_stop_criteria(old_weights, preds, labels):
                break

        return self

    def load_weights(self, path: Path) -> None:
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.max_iter = params['max_iter']
        self.lr = params['lr']
        self.eps = params['eps']
        self.weights = np.array(params['weights'])
        self.losses = params['losses']

    def save_weights(self, path: Path) -> None:
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
