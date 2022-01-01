import numpy as np


class SVM:
    def __init__(self, lr: float = 0.01, alpha: float = 0.1, epochs: int = 150, bias: bool = False):
        self._lr = lr
        self._alpha = alpha
        self._epochs = epochs
        self._bias = bias
        self._weights = None
        self.losses = []

    def fit(self, x: np.array, y: np.array) -> 'SVM':
        if not self._bias:
            x = np.column_stack((x, np.ones(x.shape[0]).astype(float)))

        self._weights = np.random.normal(loc=0, scale=0.05, size=x.shape[1])

        for epoch in range(self._epochs):
            cur_losses = []
            for i in range(len(x) - 1):
                margin = self.calc_margin(x[i], y[i])
                if margin >= 1:
                    grad = self.calc_grad(x[i], y[i], 1)
                    self._weights -= grad
                else:
                    grad = self.calc_grad(x[i], y[i], -1)
                    self._weights -= grad
                cur_losses.append(self.hinge_loss(x[i], y[i]))
            self.losses.append(np.mean(cur_losses))
        return self

    def calc_grad(self, x: np.array, y: np.array, margin_sign: int) -> np.array:
        if margin_sign == 1:
            return self._lr * self._alpha * self._weights
        else:
            return self._lr * (self._alpha * self._weights / self._epochs - y * x)

    def calc_margin(self, x: np.array, y: int) -> float:
        return y * np.dot(self._weights, x)

    def _predict_for_one(self, x: np.array) -> float:
        return np.sign(np.dot(self._weights, x))

    def predict(self, x: np.array) -> np.array:
        if x.shape[1] != self._weights.shape[0]:
            x = np.column_stack((x, np.ones(x.shape[0]).astype(float)))
        preds = []
        for cur_x in x:
            preds.append(self._predict_for_one(cur_x))
        return preds

    def hinge_loss(self, x: np.array, y: np.array) -> float:
        return max(0, 1 - y * np.dot(x, self._weights))
