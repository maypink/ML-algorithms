import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score
from typing import List
from funcs import sigmoid


class LogisticRegression:

    def __init__(self, weights: List[float] = None, is_bias: bool = False, losses: List[float] = None,
                 lr: float = 0.001, max_iter: int = 1000):
        self._weights = weights
        self._is_bias = is_bias
        self._losses = losses
        self._lr = lr
        self._max_iter = max_iter

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

    @staticmethod
    def activate(x: np.ndarray) -> np.ndarray:
        return sigmoid(x)

    @staticmethod
    def loss(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))  # if nan, (1 - y_pred) < 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Эта функция рассчитывает ответ нейрона при предъявлении набора объектов
        :param: X -- матрица объектов размера (n, m), каждая строка - отдельный объект
        :return: вектор размера (n, 1) из нулей и единиц с ответами перцептрона
        """

        if X.shape[1] != self._weights.shape[0]:
            X = np.column_stack((X, np.ones(X.shape[0]).astype(float)))

        return self.activate(X.dot(self._weights))

    def backward_pass(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, learning_rate: float = 0.1) -> None:
        """
        Обновляет значения весов нейрона в соответствие с этим объектом
        :param: X -- матрица объектов размера (n, m)
                y -- вектор правильных ответов размера (n, 1)
                learning_rate - "скорость обучения" (символ alpha в формулах выше)
        В этом методе ничего возвращать не нужно, только правильно поменять веса
        с помощью градиентного спуска.
        """

        grad = (X.transpose().dot(y_pred - y)) / X.shape[0]

        if len(self._weights.shape) == 1:
            self._weights = self._weights.reshape(-1, 1)

        self._weights -= learning_rate * grad

    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 1000) -> np.ndarray:
        """
        :param: X -- матрица объектов размера (n, m)
                y -- вектор правильных ответов размера (n, 1)
                num_epochs -- количество итераций обучения
        :return: loss_values -- вектор значений функции потерь
        """
        if not self._is_bias:
            X = np.column_stack((X, np.ones(X.shape[0]).astype(float)))

        self._weights = np.random.rand(X.shape[1]).reshape(-1, 1)

        loss_values = []

        for i in range(num_epochs):
            # предсказания с текущими весами
            y_pred = self.predict_proba(X)
            # считаем функцию потерь с текущими весами
            loss_values.append(self.loss(y_pred, y))
            # обновляем веса по формуле градиентного спуска
            self.backward_pass(X, y, y_pred)

        return loss_values

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)
        return preds

    def score(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
        return accuracy_score(y, self.predict(X), threshold)

    @classmethod
    def from_pickle(cls, path: Path) -> 'LogisticRegression':
        with open(path, 'rb') as f:
            params = pickle.load(f)
        model = cls(weights=params['weights'], losses=['losses'], lr=params['lr'],
                    max_iter=params['max_iter'])
        model.weights = np.array(params['weights'])
        model.losses = params['losses']
        return model

    def to_pickle(self, path: Path) -> None:
        params = {'weights': self._weights,
                  'losses': self._losses,
                  'lr': self._lr,
                  'max_iter': self._max_iter}
        with open(path, 'wb') as f:
            pickle.dump(params, f)

