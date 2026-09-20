"""Линейные модели, обучаемые градиентным спуском (реализация «с нуля»).

Класс :class:`GradientDescentClassifier` реализует две модели:

* ``loss="mse"``     -- линейная регрессия на one-hot кодированные метки
  (постановка из задания: «обучить модель линейной регрессии»); предсказанием
  класса считается argmax по выходам линейных функций;
* ``loss="logloss"`` -- softmax-регрессия (многоклассовая логистическая),
  обучается минимизацией перекрёстной энтропии.

Обе модели обучаются одним и тем же итеративным алгоритмом (пакетный
градиентный спуск), поэтому их можно честно сравнивать по графикам сходимости.
На каждой итерации сохраняется история: значение функции потерь, accuracy,
precision, L2-норма вектора весов ||W|| и L2-норма его изменения ||dW||.

Асимптотическая сложность обучения: O(iterations * n * m * k),
где n -- число объектов, m -- число признаков, k -- число классов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import ClassificationMetrics


class StandardScaler:
    """Стандартизация признаков: z = (x - mean) / std.

    Собственная реализация, чтобы не зависеть от scikit-learn в той части
    работы, которую требуется написать вручную.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)   # защита от деления на ноль
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler не обучен: вызовите fit()")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class GradientDescentClassifier:
    """Линейный классификатор, обучаемый пакетным градиентным спуском."""

    def __init__(
        self,
        class_names: list[str],
        loss: str = "logloss",
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        l2: float = 0.0,
        standardize: bool = True,
        random_state: int = 42,
    ) -> None:
        if loss not in ("mse", "logloss"):
            raise ValueError("loss должен быть 'mse' или 'logloss'")
        self.class_names = list(class_names)
        self.loss = loss
        self.n_iterations = int(n_iterations)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.standardize = standardize
        self.random_state = random_state

        self.metrics = ClassificationMetrics(self.class_names)
        self.scaler: StandardScaler | None = None
        self.W: np.ndarray | None = None          # веса, форма (m + 1, k)
        self.history: list[dict[str, float]] = []

    # -------------------------------------------------------------- сервис
    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def _prepare(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Стандартизация и добавление столбца единиц (свободный член)."""
        X = np.asarray(X, dtype=float)
        if self.standardize:
            if fit:
                self.scaler = StandardScaler().fit(X)
            if self.scaler is None:
                raise RuntimeError("Модель не обучена: вызовите fit()")
            X = self.scaler.transform(X)
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=int)
        one_hot = np.zeros((len(y), self.n_classes), dtype=float)
        one_hot[np.arange(len(y)), y] = 1.0
        return one_hot

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        """Численно устойчивый softmax по строкам."""
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, Xb: np.ndarray) -> np.ndarray:
        """Выход модели: линейные значения либо вероятности классов."""
        scores = Xb @ self.W
        return scores if self.loss == "mse" else self._softmax(scores)

    def _loss_value(self, output: np.ndarray, Y: np.ndarray) -> float:
        if self.loss == "mse":
            base = float(np.mean(np.sum((output - Y) ** 2, axis=1)))
        else:
            base = float(-np.mean(np.sum(Y * np.log(np.clip(output, 1e-12, 1.0)), axis=1)))
        if self.l2 > 0:
            base += self.l2 * float(np.sum(self.W[:-1] ** 2))
        return base

    # ------------------------------------------------------------ обучение
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> "GradientDescentClassifier":
        """Обучает модель и заполняет историю обучения по итерациям."""
        Xb = self._prepare(X_train, fit=True)
        Y = self._one_hot(y_train)
        y_train = np.asarray(y_train, dtype=int)
        n_samples = Xb.shape[0]

        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(scale=0.01, size=(Xb.shape[1], self.n_classes))
        self.history = []

        has_test = X_test is not None and y_test is not None
        if has_test:
            Xb_test = self._prepare(X_test, fit=False)
            Y_test = self._one_hot(y_test)
            y_test = np.asarray(y_test, dtype=int)

        for iteration in range(1, self.n_iterations + 1):
            output = self._forward(Xb)

            # Градиент по форме одинаков для MSE (с множителем 2) и softmax.
            error = output - Y
            scale = 2.0 if self.loss == "mse" else 1.0
            grad = scale * (Xb.T @ error) / n_samples
            if self.l2 > 0:
                grad[:-1] += 2 * self.l2 * self.W[:-1]

            step = self.learning_rate * grad
            self.W -= step

            # --- запись истории обучения после шага ---
            output = self._forward(Xb)
            y_pred = output.argmax(axis=1)
            record: dict[str, float] = {
                "iteration": iteration,
                "train_loss": self._loss_value(output, Y),
                "train_accuracy": self.metrics.accuracy(y_train, y_pred),
                "train_precision_macro": self.metrics.macro_precision(y_train, y_pred),
                "weight_norm_l2": float(np.linalg.norm(self.W)),
                "weight_update_norm_l2": float(np.linalg.norm(step)),
            }
            if has_test:
                out_test = self._forward(Xb_test)
                y_pred_test = out_test.argmax(axis=1)
                record["test_loss"] = self._loss_value(out_test, Y_test)
                record["test_accuracy"] = self.metrics.accuracy(y_test, y_pred_test)
                record["test_precision_macro"] = self.metrics.macro_precision(y_test, y_pred_test)
                for name, value in zip(
                    self.class_names, self.metrics.precision_per_class(y_test, y_pred_test)
                ):
                    record["test_precision_" + name] = float(value)
            self.history.append(record)

        return self

    # --------------------------------------------------------- предсказание
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Модель не обучена: вызовите fit()")
        return self._forward(self._prepare(X, fit=False))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток классов (целые числа 0..k-1)."""
        return self.decision_function(X).argmax(axis=1)

    def predict_names(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток классов в виде названий."""
        return np.array([self.class_names[i] for i in self.predict(X)])

    # --------------------------------------------------------------- отчёты
    def history_frame(self) -> pd.DataFrame:
        """История обучения в виде таблицы pandas.DataFrame."""
        return pd.DataFrame(self.history)

    def weights_frame(self, feature_names: list[str]) -> pd.DataFrame:
        """Итоговые веса модели с расшифровкой по признакам."""
        rows = list(feature_names) + ["bias"]
        return pd.DataFrame(self.W, index=rows, columns=self.class_names).round(4)
