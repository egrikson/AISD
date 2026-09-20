"""Метрики качества классификации.

Метрики реализованы вручную на NumPy, чтобы явно показать алгоритм их
вычисления (это часть учебного задания), и проверены на совпадение с
scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ClassificationMetrics:
    """Набор метрик для задачи многоклассовой классификации."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = list(class_names)
        self.n_classes = len(class_names)

    # ------------------------------------------------------------- базовые
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Матрица ошибок: cm[i][j] -- объектов класса i, предсказано j.

        Сложность: O(n), где n -- число объектов.
        """
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[int(true_label), int(pred_label)] += 1
        return cm

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Доля верных ответов: (TP + TN) / всего. Сложность O(n)."""
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    def precision_per_class(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Точность по каждому классу: TP / (TP + FP)."""
        cm = self.confusion_matrix(y_true, y_pred)
        predicted = cm.sum(axis=0)          # сколько раз класс был предсказан
        true_positive = np.diag(cm)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(predicted > 0, true_positive / np.maximum(predicted, 1), 0.0)
        return precision.astype(float)

    def recall_per_class(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Полнота по каждому классу: TP / (TP + FN)."""
        cm = self.confusion_matrix(y_true, y_pred)
        actual = cm.sum(axis=1)
        true_positive = np.diag(cm)
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.where(actual > 0, true_positive / np.maximum(actual, 1), 0.0)
        return recall.astype(float)

    def macro_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Усреднённая по классам точность (macro-average)."""
        return float(np.mean(self.precision_per_class(y_true, y_pred)))

    def macro_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(self.recall_per_class(y_true, y_pred)))

    def macro_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = self.macro_precision(y_true, y_pred)
        r = self.macro_recall(y_true, y_pred)
        return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))

    # ------------------------------------------------------------- отчёты
    def summary(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Сводка основных метрик одним словарём."""
        result = {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision_macro": self.macro_precision(y_true, y_pred),
            "recall_macro": self.macro_recall(y_true, y_pred),
            "f1_macro": self.macro_f1(y_true, y_pred),
        }
        for name, value in zip(self.class_names, self.precision_per_class(y_true, y_pred)):
            result[f"precision_{name}"] = float(value)
        return result

    def report(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Таблица precision/recall/support по классам."""
        cm = self.confusion_matrix(y_true, y_pred)
        return pd.DataFrame(
            {
                "class": self.class_names,
                "precision": self.precision_per_class(y_true, y_pred).round(4),
                "recall": self.recall_per_class(y_true, y_pred).round(4),
                "support": cm.sum(axis=1),
            }
        )

    def confusion_frame(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        cm = self.confusion_matrix(y_true, y_pred)
        return pd.DataFrame(
            cm,
            index=[f"истинный {c}" for c in self.class_names],
            columns=[f"предсказан {c}" for c in self.class_names],
        )
