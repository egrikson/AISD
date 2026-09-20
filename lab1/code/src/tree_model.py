"""Эксперимент с деревом решений.

Обёртка над ``sklearn.tree.DecisionTreeClassifier``, которая перебирает
значения максимальной глубины дерева, измеряет качество на обучающей и
тестовой выборках и извлекает текстовое представление построенных правил.

Сложность обучения дерева решений: O(n * m * log n), где n -- число объектов,
m -- число признаков; сложность классификации одного объекта -- O(h), где
h -- глубина дерева.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from .benchmark import Benchmark
from .metrics import ClassificationMetrics


@dataclass
class TreeResult:
    """Результат обучения одного дерева."""

    max_depth: int | None
    model: DecisionTreeClassifier
    metrics_train: dict[str, float]
    metrics_test: dict[str, float]
    rules_text: str
    fit_seconds: float
    peak_memory_kb: float

    @property
    def depth_label(self) -> str:
        """Подпись глубины для таблиц и графиков."""
        return "без ограничения" if self.max_depth is None else str(self.max_depth)

    @property
    def depth_slug(self) -> str:
        """Безопасное для имени файла обозначение глубины."""
        return "none" if self.max_depth is None else str(self.max_depth)


class DecisionTreeExperiment:
    """Перебор глубин дерева решений и сравнение результатов."""

    def __init__(
        self,
        feature_names: list[str],
        class_names: list[str],
        depths: tuple[int | None, ...] = (3, 5, 7, 10, None),
        criterion: str = "gini",
        random_state: int = 42,
    ) -> None:
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.depths = depths
        self.criterion = criterion
        self.random_state = random_state
        self.metrics = ClassificationMetrics(self.class_names)
        self.results: list[TreeResult] = []

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> list[TreeResult]:
        """Обучает по дереву на каждую глубину из self.depths."""
        self.results = []
        for depth in self.depths:
            model = DecisionTreeClassifier(
                max_depth=depth,
                criterion=self.criterion,
                random_state=self.random_state,
            )
            with Benchmark() as bench:
                model.fit(X_train, y_train)
            result = TreeResult(
                max_depth=depth,
                model=model,
                metrics_train=self.metrics.summary(y_train, model.predict(X_train)),
                metrics_test=self.metrics.summary(y_test, model.predict(X_test)),
                rules_text=export_text(
                    model, feature_names=self.feature_names, class_names=self.class_names
                ),
                fit_seconds=bench.seconds,
                peak_memory_kb=bench.peak_kb,
            )
            self.results.append(result)
        return self.results

    def comparison_frame(self) -> pd.DataFrame:
        """Сводная таблица «глубина -> качество и размер дерева»."""
        rows = []
        for result in self.results:
            rows.append(
                {
                    "max_depth": result.depth_label,
                    "train_accuracy": round(result.metrics_train["accuracy"], 4),
                    "test_accuracy": round(result.metrics_test["accuracy"], 4),
                    "train_precision_macro": round(result.metrics_train["precision_macro"], 4),
                    "test_precision_macro": round(result.metrics_test["precision_macro"], 4),
                    "real_depth": int(result.model.get_depth()),
                    "leaves": int(result.model.get_n_leaves()),
                    "nodes": int(result.model.tree_.node_count),
                    "fit_seconds": round(result.fit_seconds, 6),
                    "peak_memory_kb": round(result.peak_memory_kb, 2),
                }
            )
        return pd.DataFrame(rows)

    def best_result(self) -> TreeResult:
        """Лучшее дерево по точности на тесте; при равенстве -- самое простое."""
        return min(
            self.results,
            key=lambda r: (-r.metrics_test["accuracy"], r.model.get_n_leaves()),
        )

    def feature_importance_frame(self, result: TreeResult) -> pd.DataFrame:
        """Важность признаков по выбранному дереву."""
        frame = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": result.model.feature_importances_.round(4),
            }
        )
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)
