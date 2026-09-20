"""Разведочный анализ датасета.

Реализует первый пункт задания: определить признаки объекта, баланс классов,
понять, как разделяются объекты, и найти наиболее информативные признаки.

Для оценки «разделяющей силы» признака используется корреляционное отношение
eta^2 (доля межгрупповой дисперсии в общей дисперсии признака):

    eta^2 = SS_between / SS_total,  0 <= eta^2 <= 1

Значение, близкое к 1, означает, что признак почти полностью объясняется
принадлежностью к классу, то есть хорошо разделяет классы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import Dataset


class DatasetAnalyzer:
    """Набор методов разведочного анализа для объекта :class:`Dataset`."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    # ---------------------------------------------------------- общие факты
    def overview_frame(self) -> pd.DataFrame:
        """Общая карточка датасета: размеры, признаки, пропуски."""
        ds = self.dataset
        rows = [
            ("Название", ds.title),
            ("Источник", ds.source),
            ("Число объектов", ds.n_samples),
            ("Число признаков", ds.n_features),
            ("Число классов", ds.n_classes),
            ("Признаки", ", ".join(ds.feature_names)),
            ("Классы (метки)", ", ".join(ds.target_names)),
            ("Пропусков в признаках", int(ds.X.isna().sum().sum())),
            ("Тип признаков", "вещественные (количественные)"),
        ]
        rows += [("Примечание", note) for note in ds.notes]
        return pd.DataFrame(rows, columns=["параметр", "значение"])

    def describe_frame(self) -> pd.DataFrame:
        """Описательные статистики признаков (аналог X.describe())."""
        return self.dataset.X.describe().T.round(3).reset_index(names="feature")

    def class_balance_frame(self) -> pd.DataFrame:
        """Баланс классов и вывод о сбалансированности выборки."""
        balance = self.dataset.class_balance()
        ratio = balance["count"].max() / max(balance["count"].min(), 1)
        balance["imbalance_ratio_max_min"] = round(float(ratio), 2)
        return balance

    def class_means_frame(self) -> pd.DataFrame:
        """Средние значения признаков по классам: видно, чем классы отличаются."""
        frame = self.dataset.X.copy()
        frame["class"] = self.dataset.y
        return frame.groupby("class").mean().round(3).reset_index()

    # ----------------------------------------------------- информативность
    def correlation_matrix(self) -> pd.DataFrame:
        """Матрица парных корреляций Пирсона между признаками."""
        return self.dataset.X.corr().round(3)

    @staticmethod
    def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
        """Корреляционное отношение eta^2 для одного признака."""
        overall_mean = values.mean()
        ss_total = float(((values - overall_mean) ** 2).sum())
        if ss_total == 0.0:
            return 0.0
        ss_between = 0.0
        for label in np.unique(labels):
            group = values[labels == label]
            ss_between += len(group) * (group.mean() - overall_mean) ** 2
        return float(ss_between / ss_total)

    def feature_importance_frame(self) -> pd.DataFrame:
        """Ранжирование признаков по разделяющей силе (eta^2 и |корреляция|)."""
        y_num = self.dataset.y_encoded()
        labels = self.dataset.y.to_numpy()
        rows = []
        for feature in self.dataset.feature_names:
            values = self.dataset.X[feature].to_numpy(dtype=float)
            rows.append(
                {
                    "feature": feature,
                    "eta_squared": round(self._eta_squared(values, labels), 4),
                    "abs_corr_with_label": round(abs(float(np.corrcoef(values, y_num)[0, 1])), 4),
                    "min": round(float(values.min()), 3),
                    "mean": round(float(values.mean()), 3),
                    "max": round(float(values.max()), 3),
                }
            )
        frame = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
        return frame.reset_index(drop=True)

    def top_features(self, k: int = 2) -> list[str]:
        """k наиболее информативных признаков (для ручных правил и графиков)."""
        return self.feature_importance_frame()["feature"].head(k).tolist()

    # -------------------------------------------------------------- вывод
    def print_report(self) -> None:
        """Печатает результаты анализа в консоль."""
        print(f"\n--- Анализ датасета: {self.dataset.title} ---")
        print(self.overview_frame().to_string(index=False))
        print("\nОписательные статистики:")
        print(self.describe_frame().to_string(index=False))
        print("\nБаланс классов:")
        print(self.class_balance_frame().to_string(index=False))
        print("\nСредние значения признаков по классам:")
        print(self.class_means_frame().to_string(index=False))
        print("\nИнформативность признаков (eta^2 -- доля межгрупповой дисперсии):")
        print(self.feature_importance_frame().to_string(index=False))
