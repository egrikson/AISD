"""Построение графиков для отчёта.

Все графики сохраняются в PNG в каталог, переданный в конструктор
:class:`PlotBuilder`. Используется backend ``Agg``, поэтому окна не
открываются и скрипт одинаково работает в консоли и в CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.autolayout": True,
    }
)

PALETTE = ["#2f6fdb", "#e07b39", "#3f9b6d", "#b4508f", "#8a8f98"]


class PlotBuilder:
    """Фабрика графиков: каждый метод сохраняет картинку и возвращает путь."""

    def __init__(self, output_dir: Path | str, prefix: str = "") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.saved: list[Path] = []

    # ------------------------------------------------------------- служебное
    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / f"{self.prefix}{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.saved.append(path)
        return path

    @staticmethod
    def _color(index: int) -> str:
        return PALETTE[index % len(PALETTE)]

    # -------------------------------------------------------- анализ данных
    def class_balance(self, balance: pd.DataFrame, title: str) -> Path:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        colors = [self._color(i) for i in range(len(balance))]
        ax.bar(balance["class"], balance["count"], color=colors)
        for x, (count, share) in enumerate(zip(balance["count"], balance["share_%"])):
            ax.text(x, count, f"{count} ({share}%)", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Баланс классов: {title}")
        ax.set_ylabel("количество объектов")
        ax.set_ylim(0, balance["count"].max() * 1.18)
        return self._save(fig, "01_class_balance")

    def pairplot(
        self, X: pd.DataFrame, y: pd.Series, class_names: list[str], features: list[str], title: str
    ) -> Path:
        """Попарные диаграммы рассеяния для выбранных признаков."""
        features = features[:4]
        n = len(features)
        fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n), squeeze=False)
        for i, fy in enumerate(features):
            for j, fx in enumerate(features):
                ax = axes[i][j]
                for c_idx, cls in enumerate(class_names):
                    mask = (y == cls).to_numpy()
                    if i == j:
                        ax.hist(X.loc[mask, fx], bins=15, alpha=0.6, color=self._color(c_idx))
                    else:
                        ax.scatter(
                            X.loc[mask, fx],
                            X.loc[mask, fy],
                            s=10,
                            alpha=0.75,
                            color=self._color(c_idx),
                            label=cls if (i == 0 and j == 1) else None,
                        )
                if i == n - 1:
                    ax.set_xlabel(fx, fontsize=8)
                if j == 0:
                    ax.set_ylabel(fy, fontsize=8)
                ax.tick_params(labelsize=7)
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=self._color(i), label=cls)
            for i, cls in enumerate(class_names)
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=len(class_names),
            frameon=False, bbox_to_anchor=(0.5, -0.03),
        )
        fig.suptitle(f"Попарные графики признаков: {title}", y=1.02)
        return self._save(fig, "02_pairplot")

    def correlation_heatmap(self, corr: pd.DataFrame, title: str) -> Path:
        fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(corr), 0.9 + 0.5 * len(corr)))
        image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
        if len(corr) <= 8:
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(
                        j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7
                    )
        ax.grid(False)
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.set_title(f"Корреляция признаков: {title}", fontsize=10)
        return self._save(fig, "03_correlation")

    def feature_importance(self, importance: pd.DataFrame, title: str) -> Path:
        fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(importance) + 1.4))
        order = importance.sort_values("eta_squared")
        ax.barh(order["feature"], order["eta_squared"], color=self._color(0))
        ax.set_xlabel("eta^2 (доля межгрупповой дисперсии)")
        ax.set_title(f"Разделяющая сила признаков: {title}")
        ax.set_xlim(0, 1)
        return self._save(fig, "04_feature_importance")

    # ------------------------------------------------------ кривые обучения
    def loss_curves(self, histories: dict[str, pd.DataFrame], title: str) -> Path:
        """График потерь по итерациям обучения для всех линейных моделей."""
        fig, axes = plt.subplots(1, len(histories), figsize=(5.2 * len(histories), 3.8), squeeze=False)
        for idx, (name, history) in enumerate(histories.items()):
            ax = axes[0][idx]
            ax.plot(history["iteration"], history["train_loss"], color=self._color(0), label="train")
            if "test_loss" in history:
                ax.plot(
                    history["iteration"],
                    history["test_loss"],
                    color=self._color(1),
                    linestyle="--",
                    label="test",
                )
            ax.set_xlabel("итерация")
            ax.set_ylabel("значение функции потерь")
            ax.set_title(name)
            ax.legend()
        fig.suptitle(f"Потери при обучении: {title}", y=1.03)
        return self._save(fig, "05_loss")

    def accuracy_precision_curves(self, histories: dict[str, pd.DataFrame], title: str) -> Path:
        """Accuracy и Precision в зависимости от номера итерации."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for idx, (name, history) in enumerate(histories.items()):
            axes[0].plot(
                history["iteration"], history["train_accuracy"], color=self._color(idx),
                label=f"{name} (train)"
            )
            if "test_accuracy" in history:
                axes[0].plot(
                    history["iteration"], history["test_accuracy"], color=self._color(idx),
                    linestyle="--", label=f"{name} (test)"
                )
            axes[1].plot(
                history["iteration"], history["train_precision_macro"], color=self._color(idx),
                label=f"{name} (train)"
            )
            if "test_precision_macro" in history:
                axes[1].plot(
                    history["iteration"], history["test_precision_macro"], color=self._color(idx),
                    linestyle="--", label=f"{name} (test)"
                )
        axes[0].set_title("Accuracy")
        axes[1].set_title("Precision (macro)")
        for ax in axes:
            ax.set_xlabel("итерация")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
        fig.suptitle(f"Метрики по итерациям обучения: {title}", y=1.03)
        return self._save(fig, "06_accuracy_precision")

    def weight_norm(self, histories: dict[str, pd.DataFrame], title: str) -> Path:
        """L2-норма вектора весов ||W|| по итерациям."""
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for idx, (name, history) in enumerate(histories.items()):
            ax.plot(history["iteration"], history["weight_norm_l2"], color=self._color(idx), label=name)
        ax.set_xlabel("итерация")
        ax.set_ylabel("||W||")
        ax.set_title(f"L2-норма вектора весов: {title}")
        ax.legend()
        return self._save(fig, "07_weight_norm")

    def weight_update_norm(self, histories: dict[str, pd.DataFrame], title: str) -> Path:
        """L2-норма изменения весов ||dW|| по итерациям (лог. шкала)."""
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for idx, (name, history) in enumerate(histories.items()):
            ax.plot(
                history["iteration"], history["weight_update_norm_l2"],
                color=self._color(idx), label=name
            )
        ax.set_yscale("log")
        ax.set_xlabel("итерация")
        ax.set_ylabel("||dW|| (логарифмическая шкала)")
        ax.set_title(f"L2-норма обновления весов: {title}")
        ax.legend()
        return self._save(fig, "08_weight_update_norm")

    # ------------------------------------------------------------ сравнения
    def iterations_sweep(self, sweep: pd.DataFrame, title: str) -> Path:
        """Качество модели в зависимости от заданного числа итераций."""
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for idx, (name, group) in enumerate(sweep.groupby("model", sort=False)):
            ax.plot(
                group["n_iterations"], group["test_accuracy"], marker="o",
                color=self._color(idx), label=f"{name} accuracy"
            )
            ax.plot(
                group["n_iterations"], group["test_precision_macro"], marker="s", linestyle="--",
                color=self._color(idx), label=f"{name} precision"
            )
        ax.set_xlabel("количество итераций обучения")
        ax.set_ylabel("метрика на тестовой выборке")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Влияние числа итераций: {title}")
        ax.legend(fontsize=8)
        return self._save(fig, "09_iterations_sweep")

    def tree_depth_comparison(self, comparison: pd.DataFrame, title: str) -> Path:
        """Качество дерева решений в зависимости от максимальной глубины."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        labels = comparison["max_depth"].astype(str)
        x = np.arange(len(labels))
        axes[0].plot(x, comparison["train_accuracy"], marker="o", color=self._color(0), label="train")
        axes[0].plot(x, comparison["test_accuracy"], marker="s", color=self._color(1), label="test")
        axes[0].set_ylabel("accuracy")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Точность дерева решений")
        axes[0].legend()
        axes[1].bar(x, comparison["leaves"], color=self._color(2))
        axes[1].set_ylabel("число листьев")
        axes[1].set_title("Сложность дерева")
        for ax in axes:
            ax.set_xticks(x, labels)
            ax.set_xlabel("максимальная глубина")
        fig.suptitle(f"Дерево решений: {title}", y=1.03)
        return self._save(fig, "10_tree_depth")

    def confusion_matrices(self, matrices: dict[str, np.ndarray], class_names: list[str], title: str) -> Path:
        """Матрицы ошибок всех моделей на тестовой выборке."""
        n = len(matrices)
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
        for idx, (name, cm) in enumerate(matrices.items()):
            ax = axes[0][idx]
            ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9
                    )
            ax.set_xticks(range(len(class_names)), class_names, rotation=45, fontsize=8)
            ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
            ax.set_xlabel("предсказано")
            if idx == 0:
                ax.set_ylabel("истина")
            ax.set_title(name, fontsize=9)
            ax.grid(False)
        fig.suptitle(f"Матрицы ошибок (тестовая выборка): {title}", y=1.04)
        return self._save(fig, "11_confusion_matrices")

    def model_comparison(self, comparison: pd.DataFrame, title: str) -> Path:
        """Итоговое сравнение всех моделей по accuracy и precision."""
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        x = np.arange(len(comparison))
        width = 0.38
        ax.bar(x - width / 2, comparison["test_accuracy"], width, color=self._color(0), label="accuracy")
        ax.bar(
            x + width / 2, comparison["test_precision_macro"], width,
            color=self._color(1), label="precision (macro)"
        )
        for xi, (acc, prec) in enumerate(
            zip(comparison["test_accuracy"], comparison["test_precision_macro"])
        ):
            ax.text(xi - width / 2, acc, f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + width / 2, prec, f"{prec:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, comparison["model"], rotation=15, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Сравнение моделей: {title}")
        ax.legend()
        return self._save(fig, "12_model_comparison")

    def decision_regions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        class_names: list[str],
        features: tuple[str, str],
        title: str,
    ) -> Path:
        """Диаграмма рассеяния по двум самым информативным признакам."""
        fx, fy = features
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for idx, cls in enumerate(class_names):
            mask = (y == cls).to_numpy()
            ax.scatter(X.loc[mask, fx], X.loc[mask, fy], s=22, alpha=0.8,
                       color=self._color(idx), label=cls)
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(f"Разделимость классов по двум признакам: {title}")
        ax.legend()
        return self._save(fig, "13_top2_scatter")
