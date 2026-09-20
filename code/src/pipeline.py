"""Сценарий лабораторной работы №1 целиком.

Класс :class:`Lab1Pipeline` последовательно выполняет все пункты задания для
одного датасета:

1. разведочный анализ (признаки, баланс классов, разделимость);
2. классификация по правилам, сформулированным вручную;
3. дерево решений с перебором максимальной глубины;
4. линейная и softmax-регрессия с перебором числа итераций обучения;
5. сохранение всех прогнозов и таблиц в CSV;
6. построение графиков;
7. замеры времени и памяти.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .analysis import DatasetAnalyzer
from .benchmark import Benchmark, BenchmarkTable
from .datasets import Dataset, DatasetLoader
from .linear_model import GradientDescentClassifier
from .metrics import ClassificationMetrics
from .plots import PlotBuilder
from .rules import RuleLibrary
from .tree_model import DecisionTreeExperiment

CSV_ENCODING = "utf-8-sig"   # BOM, чтобы Excel корректно открывал кириллицу


@dataclass
class LabConfig:
    """Параметры эксперимента (вынесены отдельно, чтобы легко менять)."""

    test_size: float = 0.3
    random_state: int = 42
    tree_depths: tuple[int | None, ...] = (3, 5, 7, 10, None)
    iteration_grid: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {"mse": 0.1, "logloss": 0.5}
    )
    model_titles: dict[str, str] = field(
        default_factory=lambda: {
            "mse": "Линейная регрессия (MSE)",
            "logloss": "Softmax-регрессия (log-loss)",
        }
    )


class Lab1Pipeline:
    """Полный прогон лабораторной работы для одного набора данных."""

    def __init__(
        self,
        dataset: Dataset,
        results_root: Path | str,
        figures_root: Path | str,
        config: LabConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config or LabConfig()
        self.results_dir = Path(results_root) / dataset.name
        self.figures_dir = Path(figures_root) / dataset.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = ClassificationMetrics(dataset.target_names)
        self.plots = PlotBuilder(self.figures_dir)
        self.bench = BenchmarkTable()
        self.summary_rows: list[dict[str, object]] = []
        self.confusion: dict[str, np.ndarray] = {}
        self.slugs: dict[str, str] = {}
        self.notes: list[str] = []

        # --- разбиение выборки (стратифицированное, фиксированный seed) ---
        y_encoded = dataset.y_encoded()
        indices = np.arange(dataset.n_samples)
        self.idx_train, self.idx_test = train_test_split(
            indices,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_encoded,
        )
        self.X = dataset.X
        self.y = dataset.y
        self.y_enc = y_encoded
        self.X_train = dataset.X.iloc[self.idx_train]
        self.X_test = dataset.X.iloc[self.idx_test]
        self.y_train = y_encoded[self.idx_train]
        self.y_test = y_encoded[self.idx_test]

    # ------------------------------------------------------------- служебное
    def _save_csv(self, frame: pd.DataFrame, filename: str, index: bool = False) -> Path:
        path = self.results_dir / filename
        frame.to_csv(path, index=index, encoding=CSV_ENCODING)
        return path

    def _save_text(self, text: str, filename: str) -> Path:
        path = self.results_dir / filename
        path.write_text(text, encoding="utf-8")
        return path

    def _predictions_frame(self, predicted: np.ndarray, subset: str = "весь датасет",
                           index: np.ndarray | None = None) -> pd.DataFrame:
        """Таблица «объект -> класс»: признаки, истинная и предсказанная метка."""
        idx = np.arange(self.dataset.n_samples) if index is None else index
        frame = self.dataset.X.iloc[idx].copy().reset_index(drop=True)
        frame.insert(0, "object_id", idx)
        frame["true_class"] = self.dataset.y.iloc[idx].to_numpy()
        frame["predicted_class"] = predicted
        frame["correct"] = frame["true_class"] == frame["predicted_class"]
        frame["subset"] = subset
        return frame

    def _register(self, model: str, slug: str, y_true: np.ndarray, y_pred: np.ndarray,
                  params: str, fit_ms: float, mem_kb: float) -> dict[str, float]:
        """Добавляет модель в итоговое сравнение и сохраняет её отчёт."""
        summary = self.metrics.summary(y_true, y_pred)
        row: dict[str, object] = {
            "model": model,
            "params": params,
            "test_accuracy": round(summary["accuracy"], 4),
            "test_precision_macro": round(summary["precision_macro"], 4),
            "test_recall_macro": round(summary["recall_macro"], 4),
            "test_f1_macro": round(summary["f1_macro"], 4),
            "fit_time_ms": round(fit_ms, 3),
            "peak_memory_kb": round(mem_kb, 2),
        }
        for name in self.dataset.target_names:
            row[f"precision_{name}"] = round(summary[f"precision_{name}"], 4)
        self.summary_rows.append(row)
        self.confusion[model] = self.metrics.confusion_matrix(y_true, y_pred)
        self.slugs[model] = slug
        return summary

    # ------------------------------------------------------- 1. анализ данных
    def step_analysis(self) -> DatasetAnalyzer:
        analyzer = DatasetAnalyzer(self.dataset)
        with Benchmark() as bench:
            overview = analyzer.overview_frame()
            describe = analyzer.describe_frame()
            balance = analyzer.class_balance_frame()
            means = analyzer.class_means_frame()
            corr = analyzer.correlation_matrix()
            importance = analyzer.feature_importance_frame()
        self.bench.add("Анализ датасета", bench, f"{self.dataset.n_samples} объектов")

        self._save_csv(overview, "01_dataset_overview.csv")
        self._save_csv(describe, "02_describe.csv")
        self._save_csv(balance, "03_class_balance.csv")
        self._save_csv(means, "04_class_means.csv")
        self._save_csv(corr, "05_correlation.csv", index=True)
        self._save_csv(importance, "06_feature_importance.csv")

        title = self.dataset.title
        self.plots.class_balance(balance, title)
        self.plots.pairplot(
            self.X, self.y, self.dataset.target_names,
            analyzer.feature_importance_frame()["feature"].tolist(), title,
        )
        self.plots.correlation_heatmap(corr, title)
        self.plots.feature_importance(importance, title)
        top2 = analyzer.top_features(2)
        self.plots.decision_regions(self.X, self.y, self.dataset.target_names, tuple(top2), title)

        analyzer.print_report()
        self.notes.append(
            "Наиболее информативные признаки: "
            + ", ".join(f"{r.feature} (eta^2={r.eta_squared})" for r in importance.head(3).itertuples())
        )
        return analyzer

    # ------------------------------------------- 2. правила, заданные вручную
    def step_manual_rules(self) -> None:
        classifier = RuleLibrary.build(self.dataset.name)
        with Benchmark() as bench:
            predicted_all = classifier.predict(self.X)
        self.bench.add("Прогноз по ручным правилам", bench, f"{len(classifier.rules) + 1} правил")

        self._save_text(classifier.as_text(), "12_manual_rules.txt")
        self._save_csv(self._predictions_frame(predicted_all), "10_predictions_manual_rules.csv")
        self._save_csv(classifier.coverage_frame(self.X, self.y), "11_manual_rules_coverage.csv")

        name_to_idx = {name: i for i, name in enumerate(self.dataset.target_names)}
        pred_enc = np.array([name_to_idx[p] for p in predicted_all])
        full_accuracy = self.metrics.accuracy(self.y_enc, pred_enc)
        summary = self._register(
            "Ручные правила",
            "manual_rules",
            self.y_test,
            pred_enc[self.idx_test],
            params=f"{len(classifier.rules) + 1} правил",
            fit_ms=0.0,
            mem_kb=bench.peak_kb,
        )
        self._save_csv(
            self.metrics.report(self.y_test, pred_enc[self.idx_test]),
            "41_report_manual_rules.csv",
        )
        print("\n--- Ручные правила ---")
        print(classifier.as_text())
        print(f"Accuracy на всём датасете: {full_accuracy:.4f}")
        print(f"Accuracy на тестовой выборке: {summary['accuracy']:.4f}")
        self.notes.append(
            f"Ручные правила дают accuracy {full_accuracy:.4f} на всём датасете "
            f"и {summary['accuracy']:.4f} на тестовой выборке."
        )

    # ------------------------------------------------- 3. дерево решений
    def step_decision_tree(self) -> None:
        experiment = DecisionTreeExperiment(
            feature_names=self.dataset.feature_names,
            class_names=self.dataset.target_names,
            depths=self.config.tree_depths,
            random_state=self.config.random_state,
        )
        results = experiment.run(
            self.X_train.to_numpy(), self.y_train, self.X_test.to_numpy(), self.y_test
        )
        comparison = experiment.comparison_frame()
        self._save_csv(comparison, "20_tree_depth_comparison.csv")

        for result in results:
            label = result.depth_slug
            predicted = result.model.predict(self.X.to_numpy())
            names = np.array([self.dataset.target_names[i] for i in predicted])
            subset = np.where(np.isin(np.arange(self.dataset.n_samples), self.idx_test),
                              "test", "train")
            frame = self._predictions_frame(names)
            frame["subset"] = subset
            self._save_csv(frame, f"21_predictions_tree_depth_{label}.csv")
            self._save_text(result.rules_text, f"22_tree_rules_depth_{label}.txt")
            self.bench.add_manual(
                f"Обучение дерева (глубина {result.depth_label})",
                result.fit_seconds * 1000,
                result.peak_memory_kb,
                f"листьев: {result.model.get_n_leaves()}",
            )

        best = experiment.best_result()
        self._save_csv(experiment.feature_importance_frame(best), "23_tree_feature_importance.csv")
        self._register(
            f"Дерево решений (глубина {best.depth_label})",
            f"tree_depth_{best.depth_slug}",
            self.y_test,
            best.model.predict(self.X_test.to_numpy()),
            params=f"max_depth={best.depth_label}, листьев={best.model.get_n_leaves()}",
            fit_ms=best.fit_seconds * 1000,
            mem_kb=best.peak_memory_kb,
        )
        self._save_csv(
            self.metrics.report(self.y_test, best.model.predict(self.X_test.to_numpy())),
            "41_report_tree.csv",
        )
        self.plots.tree_depth_comparison(comparison, self.dataset.title)

        print("\n--- Дерево решений ---")
        print(comparison.to_string(index=False))
        print(f"\nПравила лучшего дерева (глубина {best.depth_label}):")
        print(best.rules_text)
        self.notes.append(
            f"Лучшее дерево: глубина {best.depth_label}, "
            f"test accuracy {best.metrics_test['accuracy']:.4f}, листьев {best.model.get_n_leaves()}."
        )

    # ------------------------------------ 4. линейная и softmax регрессия
    def step_linear_models(self) -> None:
        histories: dict[str, pd.DataFrame] = {}
        sweep_rows: list[dict[str, object]] = []
        max_iterations = max(self.config.iteration_grid)

        for loss, title in self.config.model_titles.items():
            learning_rate = self.config.learning_rates[loss]

            # --- перебор количества итераций обучения (10, 20, ..., 100) ---
            for n_iter in self.config.iteration_grid:
                model = GradientDescentClassifier(
                    self.dataset.target_names, loss=loss, n_iterations=n_iter,
                    learning_rate=learning_rate, random_state=self.config.random_state,
                )
                with Benchmark() as bench:
                    model.fit(self.X_train.to_numpy(), self.y_train)
                y_pred_test = model.predict(self.X_test.to_numpy())
                y_pred_train = model.predict(self.X_train.to_numpy())
                last = model.history[-1]
                sweep_rows.append(
                    {
                        "model": title,
                        "loss": loss,
                        "learning_rate": learning_rate,
                        "n_iterations": n_iter,
                        "train_loss": round(last["train_loss"], 6),
                        "train_accuracy": round(self.metrics.accuracy(self.y_train, y_pred_train), 4),
                        "test_accuracy": round(self.metrics.accuracy(self.y_test, y_pred_test), 4),
                        "test_precision_macro": round(
                            self.metrics.macro_precision(self.y_test, y_pred_test), 4
                        ),
                        "weight_norm_l2": round(last["weight_norm_l2"], 4),
                        "weight_update_norm_l2": round(last["weight_update_norm_l2"], 8),
                        "fit_time_ms": round(bench.seconds * 1000, 3),
                        "peak_memory_kb": round(bench.peak_kb, 2),
                    }
                )

            # --- полный прогон на максимум итераций: история и прогнозы ---
            model = GradientDescentClassifier(
                self.dataset.target_names, loss=loss, n_iterations=max_iterations,
                learning_rate=learning_rate, random_state=self.config.random_state,
            )
            with Benchmark() as bench:
                model.fit(self.X_train.to_numpy(), self.y_train,
                          self.X_test.to_numpy(), self.y_test)
            history = model.history_frame()
            histories[title] = history
            self._save_csv(history.round(6), f"31_training_history_{loss}.csv")
            self._save_csv(
                model.weights_frame(self.dataset.feature_names), f"33_weights_{loss}.csv", index=True
            )

            predicted_all = model.predict_names(self.X.to_numpy())
            frame = self._predictions_frame(predicted_all)
            frame["subset"] = np.where(
                np.isin(np.arange(self.dataset.n_samples), self.idx_test), "test", "train"
            )
            self._save_csv(frame, f"32_predictions_linear_{loss}.csv")

            y_pred_test = model.predict(self.X_test.to_numpy())
            self._register(
                title, f"linear_{loss}", self.y_test, y_pred_test,
                params=f"lr={learning_rate}, iterations={max_iterations}",
                fit_ms=bench.seconds * 1000, mem_kb=bench.peak_kb,
            )
            self._save_csv(self.metrics.report(self.y_test, y_pred_test), f"41_report_{loss}.csv")
            self.bench.add(
                f"Обучение: {title}", bench, f"{max_iterations} итераций, lr={learning_rate}"
            )

        sweep = pd.DataFrame(sweep_rows)
        self._save_csv(sweep, "30_linear_iterations_comparison.csv")

        title = self.dataset.title
        self.plots.loss_curves(histories, title)
        self.plots.accuracy_precision_curves(histories, title)
        self.plots.weight_norm(histories, title)
        self.plots.weight_update_norm(histories, title)
        self.plots.iterations_sweep(sweep, title)

        print("\n--- Линейные модели: влияние числа итераций ---")
        print(sweep.to_string(index=False))
        for name, history in histories.items():
            first, last = history.iloc[0], history.iloc[-1]
            self.notes.append(
                f"{name}: loss {first['train_loss']:.4f} -> {last['train_loss']:.4f}, "
                f"||W|| {first['weight_norm_l2']:.3f} -> {last['weight_norm_l2']:.3f}, "
                f"||dW|| {first['weight_update_norm_l2']:.5f} -> {last['weight_update_norm_l2']:.5f}."
            )

    # ------------------------------------------------- 5. итоги и сохранение
    def step_summary(self) -> pd.DataFrame:
        comparison = pd.DataFrame(self.summary_rows)
        self._save_csv(comparison, "40_model_comparison.csv")
        for model, matrix in self.confusion.items():
            safe = self.slugs[model]
            frame = pd.DataFrame(
                matrix,
                index=[f"истинный {c}" for c in self.dataset.target_names],
                columns=[f"предсказан {c}" for c in self.dataset.target_names],
            )
            self._save_csv(frame, f"42_confusion_{safe}.csv", index=True)
        self._save_csv(self.bench.to_frame(), "50_benchmark.csv")

        self.plots.model_comparison(comparison, self.dataset.title)
        self.plots.confusion_matrices(self.confusion, self.dataset.target_names, self.dataset.title)

        best = comparison.sort_values("test_accuracy", ascending=False).iloc[0]
        self.notes.append(
            f"Лучшая модель по accuracy на тесте: {best['model']} ({best['test_accuracy']})."
        )
        summary_text = "\n".join(
            [
                f"Датасет: {self.dataset.title}",
                f"Объектов: {self.dataset.n_samples}, признаков: {self.dataset.n_features}, "
                f"классов: {self.dataset.n_classes}",
                f"Разбиение: train {len(self.idx_train)} / test {len(self.idx_test)} "
                f"(test_size={self.config.test_size}, random_state={self.config.random_state})",
                "",
                "Выводы:",
                *[f"  - {note}" for note in self.notes],
                "",
                "Сравнение моделей:",
                comparison.to_string(index=False),
                "",
                "Время и память:",
                self.bench.to_frame().to_string(index=False),
            ]
        )
        self._save_text(summary_text, "90_summary.txt")

        print("\n--- Итоговое сравнение моделей ---")
        print(comparison.to_string(index=False))
        print("\n--- Время и память ---")
        print(self.bench.to_frame().to_string(index=False))
        return comparison

    # -------------------------------------------------------------- запуск
    def run(self) -> pd.DataFrame:
        """Выполняет все этапы лабораторной работы по порядку."""
        print("=" * 78)
        print(f"ДАТАСЕТ: {self.dataset.title}")
        print("=" * 78)
        self.step_analysis()
        self.step_manual_rules()
        self.step_decision_tree()
        self.step_linear_models()
        comparison = self.step_summary()
        print(f"\nCSV-файлы: {self.results_dir}")
        print(f"Графики:   {self.figures_dir}")
        return comparison


def run_dataset(
    name: str,
    results_root: Path | str,
    figures_root: Path | str,
    config: LabConfig | None = None,
) -> pd.DataFrame:
    """Загружает датасет по имени и полностью выполняет лабораторную работу."""
    dataset = DatasetLoader().load(name)
    return Lab1Pipeline(dataset, results_root, figures_root, config).run()
