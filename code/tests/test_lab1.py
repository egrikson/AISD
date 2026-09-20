"""Тесты работоспособности лабораторной работы на разных входных данных.

Запуск из каталога code/:
    python -m unittest discover -s tests -v

Проверяется:
  * корректность метрик (сверка с scikit-learn);
  * поведение классификатора на правилах;
  * сходимость градиентного спуска и устойчивость к вырожденным данным;
  * загрузка всех трёх наборов данных;
  * полный прогон конвейера на синтетическом наборе.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.analysis import DatasetAnalyzer                      # noqa: E402
from src.datasets import Dataset, DatasetLoader               # noqa: E402
from src.linear_model import GradientDescentClassifier, StandardScaler  # noqa: E402
from src.metrics import ClassificationMetrics                 # noqa: E402
from src.pipeline import Lab1Pipeline, LabConfig              # noqa: E402
from src.rules import Condition, Rule, RuleBasedClassifier, RuleLibrary  # noqa: E402
from src.tree_model import DecisionTreeExperiment             # noqa: E402


def make_synthetic_dataset(n_per_class: int = 40, seed: int = 0) -> Dataset:
    """Три хорошо разделимых гауссовых облака в двумерном пространстве."""
    rng = np.random.default_rng(seed)
    centers = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    names = ["A", "B", "C"]
    rows, labels = [], []
    for center, name in zip(centers, names):
        points = rng.normal(loc=center, scale=0.6, size=(n_per_class, 2))
        rows.append(points)
        labels += [name] * n_per_class
    X = pd.DataFrame(np.vstack(rows), columns=["f1", "f2"])
    return Dataset(
        name="synthetic",
        title="Синтетический набор",
        X=X,
        y=pd.Series(labels),
        feature_names=["f1", "f2"],
        target_names=names,
        source="сгенерирован в тесте",
    )


class TestMetrics(unittest.TestCase):
    """Ручные метрики должны совпадать с реализацией scikit-learn."""

    def setUp(self) -> None:
        self.metrics = ClassificationMetrics(["a", "b", "c"])
        rng = np.random.default_rng(7)
        self.y_true = rng.integers(0, 3, 300)
        self.y_pred = rng.integers(0, 3, 300)

    def test_accuracy_matches_sklearn(self) -> None:
        from sklearn.metrics import accuracy_score

        self.assertAlmostEqual(
            self.metrics.accuracy(self.y_true, self.y_pred),
            accuracy_score(self.y_true, self.y_pred),
        )

    def test_precision_matches_sklearn(self) -> None:
        from sklearn.metrics import precision_score

        self.assertAlmostEqual(
            self.metrics.macro_precision(self.y_true, self.y_pred),
            precision_score(self.y_true, self.y_pred, average="macro", zero_division=0),
        )

    def test_confusion_matrix_matches_sklearn(self) -> None:
        from sklearn.metrics import confusion_matrix

        np.testing.assert_array_equal(
            self.metrics.confusion_matrix(self.y_true, self.y_pred),
            confusion_matrix(self.y_true, self.y_pred, labels=[0, 1, 2]),
        )

    def test_perfect_prediction(self) -> None:
        y = np.array([0, 1, 2, 2, 1, 0])
        self.assertEqual(self.metrics.accuracy(y, y), 1.0)
        self.assertEqual(self.metrics.macro_precision(y, y), 1.0)

    def test_empty_input(self) -> None:
        empty = np.array([], dtype=int)
        self.assertEqual(self.metrics.accuracy(empty, empty), 0.0)

    def test_missing_class_does_not_crash(self) -> None:
        """Если класс ни разу не предсказан, precision для него равен 0."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 1])
        precision = self.metrics.precision_per_class(y_true, y_pred)
        self.assertEqual(precision[2], 0.0)
        self.assertEqual(precision[0], 1.0)


class TestRules(unittest.TestCase):
    """Классификатор на правилах: порядок правил и правило по умолчанию."""

    def test_condition_operators(self) -> None:
        column = pd.Series([1.0, 5.0, 10.0])
        np.testing.assert_array_equal(
            Condition("f", "<", 5.0).evaluate(column), [True, False, False]
        )
        np.testing.assert_array_equal(
            Condition("f", ">=", 5.0).evaluate(column), [False, True, True]
        )

    def test_invalid_operator_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Condition("f", "==", 1.0)

    def test_first_matching_rule_wins(self) -> None:
        X = pd.DataFrame({"f": [1.0, 6.0, 20.0]})
        clf = RuleBasedClassifier(
            [
                Rule([Condition("f", "<", 5.0)], "low"),
                Rule([Condition("f", "<", 10.0)], "mid"),
            ],
            default_class="high",
        )
        np.testing.assert_array_equal(clf.predict(X), ["low", "mid", "high"])

    def test_conjunction_of_conditions(self) -> None:
        X = pd.DataFrame({"a": [1.0, 1.0, 9.0], "b": [1.0, 9.0, 1.0]})
        clf = RuleBasedClassifier(
            [Rule([Condition("a", "<", 5.0), Condition("b", "<", 5.0)], "both")],
            default_class="other",
        )
        np.testing.assert_array_equal(clf.predict(X), ["both", "other", "other"])

    def test_library_rules_are_accurate(self) -> None:
        """Ручные правила должны давать заметно лучше случайного угадывания."""
        loader = DatasetLoader()
        expectations = {"iris": 0.90, "wine": 0.85, "penguins": 0.85}
        for name, threshold in expectations.items():
            with self.subTest(dataset=name):
                dataset = loader.load(name)
                predicted = RuleLibrary.build(name).predict(dataset.X)
                accuracy = float((predicted == dataset.y.to_numpy()).mean())
                self.assertGreaterEqual(accuracy, threshold)

    def test_coverage_frame_totals(self) -> None:
        dataset = DatasetLoader().load("iris")
        clf = RuleLibrary.build("iris")
        coverage = clf.coverage_frame(dataset.X, dataset.y)
        self.assertEqual(int(coverage["covered"].sum()), dataset.n_samples)


class TestStandardScaler(unittest.TestCase):
    def test_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(loc=5.0, scale=3.0, size=(200, 3))
        Z = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-10)

    def test_constant_column_does_not_divide_by_zero(self) -> None:
        X = np.column_stack([np.ones(10), np.arange(10.0)])
        Z = StandardScaler().fit_transform(X)
        self.assertTrue(np.all(np.isfinite(Z)))

    def test_transform_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            StandardScaler().transform(np.zeros((2, 2)))


class TestGradientDescentClassifier(unittest.TestCase):
    """Сходимость и корректность линейных моделей."""

    def setUp(self) -> None:
        self.dataset = make_synthetic_dataset()
        self.X = self.dataset.X.to_numpy()
        self.y = self.dataset.y_encoded()

    def test_invalid_loss_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GradientDescentClassifier(["a", "b"], loss="hinge")

    def test_predict_before_fit_raises(self) -> None:
        model = GradientDescentClassifier(self.dataset.target_names)
        with self.assertRaises(RuntimeError):
            model.predict(self.X)

    def test_loss_decreases(self) -> None:
        for loss, lr in (("mse", 0.1), ("logloss", 0.5)):
            with self.subTest(loss=loss):
                model = GradientDescentClassifier(
                    self.dataset.target_names, loss=loss, n_iterations=100, learning_rate=lr
                ).fit(self.X, self.y)
                history = model.history_frame()
                self.assertLess(history["train_loss"].iloc[-1], history["train_loss"].iloc[0])

    def test_separable_data_is_learned(self) -> None:
        model = GradientDescentClassifier(
            self.dataset.target_names, loss="logloss", n_iterations=200, learning_rate=0.5
        ).fit(self.X, self.y)
        accuracy = float((model.predict(self.X) == self.y).mean())
        self.assertGreaterEqual(accuracy, 0.95)

    def test_history_has_all_required_columns(self) -> None:
        model = GradientDescentClassifier(
            self.dataset.target_names, n_iterations=25
        ).fit(self.X, self.y, self.X, self.y)
        history = model.history_frame()
        for column in (
            "iteration", "train_loss", "test_loss", "train_accuracy", "test_accuracy",
            "train_precision_macro", "weight_norm_l2", "weight_update_norm_l2",
        ):
            self.assertIn(column, history.columns)
        self.assertEqual(len(history), 25)

    def test_update_norm_shrinks_when_converging(self) -> None:
        """||dW|| должна убывать -- это индикатор сходимости."""
        model = GradientDescentClassifier(
            self.dataset.target_names, loss="logloss", n_iterations=150, learning_rate=0.5
        ).fit(self.X, self.y)
        norms = model.history_frame()["weight_update_norm_l2"].to_numpy()
        self.assertLess(norms[-1], norms[0])

    def test_reproducible_with_same_seed(self) -> None:
        kwargs = dict(loss="logloss", n_iterations=30, learning_rate=0.5, random_state=123)
        first = GradientDescentClassifier(self.dataset.target_names, **kwargs).fit(self.X, self.y)
        second = GradientDescentClassifier(self.dataset.target_names, **kwargs).fit(self.X, self.y)
        np.testing.assert_allclose(first.W, second.W)

    def test_single_class_input(self) -> None:
        """Вырожденный случай: в обучающей выборке только один класс."""
        X = np.random.default_rng(3).normal(size=(20, 2))
        y = np.zeros(20, dtype=int)
        model = GradientDescentClassifier(["only"], n_iterations=10).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_tiny_dataset(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        model = GradientDescentClassifier(["a", "b"], n_iterations=50, learning_rate=0.5)
        model.fit(X, y)
        self.assertEqual(len(model.predict(X)), 2)


class TestDecisionTreeExperiment(unittest.TestCase):
    def test_depth_sweep_produces_row_per_depth(self) -> None:
        dataset = make_synthetic_dataset()
        X, y = dataset.X.to_numpy(), dataset.y_encoded()
        experiment = DecisionTreeExperiment(
            dataset.feature_names, dataset.target_names, depths=(1, 3, None)
        )
        experiment.run(X, y, X, y)
        comparison = experiment.comparison_frame()
        self.assertEqual(len(comparison), 3)
        self.assertIn("leaves", comparison.columns)

    def test_deeper_tree_is_not_worse_on_train(self) -> None:
        dataset = DatasetLoader().load("iris")
        X, y = dataset.X.to_numpy(), dataset.y_encoded()
        experiment = DecisionTreeExperiment(dataset.feature_names, dataset.target_names,
                                            depths=(1, 3, 5, None))
        experiment.run(X, y, X, y)
        accuracies = experiment.comparison_frame()["train_accuracy"].tolist()
        self.assertEqual(accuracies, sorted(accuracies))

    def test_best_result_is_selected(self) -> None:
        dataset = make_synthetic_dataset()
        X, y = dataset.X.to_numpy(), dataset.y_encoded()
        experiment = DecisionTreeExperiment(dataset.feature_names, dataset.target_names,
                                            depths=(1, 2, 3))
        experiment.run(X, y, X, y)
        self.assertIsNotNone(experiment.best_result().model)


class TestDatasets(unittest.TestCase):
    """Загрузка всех наборов входных данных."""

    def test_all_datasets_load(self) -> None:
        loader = DatasetLoader()
        expected = {"iris": (150, 4, 3), "wine": (178, 13, 3), "penguins": (342, 4, 3)}
        for name, (n, m, k) in expected.items():
            with self.subTest(dataset=name):
                dataset = loader.load(name)
                self.assertEqual((dataset.n_samples, dataset.n_features, dataset.n_classes),
                                 (n, m, k))
                self.assertEqual(dataset.X.isna().sum().sum(), 0)
                self.assertEqual(len(dataset.y), dataset.n_samples)

    def test_unknown_dataset_raises(self) -> None:
        with self.assertRaises(ValueError):
            DatasetLoader().load("titanic")

    def test_encoded_labels_are_consistent(self) -> None:
        dataset = DatasetLoader().load("iris")
        encoded = dataset.y_encoded()
        self.assertEqual(set(encoded.tolist()), {0, 1, 2})
        self.assertEqual(dataset.target_names[encoded[0]], dataset.y.iloc[0])

    def test_class_balance_sums_to_total(self) -> None:
        for name in DatasetLoader.AVAILABLE:
            with self.subTest(dataset=name):
                dataset = DatasetLoader().load(name)
                self.assertEqual(int(dataset.class_balance()["count"].sum()), dataset.n_samples)


class TestAnalyzer(unittest.TestCase):
    def test_eta_squared_is_high_for_separating_feature(self) -> None:
        analyzer = DatasetAnalyzer(make_synthetic_dataset())
        importance = analyzer.feature_importance_frame()
        self.assertGreater(importance["eta_squared"].max(), 0.8)

    def test_eta_squared_in_range(self) -> None:
        for name in DatasetLoader.AVAILABLE:
            with self.subTest(dataset=name):
                importance = DatasetAnalyzer(DatasetLoader().load(name)).feature_importance_frame()
                self.assertTrue(((importance["eta_squared"] >= 0)
                                 & (importance["eta_squared"] <= 1)).all())

    def test_top_features_count(self) -> None:
        analyzer = DatasetAnalyzer(DatasetLoader().load("wine"))
        self.assertEqual(len(analyzer.top_features(3)), 3)


class TestPipeline(unittest.TestCase):
    """Сквозной прогон конвейера на синтетических данных."""

    def test_pipeline_end_to_end(self) -> None:
        dataset = make_synthetic_dataset(n_per_class=30, seed=5)
        # Для синтетического набора нет библиотечных правил -- подставляем свои.
        original_build = RuleLibrary.build
        RuleLibrary.build = staticmethod(                       # type: ignore[assignment]
            lambda _name: RuleBasedClassifier(
                [
                    Rule([Condition("f1", "<", 2.5)], "A"),
                    Rule([Condition("f2", ">", 2.5)], "B"),
                ],
                default_class="C",
            )
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                config = LabConfig(iteration_grid=(5, 10), tree_depths=(2, 3))
                pipeline = Lab1Pipeline(dataset, tmp_path / "results", tmp_path / "figures", config)
                with contextlib.redirect_stdout(io.StringIO()):
                    comparison = pipeline.run()

                self.assertEqual(len(comparison), 4)   # правила, дерево, 2 линейные модели
                results_dir = tmp_path / "results" / "synthetic"
                figures_dir = tmp_path / "figures" / "synthetic"
                for filename in (
                    "01_dataset_overview.csv", "10_predictions_manual_rules.csv",
                    "20_tree_depth_comparison.csv", "30_linear_iterations_comparison.csv",
                    "40_model_comparison.csv", "50_benchmark.csv", "90_summary.txt",
                ):
                    self.assertTrue((results_dir / filename).exists(), filename)
                self.assertGreaterEqual(len(list(figures_dir.glob("*.png"))), 10)

                predictions = pd.read_csv(results_dir / "10_predictions_manual_rules.csv")
                self.assertEqual(len(predictions), dataset.n_samples)
                self.assertIn("predicted_class", predictions.columns)
        finally:
            RuleLibrary.build = original_build           # type: ignore[assignment]

    def test_split_is_stratified_and_disjoint(self) -> None:
        dataset = make_synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Lab1Pipeline(dataset, Path(tmp) / "r", Path(tmp) / "f")
            self.assertEqual(
                len(set(pipeline.idx_train) & set(pipeline.idx_test)), 0
            )
            self.assertEqual(
                len(pipeline.idx_train) + len(pipeline.idx_test), dataset.n_samples
            )
            self.assertEqual(len(np.unique(pipeline.y_test)), dataset.n_classes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
