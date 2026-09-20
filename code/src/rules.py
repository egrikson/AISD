"""Классификатор на основе правил, сформулированных вручную.

Реализует пункт задания «Вручную сформировать правила вида
ЕСЛИ sepal length > 3 cm AND sepal width < 4.5 cm ТО Versicolor» и
автоматизированное применение этих правил к датасету.

Структура данных: упорядоченный список объектов :class:`Rule`; каждое правило
-- это конъюнкция элементарных условий :class:`Condition`. Классификация
объекта выполняется линейным проходом по списку до первого сработавшего
правила, то есть за O(r * c), где r -- число правил, c -- условий в правиле.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass(frozen=True)
class Condition:
    """Элементарное условие вида «признак ОПЕРАТОР порог»."""

    feature: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in OPERATORS:
            raise ValueError(f"Недопустимый оператор {self.op!r}")

    def evaluate(self, column: pd.Series) -> np.ndarray:
        """Векторизованная проверка условия для всего столбца."""
        return OPERATORS[self.op](column.to_numpy(dtype=float), self.threshold)

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:g}"


@dataclass
class Rule:
    """Правило: конъюнкция условий и предсказываемый класс."""

    conditions: list[Condition]
    target: str
    comment: str = ""
    _fired: int = field(default=0, init=False, repr=False)

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        """Булева маска объектов, для которых выполнены все условия."""
        result = np.ones(len(X), dtype=bool)
        for condition in self.conditions:
            result &= condition.evaluate(X[condition.feature])
        return result

    def __str__(self) -> str:
        body = " AND ".join(str(c) for c in self.conditions)
        text = f"ЕСЛИ {body} ТО {self.target}"
        return f"{text}   # {self.comment}" if self.comment else text


class RuleBasedClassifier:
    """Классификатор «если-то» с правилом по умолчанию."""

    def __init__(self, rules: list[Rule], default_class: str, name: str = "manual_rules") -> None:
        self.rules = list(rules)
        self.default_class = default_class
        self.name = name

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Применяет правила по порядку; первое сработавшее задаёт класс."""
        predictions = np.full(len(X), self.default_class, dtype=object)
        assigned = np.zeros(len(X), dtype=bool)
        for rule in self.rules:
            mask = rule.mask(X) & ~assigned
            rule._fired = int(mask.sum())
            predictions[mask] = rule.target
            assigned |= mask
        return predictions.astype(str)

    def coverage_frame(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Сколько объектов покрыло каждое правило и с какой точностью."""
        predictions = np.full(len(X), None, dtype=object)
        assigned = np.zeros(len(X), dtype=bool)
        rows = []
        for rule in self.rules:
            mask = rule.mask(X) & ~assigned
            covered = int(mask.sum())
            correct = int((y.to_numpy()[mask] == rule.target).sum()) if covered else 0
            rows.append(
                {
                    "rule": str(rule).split("   #")[0],
                    "predicted_class": rule.target,
                    "covered": covered,
                    "correct": correct,
                    "rule_precision": round(correct / covered, 4) if covered else 0.0,
                }
            )
            predictions[mask] = rule.target
            assigned |= mask
        rest = int((~assigned).sum())
        correct_rest = int((y.to_numpy()[~assigned] == self.default_class).sum()) if rest else 0
        rows.append(
            {
                "rule": f"ИНАЧЕ (правило по умолчанию) ТО {self.default_class}",
                "predicted_class": self.default_class,
                "covered": rest,
                "correct": correct_rest,
                "rule_precision": round(correct_rest / rest, 4) if rest else 0.0,
            }
        )
        return pd.DataFrame(rows)

    def as_text(self) -> str:
        """Текстовое представление набора правил (для отчёта)."""
        lines = [f"{i}. {rule}" for i, rule in enumerate(self.rules, start=1)]
        lines.append(f"{len(self.rules) + 1}. ИНАЧЕ ТО {self.default_class}")
        return "\n".join(lines)


class RuleLibrary:
    """Наборы правил, сформулированные вручную по результатам анализа данных."""

    @staticmethod
    def build(dataset_name: str) -> RuleBasedClassifier:
        name = dataset_name.lower()
        builder = getattr(RuleLibrary, f"_{name}", None)
        if builder is None:
            raise ValueError(f"Правила для датасета {dataset_name!r} не заданы")
        return builder()

    # ------------------------------------------------------------------ iris
    @staticmethod
    def _iris() -> RuleBasedClassifier:
        rules = [
            Rule(
                [Condition("petal_length", "<", 2.45)],
                "setosa",
                "setosa линейно отделима по длине лепестка",
            ),
            Rule(
                [Condition("petal_width", "<", 1.75), Condition("petal_length", "<", 4.95)],
                "versicolor",
                "узкий и короткий лепесток",
            ),
        ]
        return RuleBasedClassifier(rules, default_class="virginica", name="iris_manual_rules")

    # ------------------------------------------------------------------ wine
    @staticmethod
    def _wine() -> RuleBasedClassifier:
        rules = [
            Rule(
                [Condition("flavanoids", ">", 2.0), Condition("proline", ">", 755)],
                "class_0",
                "много флавоноидов и высокое содержание пролина",
            ),
            Rule(
                [Condition("flavanoids", "<", 1.6), Condition("color_intensity", ">", 3.8)],
                "class_2",
                "мало флавоноидов при насыщенном цвете",
            ),
        ]
        return RuleBasedClassifier(rules, default_class="class_1", name="wine_manual_rules")

    # -------------------------------------------------------------- penguins
    @staticmethod
    def _penguins() -> RuleBasedClassifier:
        rules = [
            Rule(
                [
                    Condition("flipper_length_mm", ">", 205),
                    Condition("culmen_depth_mm", "<", 17.0),
                ],
                "Gentoo",
                "длинные ласты и неглубокий клюв",
            ),
            Rule(
                [Condition("culmen_length_mm", "<", 43.0)],
                "Adelie",
                "короткий клюв",
            ),
        ]
        return RuleBasedClassifier(rules, default_class="Chinstrap", name="penguins_manual_rules")
