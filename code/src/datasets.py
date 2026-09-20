"""Загрузка и подготовка датасетов.

Модуль инкапсулирует работу с тремя наборами данных:
  * iris     -- основной датасет лабораторной работы;
  * wine     -- задание на защиту №1;
  * penguins -- задание на защиту №2 (содержит пропуски).

Данные читаются из локальной папки ``code/data`` (репозиторий самодостаточен и
работает без интернета). Если локального файла нет, выполняется попытка
скачать его по адресу из технического задания.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _plural_rows(count: int) -> str:
    """Согласование слова «строка» с числительным: 1 строка, 2 строки, 5 строк."""
    if 11 <= count % 100 <= 14:
        return "строк"
    return {1: "строка", 2: "строки", 3: "строки", 4: "строки"}.get(count % 10, "строк")


@dataclass
class Dataset:
    """Абстрактный тип данных «набор размеченных объектов».

    Атрибуты:
        name          -- машинное имя набора (iris / wine / penguins);
        title         -- человекочитаемое название;
        X             -- матрица признаков (pandas.DataFrame, n x m);
        y             -- вектор меток классов (pandas.Series длины n);
        feature_names -- список названий признаков;
        target_names  -- список названий классов;
        source        -- откуда получены данные;
        notes         -- заметки о предобработке (например, о пропусках).
    """

    name: str
    title: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    target_names: list[str]
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.target_names)

    def y_encoded(self) -> np.ndarray:
        """Метки классов в виде целых чисел 0..k-1."""
        mapping = {name: idx for idx, name in enumerate(self.target_names)}
        return self.y.map(mapping).to_numpy(dtype=int)

    def class_balance(self) -> pd.DataFrame:
        """Баланс классов: количество объектов и доля от выборки."""
        counts = self.y.value_counts().reindex(self.target_names).fillna(0).astype(int)
        return pd.DataFrame(
            {
                "class": counts.index,
                "count": counts.to_numpy(),
                "share_%": (counts.to_numpy() / self.n_samples * 100).round(2),
            }
        ).reset_index(drop=True)


class DatasetLoader:
    """Фабрика наборов данных."""

    URLS = {
        "iris": "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv",
        "penguins": "https://philchodrow.github.io/PIC16A/datasets/palmer_penguins.csv",
    }
    AVAILABLE = ("iris", "wine", "penguins")

    def __init__(self, data_dir: Path | str = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utils
    def _read_csv(self, filename: str, url_key: str) -> pd.DataFrame:
        """Читает CSV из локальной папки, при отсутствии -- скачивает."""
        path = self.data_dir / filename
        if not path.exists():
            url = self.URLS[url_key]
            print(f"[data] локальный файл {path.name} не найден, скачиваю {url}")
            urllib.request.urlretrieve(url, path)
        return pd.read_csv(path)

    # --------------------------------------------------------------- датасеты
    def load(self, name: str) -> Dataset:
        """Возвращает набор данных по его имени."""
        name = name.lower()
        if name not in self.AVAILABLE:
            raise ValueError(f"Неизвестный датасет {name!r}, доступны: {self.AVAILABLE}")
        return getattr(self, f"_load_{name}")()

    def _load_iris(self) -> Dataset:
        df = self._read_csv("iris.csv", "iris").dropna()
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        targets = ["setosa", "versicolor", "virginica"]
        return Dataset(
            name="iris",
            title="Iris (ирисы Фишера)",
            X=df[features].astype(float).reset_index(drop=True),
            y=df["species"].astype(str).reset_index(drop=True),
            feature_names=features,
            target_names=targets,
            source=self.URLS["iris"],
            notes=["Пропусков нет, датасет идеально сбалансирован (50/50/50)."],
        )

    def _load_wine(self) -> Dataset:
        from sklearn.datasets import load_wine

        wine = load_wine(as_frame=True)
        targets = [str(t) for t in wine.target_names]
        y = wine.target.map(dict(enumerate(targets)))
        return Dataset(
            name="wine",
            title="Wine (классификация сортов вина)",
            X=wine.data.astype(float).reset_index(drop=True),
            y=y.astype(str).reset_index(drop=True),
            feature_names=list(wine.feature_names),
            target_names=targets,
            source="sklearn.datasets.load_wine (оригинал: UCI Wine Data Set)",
            notes=[
                "Пропусков нет, классы слегка несбалансированы (59/71/48).",
                "Признаки имеют разный масштаб (от 0.13 до 1680) -> нужна стандартизация.",
            ],
        )

    def _load_penguins(self) -> Dataset:
        raw = self._read_csv("palmer_penguins.csv", "penguins")
        features = [
            "Culmen Length (mm)",
            "Culmen Depth (mm)",
            "Flipper Length (mm)",
            "Body Mass (g)",
        ]
        df = raw[features + ["Species"]].copy()
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_dropped = n_before - len(df)
        # "Adelie Penguin (Pygoscelis adeliae)" -> "Adelie"
        df["Species"] = df["Species"].astype(str).str.split().str[0]
        targets = ["Adelie", "Chinstrap", "Gentoo"]
        short = {
            "Culmen Length (mm)": "culmen_length_mm",
            "Culmen Depth (mm)": "culmen_depth_mm",
            "Flipper Length (mm)": "flipper_length_mm",
            "Body Mass (g)": "body_mass_g",
        }
        X = df[features].astype(float).rename(columns=short).reset_index(drop=True)
        return Dataset(
            name="penguins",
            title="Palmer Penguins (пингвины архипелага Палмера)",
            X=X,
            y=df["Species"].reset_index(drop=True),
            feature_names=list(X.columns),
            target_names=targets,
            source=self.URLS["penguins"],
            notes=[
                f"Обнаружены пропуски: удалено {n_dropped} {_plural_rows(n_dropped)} "
                f"из {n_before}.",
                "Классы несбалансированы (~146/68/119).",
            ],
        )
