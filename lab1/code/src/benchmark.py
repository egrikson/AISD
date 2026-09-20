"""Измерение времени работы и потребления памяти.

Используется для пункта отчёта «Результаты тестирования (время/память)».
Время измеряется монотонным таймером ``time.perf_counter``, память --
стандартным модулем ``tracemalloc`` (пиковый объём выделений Python).
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import pandas as pd


class Benchmark:
    """Контекстный менеджер для замера времени и пиковой памяти.

    Пример::

        with Benchmark() as bench:
            model.fit(X, y)
        print(bench.seconds, bench.peak_kb)
    """

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self.peak_kb: float = 0.0
        self._start: float = 0.0
        self._owns_tracemalloc: bool = False

    def __enter__(self) -> "Benchmark":
        self._owns_tracemalloc = not tracemalloc.is_tracing()
        if self._owns_tracemalloc:
            tracemalloc.start()
        else:
            tracemalloc.reset_peak()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.seconds = time.perf_counter() - self._start
        _current, peak = tracemalloc.get_traced_memory()
        self.peak_kb = peak / 1024
        if self._owns_tracemalloc:
            tracemalloc.stop()
        return False


@dataclass
class BenchmarkTable:
    """Накопитель результатов замеров для итоговой таблицы отчёта."""

    rows: list[dict[str, object]] = field(default_factory=list)

    def add(self, stage: str, bench: Benchmark, note: str = "") -> None:
        self.rows.append(
            {
                "stage": stage,
                "time_ms": round(bench.seconds * 1000, 3),
                "peak_memory_kb": round(bench.peak_kb, 2),
                "note": note,
            }
        )

    def add_manual(self, stage: str, time_ms: float, peak_kb: float, note: str = "") -> None:
        self.rows.append(
            {
                "stage": stage,
                "time_ms": round(time_ms, 3),
                "peak_memory_kb": round(peak_kb, 2),
                "note": note,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
