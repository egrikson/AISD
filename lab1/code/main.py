"""Лабораторная работа №1 по курсу «Алгоритмы и структуры данных».

Анализ данных, классификация по правилам, дерево решений и линейная регрессия.

Запуск:
    python main.py                      # основной датасет Iris
    python main.py --dataset wine       # задание на защиту №1
    python main.py --dataset penguins   # задание на защиту №2
    python main.py --dataset all        # все три датасета

Результаты:
    lab1/code/results/<датасет>/   -- таблицы CSV и текстовые отчёты
    lab1/report/images/<датасет>/  -- графики PNG для отчёта
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Каталог code/ добавляется в путь поиска модулей, чтобы скрипт запускался
# из любой рабочей директории.
CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets import DatasetLoader          # noqa: E402
from src.pipeline import LabConfig, run_dataset  # noqa: E402

DEFAULT_RESULTS = CODE_DIR / "results"
DEFAULT_FIGURES = ROOT_DIR / "report" / "images"


def configure_console() -> None:
    """Включает UTF-8 в консоли Windows, иначе кириллица печатается кракозябрами."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа №1: анализ данных, дерево решений, линейная регрессия",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d", default="iris",
        choices=[*DatasetLoader.AVAILABLE, "all"],
        help="какой набор данных обработать",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3,
        help="доля тестовой выборки",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="зерно генератора случайных чисел (воспроизводимость)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100,
        help="максимальное число итераций обучения линейных моделей",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help="куда сохранять таблицы CSV",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES,
        help="куда сохранять графики PNG",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)

    step = max(args.max_iterations // 10, 1)
    config = LabConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        iteration_grid=tuple(range(step, args.max_iterations + 1, step)),
    )

    names = list(DatasetLoader.AVAILABLE) if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, args.results_dir, args.figures_dir, config)

    print("\nГотово. Обработано наборов данных: " + str(len(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
