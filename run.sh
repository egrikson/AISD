#!/usr/bin/env bash
# Запуск лабораторной работы №1 одним файлом (Linux / macOS / Git Bash).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

PY=${PYTHON:-python3}

if [ ! -d .venv ]; then
    echo "[1/4] Создаю виртуальное окружение .venv ..."
    "$PY" -m venv .venv
fi

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    # shellcheck disable=SC1091
    source .venv/Scripts/activate      # Git Bash под Windows
fi

echo "[2/4] Устанавливаю зависимости ..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "[3/4] Выполняю лабораторную работу (Iris, Wine, Penguins) ..."
python code/main.py --dataset all

echo "[4/4] Собираю отчёт ..."
python report/build_report.py

echo
echo "Готово. Таблицы: code/results, графики: report/images, отчёт: report/Отчет_ЛР1.docx"
