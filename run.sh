#!/usr/bin/env bash
# Запуск лабораторной работы №1 одним файлом (Linux / macOS / Git Bash).
# Создаёт виртуальное окружение, ставит зависимости, считает и собирает отчёт.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

# Ищем рабочий интерпретатор: в Git Bash под Windows "python3" часто оказывается
# заглушкой Microsoft Store, поэтому проверяем, что команда реально запускается.
detect_python() {
    for candidate in "${PYTHON:-}" python3 python; do
        [ -z "$candidate" ] && continue
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    if command -v py >/dev/null 2>&1 && py -3 -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
        echo "py -3"
        return 0
    fi
    return 1
}

if ! PY=$(detect_python); then
    echo "Не найден Python 3. Установите Python 3.10+ и повторите запуск." >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "[1/4] Создаю виртуальное окружение .venv (интерпретатор: $PY) ..."
    # shellcheck disable=SC2086
    $PY -m venv .venv
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
python lab1/code/main.py --dataset all

echo "[4/4] Собираю отчёт ..."
python lab1/report/build_report.py

echo
echo "Готово. Таблицы: lab1/code/results, графики: lab1/report/images, отчёт: lab1/report/Отчет_ЛР1.docx"
