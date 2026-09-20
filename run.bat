@echo off
rem Запуск лабораторной работы №1 одним файлом (Windows).
rem Создаёт виртуальное окружение, ставит зависимости, считает и собирает отчёт.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist ".venv" (
    echo [1/4] Создаю виртуальное окружение .venv ...
    py -3 -m venv .venv
    if errorlevel 1 python -m venv .venv
    if errorlevel 1 (
        echo Не удалось создать виртуальное окружение. Установите Python 3.10+ и повторите.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo [2/4] Устанавливаю зависимости ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo Не удалось установить зависимости. Проверьте подключение к интернету.
    pause
    exit /b 1
)

echo [3/4] Выполняю лабораторную работу (Iris, Wine, Penguins) ...
python "code\main.py" --dataset all
if errorlevel 1 (
    echo Ошибка при выполнении лабораторной работы.
    pause
    exit /b 1
)

echo [4/4] Собираю отчёт ...
python "report\build_report.py"

echo.
echo Готово. Таблицы: code\results, графики: report\images, отчёт: report\Отчет_ЛР1.docx
pause
