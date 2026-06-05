# MetOpt Project

Учебный репозиторий с выполненными кейсами 4 и 7.

В проекте находятся ноутбуки с расчетами, вспомогательные Python-модули, файл зависимостей и итоговый PDF-отчет.

## Структура проекта

```text
.
├── notebooks/
│   ├── case4_functional_regression.ipynb
│   └── case7_additive_model.ipynb
├── src/
│   ├── __init__.py
│   ├── case4_data.py
│   ├── case4_functionals.py
│   ├── case4_models.py
│   ├── case7_basis.py
│   ├── case7_data.py
│   ├── case7_models.py
│   ├── metrics.py
│   └── plotting.py
├── report.md
├── report.pdf
└── requirements.txt
```

## Установка

Склонируйте репозиторий:

```bash
git clone https://github.com/Arsenyyya/MetOpt_project.git
cd MetOpt_project
```

Создайте виртуальное окружение:

```bash
python -m venv .venv
```

Активируйте окружение.

Для Windows:

```bash
.venv\Scripts\activate
```

Для macOS/Linux:

```bash
source .venv/bin/activate
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

## Запуск ноутбуков

Запустите Jupyter Notebook:

```bash
jupyter notebook
```

После запуска откройте нужный ноутбук из папки `notebooks/`:

```text
notebooks/case4_functional_regression.ipynb
notebooks/case7_additive_model.ipynb
```

Также можно открыть проект через JupyterLab, если он установлен:

```bash
jupyter lab
```

## Отчет

Итоговый отчет доступен в файле:

[report.pdf](report.pdf)

Markdown-версия отчета находится в файле:

[report.md](report.md)

## Зависимости

Основные зависимости перечислены в файле `requirements.txt`:

```text
numpy
scipy
scikit-learn
matplotlib
pandas
jupyter
nbconvert
pytest
```
