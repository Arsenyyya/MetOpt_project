from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / ".report_assets"
OUTPUT = ROOT / "report.pdf"
SOURCE = ROOT / "report.md"
REPO_URL = "https://github.com/Arsenyyya/MetOpt_project"

sys.path.insert(0, str(ROOT))

from src.case4_data import generate_functional_data, true_weight_function
from src.case4_functionals import (
    build_piecewise_features,
    build_trig_features,
    recover_piecewise_weight,
    recover_weight_from_basis,
)
from src.case4_models import evaluate_model, fit_ols, fit_ridge
from src.case7_data import (
    generate_additive_data,
    generate_nonadditive_data,
    true_components_on_grid,
)
from src.case7_models import (
    additive_component_values,
    center_components,
    component_curves,
    evaluate_predictions,
    fit_additive_ridge,
    fit_linear_model,
    predict_additive,
)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(ASSETS / name, dpi=180, bbox_inches="tight")
    plt.close()


def build_figures() -> None:
    ASSETS.mkdir(exist_ok=True)
    random_state = 42

    t, x_func, y, _ = generate_functional_data(
        n_samples=320,
        n_points=120,
        x_noise=0.10,
        y_noise=0.10,
        random_state=random_state,
    )
    x_train_func, x_test_func, y_train, y_test = train_test_split(
        x_func, y, test_size=0.25, random_state=random_state
    )

    plt.figure(figsize=(8, 4))
    for i in range(5):
        plt.plot(t, x_func[i], label=f"x_{i}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Примеры сгенерированных функций")
    plt.grid(alpha=0.3)
    plt.legend(ncol=3)
    save_figure("case4_functions.png")

    z_piece_train, _, _, piece_edges = build_piecewise_features(
        x_train_func, t, 10, use_average=True
    )
    z_piece_test, _, _, _ = build_piecewise_features(
        x_test_func, t, 10, use_average=True
    )
    z_trig_train, trig_basis, _ = build_trig_features(x_train_func, t, 10)
    z_trig_test, _, _ = build_trig_features(x_test_func, t, 10)

    ols_trig = fit_ols(z_trig_train, y_train)
    ridge_piece = fit_ridge(z_piece_train, y_train, alpha=0.1)
    ridge_trig = fit_ridge(z_trig_train, y_train, alpha=0.1)
    ols_eval = evaluate_model(ols_trig, z_trig_train, y_train, z_trig_test, y_test)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, ols_eval["pred_test"], alpha=0.7)
    low = min(y_test.min(), ols_eval["pred_test"].min())
    high = max(y_test.max(), ols_eval["pred_test"].max())
    plt.plot([low, high], [low, high], "k--", label="идеальный прогноз")
    plt.xlabel("Истинное y")
    plt.ylabel("Предсказанное y")
    plt.title("OLS с тригонометрическими признаками")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure("case4_prediction.png")

    plt.figure(figsize=(8, 4))
    plt.plot(t, true_weight_function(t), label="истинная w(t)", linewidth=2)
    plt.plot(
        t,
        recover_piecewise_weight(t, ridge_piece.coef_, piece_edges, True),
        label="кусочно-постоянная оценка",
    )
    plt.plot(
        t,
        recover_weight_from_basis(ridge_trig.coef_, trig_basis),
        label="тригонометрическая оценка",
    )
    plt.xlabel("t")
    plt.ylabel("w(t)")
    plt.title("Восстановление функции весов")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure("case4_weight.png")

    m_rows = []
    for m_value in [2, 4, 6, 8, 10, 15, 20]:
        for name, builder in [
            ("локальные средние", build_piecewise_features),
            ("тригонометрические", build_trig_features),
        ]:
            train_features = builder(x_train_func, t, m_value)[0]
            test_features = builder(x_test_func, t, m_value)[0]
            model = fit_ols(train_features, y_train)
            result = evaluate_model(
                model, train_features, y_train, test_features, y_test
            )
            m_rows.append(
                (m_value, name, result["train"]["MSE"], result["test"]["MSE"])
            )
    m_df = pd.DataFrame(m_rows, columns=["m", "name", "train", "test"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, ["локальные средние", "тригонометрические"]):
        part = m_df[m_df["name"] == name]
        ax.plot(part["m"], part["train"], marker="o", label="train")
        ax.plot(part["m"], part["test"], marker="o", label="test")
        ax.set_title(name)
        ax.set_xlabel("Число функционалов m")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("MSE")
    fig.suptitle("Влияние числа функционалов")
    save_figure("case4_m.png")

    lambdas = np.logspace(-4, 2, 10)
    train_rmse, test_rmse = [], []
    for alpha in lambdas:
        model = fit_ridge(z_trig_train, y_train, alpha=alpha)
        result = evaluate_model(model, z_trig_train, y_train, z_trig_test, y_test)
        train_rmse.append(result["train"]["RMSE"])
        test_rmse.append(result["test"]["RMSE"])
    plt.figure(figsize=(7, 4))
    plt.plot(lambdas, train_rmse, marker="o", label="train")
    plt.plot(lambdas, test_rmse, marker="o", label="test")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Влияние Ridge-регуляризации")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure("case4_lambda.png")

    x7, y7, _, _ = generate_additive_data(
        n_samples=450,
        p=3,
        noise=0.10,
        random_state=random_state,
    )
    x_train, x_test, y_train7, y_test7 = train_test_split(
        x7, y7, test_size=0.25, random_state=random_state
    )
    linear = fit_linear_model(x_train, y_train7)
    linear_eval = evaluate_predictions(linear, x_train, y_train7, x_test, y_test7)
    add = fit_additive_ridge(x_train, y_train7, degree=6, alpha=0.01)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test7, linear_eval["pred_test"], alpha=0.7)
    low = min(y_test7.min(), linear_eval["pred_test"].min())
    high = max(y_test7.max(), linear_eval["pred_test"].max())
    plt.plot([low, high], [low, high], "k--", label="идеальный прогноз")
    plt.xlabel("Истинное y")
    plt.ylabel("Предсказанное y")
    plt.title("Линейная модель на аддитивных данных")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure("case7_linear_prediction.png")

    component_values = additive_component_values(add, x_train, degree=6)
    _, _, means = center_components(component_values, add.intercept_)
    grid = np.linspace(0, 1, 200)
    estimated = component_curves(add, 6, 3, grid, centers=means)
    true_curves = true_components_on_grid(grid)
    true_centered = true_curves - true_curves.mean(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for j, ax in enumerate(axes):
        ax.plot(grid, estimated[j], label="оценка")
        ax.plot(grid, true_centered[j], "--", label="истинная")
        ax.set_title(f"g_{j + 1}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Центрированное значение")
    axes[0].legend()
    fig.suptitle("Восстановленные компоненты аддитивной модели")
    save_figure("case7_components.png")

    m_values = [1, 2, 3, 5, 7, 10, 12]
    train_values, test_values = [], []
    for degree in m_values:
        model = fit_additive_ridge(x_train, y_train7, degree=degree, alpha=0.01)
        result = evaluate_predictions(
            model,
            x_train,
            y_train7,
            x_test,
            y_test7,
            predict_func=lambda fitted, values, d=degree: predict_additive(
                fitted, values, d
            ),
        )
        train_values.append(result["train"]["RMSE"])
        test_values.append(result["test"]["RMSE"])
    plt.figure(figsize=(7, 4))
    plt.plot(m_values, train_values, marker="o", label="train")
    plt.plot(m_values, test_values, marker="o", label="test")
    plt.xlabel("Число базисных функций M")
    plt.ylabel("RMSE")
    plt.title("Влияние сложности аддитивной модели")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure("case7_m.png")

    x_non, y_non, _ = generate_nonadditive_data(
        n_samples=450,
        p=3,
        noise=0.10,
        random_state=random_state,
    )
    xn_train, xn_test, yn_train, yn_test = train_test_split(
        x_non, y_non, test_size=0.25, random_state=random_state
    )
    add_non = fit_additive_ridge(xn_train, yn_train, degree=6, alpha=0.01)
    add_non_eval = evaluate_predictions(
        add_non,
        xn_train,
        yn_train,
        xn_test,
        yn_test,
        predict_func=lambda fitted, values: predict_additive(fitted, values, 6),
    )
    plt.figure(figsize=(5, 5))
    plt.scatter(yn_test, add_non_eval["pred_test"], alpha=0.7)
    low = min(yn_test.min(), add_non_eval["pred_test"].min())
    high = max(yn_test.max(), add_non_eval["pred_test"].max())
    plt.plot([low, high], [low, high], "k--", label="идеальный прогноз")
    plt.xlabel("Истинное y")
    plt.ylabel("Предсказанное y")
    plt.title("Аддитивная модель на неаддитивных данных")
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure("case7_nonadditive.png")


def clean(text: str) -> str:
    text = text.replace(r"\mathbb{R}", "R")
    text = re.sub(r"\\operatorname\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\hat\{?([A-Za-z])\}?", r"\1_hat", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    replacements = {
        r"\sin": "sin",
        r"\cos": "cos",
        r"\exp": "exp",
        r"\int": "int",
        r"\sum": "sum",
        r"\lambda": "lambda",
        r"\beta": "beta",
        r"\alpha": "alpha",
        r"\gamma": "gamma",
        r"\varepsilon": "epsilon",
        r"\eta": "eta",
        r"\varphi": "phi",
        r"\theta": "theta",
        r"\pi": "pi",
        r"\approx": "~",
        r"\ldots": "...",
        r"\cdot": "*",
        r"\in": " in ",
        r"\times": "x",
        r"\left": "",
        r"\right": "",
        r"\,": " ",
        r"\|": "|",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("**", "").replace("`", "")
    text = text.replace("$", "").replace("{", "").replace("}", "")
    text = re.sub(r"([A-Za-z0-9_])(?=(sin|cos|exp)\()", r"\1*", text)
    text = re.sub(r"(\))(?=(sin|cos|exp)\()", r"\1*", text)
    text = re.sub(r"([A-Za-z]_[A-Za-z0-9])t\b", r"\1*t", text)
    text = re.sub(r"\)(t)\b", r")*\1", text)
    text = re.sub(r"(\d)(?=int_)", r"\1*", text)
    text = re.sub(r"(\d)pi", r"\1*pi", text)
    text = re.sub(r"pi\s+([A-Za-z])", r"pi*\1", text)
    text = re.sub(r"(int_0\^1)(?=[A-Za-z])", r"\1 ", text)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\s+", " ", text).strip()


def make_styles():
    font_dir = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("TimesRU", font_dir / "Times New Roman.ttf"))
    pdfmetrics.registerFont(TTFont("TimesRU-Bold", font_dir / "Times New Roman Bold.ttf"))
    pdfmetrics.registerFont(TTFont("TimesRU-Italic", font_dir / "Times New Roman Italic.ttf"))
    pdfmetrics.registerFontFamily(
        "TimesRU", normal="TimesRU", bold="TimesRU-Bold", italic="TimesRU-Italic"
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="BodyRU", fontName="TimesRU", fontSize=11, leading=15,
        alignment=TA_JUSTIFY, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="TitleRU", fontName="TimesRU-Bold", fontSize=20, leading=25,
        alignment=TA_CENTER, spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="SubtitleRU", fontName="TimesRU", fontSize=13, leading=18,
        alignment=TA_CENTER, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="H1RU", fontName="TimesRU-Bold", fontSize=16, leading=20,
        spaceBefore=14, spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="H2RU", fontName="TimesRU-Bold", fontSize=13, leading=17,
        spaceBefore=11, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="CaptionRU", fontName="TimesRU-Italic", fontSize=9.5, leading=12,
        alignment=TA_CENTER, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SmallRU", fontName="TimesRU", fontSize=9, leading=12,
        alignment=TA_JUSTIFY, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="FormulaRU", fontName="TimesRU", fontSize=11, leading=16,
        alignment=TA_CENTER, leftIndent=1 * cm, rightIndent=1 * cm,
        spaceBefore=5, spaceAfter=8,
    ))
    return styles


def render_pdf() -> None:
    styles = make_styles()

    def page_decor(canvas, doc):
        canvas.saveState()
        canvas.setFont("TimesRU", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(
            2 * cm, 1.15 * cm,
            "Кейсы 4 и 7: функциональная и аддитивная регрессия",
        )
        canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Отчет по кейсам 4 и 7",
        author="Arsenyyya",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=page_decor))

    story = [
        Spacer(1, 2.2 * cm),
        Paragraph("ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ", styles["TitleRU"]),
        Paragraph("Кейсы 4 и 7", styles["TitleRU"]),
        Spacer(1, 0.7 * cm),
        Paragraph(
            "Регрессия по функциональным данным через линейные функционалы<br/>"
            "и построение аддитивной модели",
            styles["SubtitleRU"],
        ),
        Spacer(1, 1.5 * cm),
        Paragraph(
            f'Код и ноутбуки: <link href="{REPO_URL}" color="blue">{REPO_URL}</link>',
            styles["SubtitleRU"],
        ),
        Spacer(1, 5.5 * cm),
        Paragraph("2026", styles["SubtitleRU"]),
        PageBreak(),
        Paragraph("Аннотация", styles["H1RU"]),
        Paragraph(
            "В работе исследуются два способа сведения сложных регрессионных "
            "задач к линейному оцениванию параметров. В кейсе 4 функции "
            "заменяются конечным набором интегральных признаков. В кейсе 7 "
            "многомерная зависимость представляется суммой одномерных компонент. "
            "Для обоих кейсов проведены вычислительные эксперименты на "
            "синтетических данных, сопоставлены модели и исследовано влияние "
            "сложности представления.",
            styles["BodyRU"],
        ),
        Paragraph("Воспроизводимость", styles["H1RU"]),
        Paragraph(
            f'Исходный код и выполненные ноутбуки опубликованы в открытом '
            f'репозитории: <link href="{REPO_URL}" color="blue">{REPO_URL}</link>. '
            "Основные эксперименты используют random_state = 42.",
            styles["BodyRU"],
        ),
    ]

    images_by_heading = {
        "### 2.4 Результаты": [
            ("case4_functions.png", "Рисунок 1. Примеры сгенерированных функциональных наблюдений."),
            ("case4_prediction.png", "Рисунок 2. Истинные и предсказанные значения OLS."),
            ("case4_weight.png", "Рисунок 3. Истинная и восстановленные функции весов."),
            ("case4_m.png", "Рисунок 4. Зависимость ошибки от числа функционалов."),
            ("case4_lambda.png", "Рисунок 5. Зависимость качества от параметра регуляризации."),
        ],
        "### 3.4 Результаты": [
            ("case7_linear_prediction.png", "Рисунок 6. Предсказания линейной модели на аддитивных данных."),
            ("case7_components.png", "Рисунок 7. Истинные и восстановленные центрированные компоненты."),
            ("case7_m.png", "Рисунок 8. Влияние числа базисных функций на качество."),
            ("case7_nonadditive.png", "Рисунок 9. Аддитивная модель на неаддитивных данных."),
        ],
    }

    def add_images(items):
        for filename, caption in items:
            image = Image(str(ASSETS / filename))
            max_w, max_h = 16.2 * cm, 9.2 * cm
            scale = min(max_w / image.imageWidth, max_h / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            story.append(KeepTogether([
                Spacer(1, 5),
                image,
                Paragraph(caption, styles["CaptionRU"]),
            ]))

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    formula_mode = False
    formula_lines = []
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped == "# Отчет по кейсам 4 и 7":
            i += 1
            continue
        if stripped.startswith("Публичная ссылка на репозиторий") or stripped == "добавляется после публикации проекта.":
            i += 1
            continue
        if stripped == "$$":
            if formula_mode:
                story.append(Paragraph(clean(" ".join(formula_lines)), styles["FormulaRU"]))
                formula_lines = []
                formula_mode = False
            else:
                formula_mode = True
            i += 1
            continue
        if formula_mode:
            formula_lines.append(stripped)
            i += 1
            continue
        if stripped in images_by_heading:
            story.append(Paragraph(clean(stripped[4:]), styles["H2RU"]))
            add_images(images_by_heading[stripped])
            i += 1
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(clean(stripped[3:]), styles["H1RU"]))
            i += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(clean(stripped[4:]), styles["H2RU"]))
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for line in table_lines:
                cells = [clean(cell.strip()) for cell in line.strip("|").split("|")]
                if all(re.fullmatch(r"[-:]+", cell) for cell in cells):
                    continue
                rows.append([Paragraph(cell, styles["SmallRU"]) for cell in cells])
            if rows:
                table = Table(
                    rows,
                    colWidths=[doc.width / len(rows[0])] * len(rows[0]),
                    repeatRows=1,
                    hAlign="CENTER",
                )
                table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), "TimesRU"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#777777")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.extend([table, Spacer(1, 9)])
            continue
        if re.match(r"^\d+\. рисун", stripped):
            i += 1
            continue
        if stripped.startswith("- "):
            story.append(Paragraph("• " + clean(stripped[2:]), styles["BodyRU"]))
            i += 1
            continue
        if not stripped:
            i += 1
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines):
            next_line = lines[i].strip()
            if not next_line or next_line.startswith("#") or next_line.startswith("|") or next_line == "$$" or next_line.startswith("- "):
                break
            if re.match(r"^\d+\. рисун", next_line):
                break
            paragraph.append(next_line)
            i += 1
        story.append(Paragraph(clean(" ".join(paragraph)), styles["BodyRU"]))

    doc.build(story)


def main() -> None:
    build_figures()
    render_pdf()
    print(OUTPUT)


if __name__ == "__main__":
    main()
