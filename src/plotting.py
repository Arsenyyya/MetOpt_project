import matplotlib.pyplot as plt
import numpy as np


def plot_functions(t, functions, labels=None, title="Графики функций", xlabel="t", ylabel="value", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    functions = np.asarray(functions)
    if functions.ndim == 1:
        functions = functions[None, :]

    if labels is None:
        labels = [f"f{i + 1}" for i in range(functions.shape[0])]

    for values, label in zip(functions, labels):
        ax.plot(t, values, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_error_curve(param_values, train_values, test_values, xlabel, ylabel="MSE", title="Ошибка модели"):
    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(param_values, train_values, marker="o", label="train")
    ax.plot(param_values, test_values, marker="o", label="test")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_true_vs_pred(y_true, y_pred, title="Истинные и предсказанные значения"):
    _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7)

    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))
    ax.plot([low, high], [low, high], color="black", linestyle="--", label="идеально")

    ax.set_title(title)
    ax.set_xlabel("y true")
    ax.set_ylabel("y pred")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_components(t, components, labels=None, true_components=None, title="Восстановленные компоненты"):
    components = np.asarray(components)
    if components.ndim == 1:
        components = components[None, :]

    if labels is None:
        labels = [f"g{i + 1}" for i in range(components.shape[0])]

    _, ax = plt.subplots(figsize=(8, 4))
    for idx, values in enumerate(components):
        ax.plot(t, values, label=f"{labels[idx]} estimate")

    if true_components is not None:
        true_components = np.asarray(true_components)
        if true_components.ndim == 1:
            true_components = true_components[None, :]
        for idx, values in enumerate(true_components):
            ax.plot(t, values, linestyle="--", label=f"{labels[idx]} true")

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("component value")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax

