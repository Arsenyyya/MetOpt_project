from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .case7_basis import basis_on_grid, build_additive_features, component_slices
from .metrics import regression_metrics


def fit_linear_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Fit a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fit_additive_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    degree: int,
) -> LinearRegression:
    """Fit an additive OLS model."""
    Z_train = build_additive_features(X_train, degree)
    model = LinearRegression()
    model.fit(Z_train, y_train)
    return model


def fit_additive_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    degree: int,
    alpha: float = 1.0,
) -> Ridge:
    """Fit an additive Ridge model."""
    Z_train = build_additive_features(X_train, degree)
    model = Ridge(alpha=alpha)
    model.fit(Z_train, y_train)
    return model


def fit_polynomial_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    degree: int = 3,
    alpha: float | None = None,
) -> Pipeline:
    """Fit a polynomial regression pipeline."""
    regressor = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), regressor)
    model.fit(X_train, y_train)
    return model


def predict_additive(model, X: np.ndarray, degree: int) -> np.ndarray:
    """Predict values with an additive model."""
    Z = build_additive_features(X, degree)
    return model.predict(Z)


def evaluate_predictions(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predict_func: Callable[[object, np.ndarray], np.ndarray] | None = None,
) -> dict[str, object]:
    """Evaluate train and test predictions."""
    if predict_func is None:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
    else:
        pred_train = predict_func(model, X_train)
        pred_test = predict_func(model, X_test)

    return {
        "train": regression_metrics(y_train, pred_train),
        "test": regression_metrics(y_test, pred_test),
        "pred_train": pred_train,
        "pred_test": pred_test,
    }


def additive_component_values(model, X: np.ndarray, degree: int) -> np.ndarray:
    """Compute additive component values for observations."""
    p = X.shape[1]
    Z = build_additive_features(X, degree)
    parts = []
    for sl in component_slices(p, degree):
        parts.append(Z[:, sl] @ model.coef_[sl])
    return np.column_stack(parts)


def center_components(
    component_values: np.ndarray,
    intercept: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Center component values and adjust the intercept."""
    means = component_values.mean(axis=0)
    centered = component_values - means[None, :]
    centered_intercept = intercept + means.sum()
    return centered, centered_intercept, means


def component_curves(
    model,
    degree: int,
    p: int,
    grid: np.ndarray,
    centers: np.ndarray | None = None,
) -> np.ndarray:
    """Compute component curves on a grid."""
    B = basis_on_grid(grid, degree)
    curves = []
    for sl in component_slices(p, degree):
        curves.append(B @ model.coef_[sl])
    curves = np.asarray(curves)

    if centers is not None:
        curves = curves - np.asarray(centers)[:, None]

    return curves
