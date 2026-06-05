import importlib
import numpy as np


def test_src_modules_importable():
    modules = [
        "src.case4_data",
        "src.case4_functionals",
        "src.case4_models",
        "src.case7_basis",
        "src.case7_data",
        "src.case7_models",
        "src.metrics",
        "src.plotting",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_case4_generator_shapes():
    from src.case4_data import generate_functional_data

    n_samples = 12
    n_points = 20

    t, X_func, y, params = generate_functional_data(
        n_samples=n_samples,
        n_points=n_points,
        random_state=42,
    )

    assert t.shape == (n_points,)
    assert X_func.shape == (n_samples, n_points)
    assert y.shape == (n_samples,)
    assert params["y_clean"].shape == (n_samples,)


def test_case7_generator_shapes():
    from src.case7_data import generate_additive_data, generate_nonadditive_data

    n_samples = 12
    p = 3

    X, y, components, y_clean = generate_additive_data(
        n_samples=n_samples,
        p=p,
        random_state=42,
    )

    assert X.shape == (n_samples, p)
    assert y.shape == (n_samples,)
    assert components.shape == (n_samples, 3)
    assert y_clean.shape == (n_samples,)

    X_nonadd, y_nonadd, y_clean_nonadd = generate_nonadditive_data(
        n_samples=n_samples,
        p=p,
        random_state=42,
    )

    assert X_nonadd.shape == (n_samples, p)
    assert y_nonadd.shape == (n_samples,)
    assert y_clean_nonadd.shape == (n_samples,)


def test_features_are_finite():
    from src.case4_data import generate_functional_data
    from src.case4_functionals import build_piecewise_features, build_trig_features
    from src.case7_basis import build_additive_features
    from src.case7_data import generate_additive_data

    t, X_func, _, _ = generate_functional_data(
        n_samples=10,
        n_points=30,
        random_state=42,
    )

    Z_piecewise, _, _, _ = build_piecewise_features(X_func, t, m=4)
    Z_trig, _, _ = build_trig_features(X_func, t, m=5)

    assert np.isfinite(Z_piecewise).all()
    assert np.isfinite(Z_trig).all()

    X, _, _, _ = generate_additive_data(
        n_samples=10,
        p=3,
        random_state=42,
    )

    Z_additive = build_additive_features(X, degree=3)

    assert np.isfinite(Z_additive).all()


def test_centering_components_preserves_predictions():
    from src.case7_models import center_components

    component_values = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])
    intercept = 0.5

    pred_before = intercept + component_values.sum(axis=1)

    centered, centered_intercept, _ = center_components(
        component_values,
        intercept,
    )

    pred_after = centered_intercept + centered.sum(axis=1)

    assert np.allclose(pred_before, pred_after)
    assert np.isfinite(centered).all()
    assert np.isfinite(centered_intercept)


def test_metrics_return_finite_numbers():
    from src.metrics import mse_score, rmse_score, r2_score_value, regression_metrics

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    assert np.isfinite(mse_score(y_true, y_pred))
    assert np.isfinite(rmse_score(y_true, y_pred))
    assert np.isfinite(r2_score_value(y_true, y_pred))

    values = regression_metrics(y_true, y_pred)

    assert set(values) == {"MSE", "RMSE", "R2"}
    assert all(np.isfinite(value) for value in values.values())
