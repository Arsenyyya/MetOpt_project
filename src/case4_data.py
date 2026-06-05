import numpy as np
from scipy.integrate import trapezoid


def make_grid(n_points: int = 100) -> np.ndarray:
    """Create a uniform grid on the interval [0, 1]."""
    return np.linspace(0.0, 1.0, n_points)


def true_weight_function(t: np.ndarray) -> np.ndarray:
    """Return the true weight function values on the grid."""
    return 2 * np.sin(2 * np.pi * t) - t


def generate_functional_data(
    n_samples: int = 300,
    n_points: int = 100,
    x_noise: float = 0.10,
    y_noise: float = 0.10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate synthetic functional regression data."""
    rng = np.random.default_rng(random_state)
    t = make_grid(n_points)

    a = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    b = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    c = rng.normal(loc=0.0, scale=0.6, size=n_samples)

    base = (
        a[:, None] * np.sin(2 * np.pi * t)[None, :]
        + b[:, None] * np.cos(2 * np.pi * t)[None, :]
        + c[:, None] * t[None, :]
    )
    X_func = base + rng.normal(0.0, x_noise, size=base.shape)

    y_clean = (
        2 * trapezoid(X_func * np.sin(2 * np.pi * t)[None, :], t, axis=1)
        - trapezoid(X_func * t[None, :], t, axis=1)
    )
    y = y_clean + rng.normal(0.0, y_noise, size=n_samples)

    params = {"a": a, "b": b, "c": c, "y_clean": y_clean}
    return t, X_func, y, params
