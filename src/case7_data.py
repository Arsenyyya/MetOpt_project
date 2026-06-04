import numpy as np


def additive_components(X):
    g1 = np.sin(2 * np.pi * X[:, 0])
    g2 = 0.5 * (X[:, 1] - 0.5) ** 2
    g3 = np.exp(-3 * X[:, 2])
    return np.column_stack([g1, g2, g3])


def generate_additive_data(n_samples=400, p=3, noise=0.10, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0.0, 1.0, size=(n_samples, p))

    components = additive_components(X[:, :3])
    y_clean = components.sum(axis=1)
    y = y_clean + rng.normal(0.0, noise, size=n_samples)

    return X, y, components, y_clean


def generate_nonadditive_data(n_samples=400, p=3, noise=0.10, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0.0, 1.0, size=(n_samples, p))

    y_clean = np.sin(2 * np.pi * (X[:, 0] + X[:, 1])) + 0.5 * X[:, 2]
    y = y_clean + rng.normal(0.0, noise, size=n_samples)

    return X, y, y_clean


def true_components_on_grid(t):
    return np.vstack([
        np.sin(2 * np.pi * t),
        0.5 * (t - 0.5) ** 2,
        np.exp(-3 * t),
    ])

