import numpy as np
from scipy.integrate import trapezoid


def integrate_on_grid(values, t, weights=None):
    values = np.asarray(values)
    if weights is not None:
        values = values * np.asarray(weights)
    return trapezoid(values, t, axis=-1)


def piecewise_basis(t, m):
    edges = np.linspace(0.0, 1.0, m + 1)
    basis = []
    names = []

    for k in range(m):
        if k == m - 1:
            mask = (t >= edges[k]) & (t <= edges[k + 1])
        else:
            mask = (t >= edges[k]) & (t < edges[k + 1])
        basis.append(mask.astype(float))
        names.append(f"I{ k + 1 }")

    return np.asarray(basis), names, edges


def trig_basis(t, m):
    basis = []
    names = []

    if m >= 1:
        basis.append(np.ones_like(t))
        names.append("1")

    freq = 1
    while len(basis) < m:
        basis.append(np.sin(2 * np.pi * freq * t))
        names.append(f"sin({freq})")
        if len(basis) >= m:
            break
        basis.append(np.cos(2 * np.pi * freq * t))
        names.append(f"cos({freq})")
        freq += 1

    return np.asarray(basis), names


def build_features(X_func, t, basis):
    features = []
    for phi in basis:
        features.append(integrate_on_grid(X_func, t, weights=phi[None, :]))
    return np.column_stack(features)


def build_piecewise_features(X_func, t, m, use_average=True):
    basis, names, edges = piecewise_basis(t, m)
    Z = build_features(X_func, t, basis)
    if use_average:
        lengths = np.diff(edges)
        Z = Z / lengths[None, :]
    return Z, basis, names, edges


def build_trig_features(X_func, t, m):
    basis, names = trig_basis(t, m)
    Z = build_features(X_func, t, basis)
    return Z, basis, names


def recover_weight_from_basis(coef, basis):
    return np.asarray(coef) @ np.asarray(basis)


def recover_piecewise_weight(t, coef, edges, use_average=True):
    weight = np.zeros_like(t, dtype=float)
    coef = np.asarray(coef)
    lengths = np.diff(edges)

    for k in range(len(coef)):
        if k == len(coef) - 1:
            mask = (t >= edges[k]) & (t <= edges[k + 1])
        else:
            mask = (t >= edges[k]) & (t < edges[k + 1])
        weight[mask] = coef[k] / lengths[k] if use_average else coef[k]

    return weight

