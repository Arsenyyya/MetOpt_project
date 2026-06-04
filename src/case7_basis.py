import numpy as np


def polynomial_basis_1d(x, degree):
    x = np.asarray(x)
    return np.column_stack([x ** power for power in range(1, degree + 1)])


def build_additive_features(X, degree):
    blocks = []
    for j in range(X.shape[1]):
        blocks.append(polynomial_basis_1d(X[:, j], degree))
    return np.hstack(blocks)


def component_slices(p, degree):
    return [slice(j * degree, (j + 1) * degree) for j in range(p)]


def basis_on_grid(t, degree):
    return polynomial_basis_1d(t, degree)

