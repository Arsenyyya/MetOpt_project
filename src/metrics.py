import numpy as np


def mse_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mse_score(y_true, y_pred)))


def r2_score_value(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred):
    return {
        "MSE": mse_score(y_true, y_pred),
        "RMSE": rmse_score(y_true, y_pred),
        "R2": r2_score_value(y_true, y_pred),
    }

