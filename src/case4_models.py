from sklearn.linear_model import LinearRegression, Ridge

from .metrics import regression_metrics


def fit_ols(Z_train, y_train):
    model = LinearRegression()
    model.fit(Z_train, y_train)
    return model


def fit_ridge(Z_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(Z_train, y_train)
    return model


def evaluate_model(model, Z_train, y_train, Z_test, y_test):
    pred_train = model.predict(Z_train)
    pred_test = model.predict(Z_test)
    return {
        "train": regression_metrics(y_train, pred_train),
        "test": regression_metrics(y_test, pred_test),
        "pred_train": pred_train,
        "pred_test": pred_test,
    }

