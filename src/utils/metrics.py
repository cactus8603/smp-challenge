import numpy as np
from scipy.stats import spearmanr


def compute_mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def compute_mse(y_true, y_pred):
    """
    Mean Squared Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def compute_rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(compute_mse(y_true, y_pred))


def compute_spearman(y_true, y_pred):
    """
    Spearman Rank Correlation (最重要)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 避免 nan 問題
    if len(y_true) < 2:
        return 0.0

    corr, _ = spearmanr(y_true, y_pred)

    # spearmanr 可能回傳 nan（例如全部值一樣）
    if np.isnan(corr):
        return 0.0

    return corr