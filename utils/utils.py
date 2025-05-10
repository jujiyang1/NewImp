import torch, random, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hyperimpute.plugins.utils.simulate import simulate_nan
import pandas as pd
import ot

def enable_reproducible_results(seed: int = 0) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def ampute(x, mechanism, p_miss):
    x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)


def scale_data(X):
    X = np.asarray(X)
    # preproc = MinMaxScaler()
    preproc = StandardScaler()
    # preproc.fit(X)
    return np.asarray(preproc.fit_transform(X))

def diff_scale_data(X):
    X = np.asarray(X)
    # preproc = MinMaxScaler()
    preproc = MinMaxScaler()
    print(f"we are using minmax scaler for diffusion models!")
    # preproc.fit(X)
    return np.asarray(preproc.fit_transform(X))

def simulate_scenarios(X, mechanisms=["MAR", "MNAR", "MCAR"], percentages=[0.1, 0.3, 0.5, 0.7], diff_model=False):
    X = scale_data(X) if not diff_model else diff_scale_data(X)
    datasets = {}

    for ampute_mechanism in mechanisms:
        for p_miss in percentages:
            if ampute_mechanism not in datasets:
                datasets[ampute_mechanism] = {}

            datasets[ampute_mechanism][p_miss] = ampute(X, ampute_mechanism, p_miss)

    return datasets


# third party
import numpy as np


def MAE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, verbose=0) -> np.ndarray:
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        MAE : np.ndarray
    """
    mask_ = mask.astype(bool)
    if verbose == 0:
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else:
        num_miss = mask_.sum(axis=0)
        output = []
        for i in range(X.shape[-1]):
            if num_miss[i] == 0:
                output += ['0']
            else:
                _output = np.absolute(X[:, i][mask_[:, i]]-X_true[:, i][mask_[:, i]])
                _output = _output.sum() / mask_[:, i].sum()
                output += [str(_output.round(5))]

        return output


def RMSE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, verbose=0) -> np.ndarray:
    """
    Root Mean Squared Error (RMSE) between imputed variables and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        RMSE : np.ndarray
    """
    mask_ = mask.astype(bool)
    if verbose == 0:
        return np.sqrt(np.mean(np.square(X[mask_] - X_true[mask_])))
    else:
        num_miss = mask_.sum(axis=0)
        output = []
        for i in range(X.shape[-1]):
            if num_miss[i] == 0:
                output += ['0']
            else:
                _output = np.sqrt(np.mean(np.square(X[:, i][mask_[:, i]] - X_true[:, i][mask_[:, i]])))
                output += [str(_output.round(5))]

        return output


def overall_MAE(X: np.ndarray, X_true: np.ndarray) -> float:
    """
    Calculate overall Mean Absolute Error (MAE) between two datasets.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.

    Returns:
        MAE : float
    """
    return np.mean(np.absolute(X - X_true))


def overall_RMSE(X: np.ndarray, X_true: np.ndarray) -> float:
    """
    Calculate overall Root Mean Squared Error (RMSE) between two datasets.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.

    Returns:
        RMSE : float
    """
    return np.sqrt(np.mean(np.square(X - X_true)))


def calculate_wasserstein_distance(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, max_samples: int = 5000) -> float:
    """
    Calculate Wasserstein distance between imputed values and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)
        max_samples : Maximum number of samples to use for calculation (for performance)

    Returns:
        Wasserstein distance : float
    """
    mask_ = mask.astype(bool)

    # If dataset is too large, skip calculation to avoid performance issues
    if X.shape[0] > max_samples:
        return 0.0

    # Get indices of rows with missing values
    M = mask_.sum(1) > 0
    nimp = M.sum()

    if nimp == 0:
        return 0.0

    # Calculate squared Euclidean distance matrix
    dist = ((X[M][:, None] - X_true[M]) ** 2).sum(2) / 2.

    # Calculate Wasserstein distance using POT library
    return ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dist)


__all__ = ["MAE", "RMSE", "overall_MAE", "overall_RMSE", "calculate_wasserstein_distance"]
