import numpy as np
from scipy import stats

def statistics(X, alpha=0.95):
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    SE = sig/np.sqrt(X.shape[0])
    CI = stats.norm.interval(alpha, loc=mu, scale=SE)
    return mu, sig, CI[0], CI[1]