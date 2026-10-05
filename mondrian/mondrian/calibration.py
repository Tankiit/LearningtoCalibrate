"""Binary calibration metrics from calibration-toolbox (Pearce, MIT; third_party/calibration-toolbox @ 1c6f60b).

A forecast p of P(Y=1) is passed to the toolbox as the two-class array [1-p, p].
For two classes the toolbox's class-wise (static) error SCE equals the usual binary
ECE of p against Y (tests/test_calibration_metrics.py); ECE, MCE, RMSCE and OE are
top-label, ACE uses equal-mass bins. All use the toolbox default of 15 bins.
"""
import numpy as np
import calibration_toolbox as ct

N_BINS = 15
NAMES = ("sce", "ece", "ace", "mce", "rmsce", "oe")


def two_class(p):
    p = np.clip(np.asarray(p, float), 0, 1)
    return np.column_stack([1 - p, p])


def metrics(p, y, n_bins=N_BINS):
    P, y = two_class(p), np.asarray(y).astype(int)
    return {"sce": float(ct.SCE(P, y, n_bins)), "ece": float(ct.ECE(P, y, n_bins)),
            "ace": float(ct.ACE(P, y, n_bins)), "mce": float(ct.MCE(P, y, n_bins)),
            "rmsce": float(ct.RMSCE(P, y, n_bins)), "oe": float(ct.OE(P, y, n_bins))}


def toolbox_version():
    import importlib.metadata
    return importlib.metadata.version("calibration-toolbox")
