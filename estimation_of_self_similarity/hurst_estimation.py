"""
hurst_estimation.py

Used for the estimation of the self-similarity index of a general H-self-similar process via the Lamperti transformation
This module also includes simple benchmarking helpers (serial and parallel)

Notes
- Method based on our paper "Statistical Inferences on Non-stationary Self-similar Processes via Lamperti Transformations" by W. Wu, Q. Peng 
- Simulators are imported from the `fractal_analysis` package (Ding, Peng, Wu).
"""

# estimate hurst values using Halley's method
from halleys_method import solve_H_halley, solve_H_halley_sfbm

# self-similar processes simulation
# imported from fractal-analaysis package by Ding, Peng, Wu
from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwFbmSimulator as DPW_FBM
from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwSubFbmSimulator as DPW_SFBM
from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwBiFbmSimulator as DPW_BFBM
from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwTriFbmSimulator as DPW_TFBM
from fractal_analysis.simulator.wood_chan.wood_chan_fractal_simulator import WoodChanFbmSimulator as WC_FBM
from fractal_analysis.estimator.hurst_estimator import QvHurstEstimator

# Other imports
from math import floor
import numpy as np
from scipy.optimize import minimize_scalar

# Compute A_j ^ 2 / (N - 1) for a given process X. 
# Returns a list of length N starting from i = 0 to N - 1
def compute_a(x):
    x = np.asarray(x)
    N = len(x)
    indices = [floor(N ** (i / N)) for i in range(0, N)]
    return (x[indices] ** 2) / (N-1)
    
# Compute B_j ^ 2 for a given N. 
# Returns a list of length N starting from i = 0 to N - 1
def compute_b(N=1024):
    i = np.arange(0, N, dtype=float)
    return N ** (2 * (1.0 - i/N))

# Functions for algorithm 2 in our paper, used for estimation of H when scaling parameter is unknown
# s1 is some multiple of the second order moment while s2 is some multiple of the fourth order moment
def f_H(H, a, b, N):
    logb = np.log(b)
    s1 = np.sum(a * np.exp(H*logb))
    s2 = np.sum(a**2 * np.exp(2*H*logb)) * (N-1)
    return (s1*s1/s2) ** -1

def f_H2(H, a, b, N):
    logb = np.log(b)
    s1 = np.sum(a * np.exp(H*logb))
    s2 = np.sum(a**2 * np.exp(2*H*logb)) * (N-1)
    return (s1/s2) ** -1

# Calculate lists a,b and use them to estimate H using Halley's method
# sum from j = 0 to n of a_j^2 * b_j^{2H} / (n-1) = Var(X(1))
# Note this function is designed to estimate the self-similarity index, not "H", in other words H * K is estimated for tfBm and bfBm
def estimate_hurst_method1(hss_path, process_type="DPW_FBM", *, H0=0.5, K=1, sigma=1):
    a = compute_a(hss_path) / (sigma**2)
    b = compute_b(len(hss_path))
    
    if process_type == "DPW_SFBM":
        H_halley = solve_H_halley_sfbm(a, b, H0=H0)
    elif process_type in ["DPW_FBM", "DPW_BFBM", "WC_FBM"]: 
        H_halley = solve_H_halley(a,b, H0=H0)
    elif process_type == "DPW_TFBM":
        H_halley = solve_H_halley(np.array(a / (2 - (2 ** K))),b, H0=H0)
    else: 
        raise ValueError("Unsupported type for Hurst estimation")
    return H_halley

# Estimate H when scaling parameter is unknown
# Computes the minimum value of the function given by f(H) as the estimated H value
# Estimates sigma^2 by computing the quotient of the fourth order moment and second order moment
def estimate_hurst_method2(hss_path, process_type="WC_FBM"):
    N = len(hss_path)
    a = compute_a(hss_path)
    b = compute_b(N)
    
    def objective(H):
        return f_H(H, a, b, N)

    res = minimize_scalar(
        objective,
        bounds=(1e-6, 1 - 1e-6),
        method='bounded'
    )

    H_min = res.x
    if process_type in ["WC_FBM", "DPW_FBM", "DPW_BFBM"]:
        sigma2_estimate = f_H2(H_min, a, b, N)
    elif process_type == "DPW_SFBM":
        sigma2_estimate = f_H2(H_min, a, b, N) / (2 - 2 ** (2 * H_min))
    else: 
        raise ValueError("Unsupported type for Hurst estimation")
    return H_min, sigma2_estimate

# Example testing of self-similarity parameter estimation
if __name__ == "__main__":
    fbm_path = DPW_FBM(sample_size=1024, hurst_parameter=0.5, lamperti_multiplier=5).get_fbm()
    print(estimate_hurst_method1(fbm_path, process_type=DPW_FBM, H0=0.5))
    print(estimate_hurst_method2(2 * fbm_path, process_type=DPW_FBM))