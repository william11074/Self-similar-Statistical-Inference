"""
hurst_estimation.py

Used for the estimation of the self-similarity index of a general H-self-similar process via the Lamperti transformation
This module also includes simple benchmarking helpers (serial and parallel)

Notes
- Method based on our paper "Detection and Estimation of Self-similarity via Lamperti Transformations" by W. Wu, Q. Peng 
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
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares
from itertools import permutations
import pandas as pd

# for parallel processing, opens multiple cores if available
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

# Calculate lists a,b and use them to estimate H using Halley's method
# sum from j = 0 to n of a_j^2 * b_j^{2H} / (n-1) = Var(X(1))
# Note this function is designed to estimate the self-similarity index, not "H", in other words H * K is estimated for tfBm and bfBm
def estimate_hurst(hss_path, process_type=DPW_FBM, *, H0=0.5, K=1):
    a = compute_a(hss_path)
    b = compute_b(len(hss_path))
    
    if process_type == DPW_SFBM:
        H_halley = solve_H_halley_sfbm(a, b, H0=H0)
    elif process_type in [DPW_FBM, DPW_BFBM, WC_FBM]: 
        H_halley = solve_H_halley(a,b, H0=H0)
    elif process_type == DPW_TFBM:
        H_halley = solve_H_halley(np.array(a / (2 - (2 ** K))),b, H0=H0)
    else: 
        raise ValueError("Unsupported type for Hurst estimation")
    return H_halley

# Estimate H for a single simulated path given path length, self-similarity index H, process type, K (bi factor or tri factor) and Lamperti multiplier
def single_trial(N=1024, H=0.5, process_type=DPW_FBM, K=1, Lamperti_multiplier=5):
    if process_type == WC_FBM:
        hss_path = process_type(sample_size = N, hurst_parameter=H).get_fbm()
    elif process_type == DPW_FBM:
        hss_path = process_type(sample_size = N, hurst_parameter=H, lamperti_multiplier=Lamperti_multiplier).get_fbm()
    elif process_type == DPW_SFBM:
        hss_path = process_type(sample_size = N, hurst_parameter=H, lamperti_multiplier=Lamperti_multiplier).get_sub_fbm()
    elif process_type == DPW_BFBM:
        hss_path = process_type(sample_size = N, hurst_parameter=H, bi_factor=K, lamperti_multiplier=Lamperti_multiplier).get_bi_fbm()
    elif process_type == DPW_TFBM:
        hss_path = process_type(sample_size = N, hurst_parameter=H, tri_factor=K, lamperti_multiplier=Lamperti_multiplier).get_tri_fbm()
    elif process_type == "qv":
        hss_path = WC_FBM(sample_size = N, hurst_parameter=H).get_fbm()
        return np.mean(QvHurstEstimator(mbm_series=hss_path, alpha=0.2).holder_exponents)
    else:
        raise ValueError("Unsupported type for Hurst estimation")
    return estimate_hurst(hss_path, process_type=process_type, H0=H, K=K)

# Test multiplate trials of a certan process in parallel (fast method)
# Prints average estimated H and mean squared error
# H_list consists of all H values to be tested
# Progress can be chosen to be printed or not
def hurst_tester(N=1024, *, H_list=[0.5], trials=1, process_type=DPW_FBM, speed="fast", progress=True, K=1, Lamperti_multiplier=5):
    workers = 1
    if speed == "fast" and trials > 1:
        workers = 12
        
    max_workers = min(workers, multiprocessing.cpu_count() - 2)
    H_estimates = []
    
    for H in H_list:
        H_estimates = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_trial, N, H, process_type, K, Lamperti_multiplier) for _ in range(trials)]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    estimate = future.result()
                    H_estimates.append(estimate)
                    if progress:
                        print(f"H={H} estimate {i}: {estimate}")
                except Exception as e:
                    print(f"Trial {i} failed: {e}")
        avg = np.mean(H_estimates)
        if len(H_estimates) == 0:
            print(f"No successful trials for H={H}")
            continue
        mse = sum((H * K - H_estimates[i])**2 for i in range(len(H_estimates))) / len(H_estimates)
        print(f"H: {H} K = {K} Average: {avg} MSE: {mse} trials: {trials} type: {process_type}")

# Testing of self-similarity estimation
if __name__ == "__main__":
    hurst_tester(N=128, H_list=[0.2,0.5,0.7,0.8], trials=1, process_type=WC_FBM, speed="fast", progress=False, K=1, Lamperti_multiplier=5)