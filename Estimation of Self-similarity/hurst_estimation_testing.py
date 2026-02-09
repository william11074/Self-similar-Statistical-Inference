"""
hurst_estimation.py

Used for the estimation of the self-similarity index of a general H-self-similar process via the Lamperti transformation
This module also includes simple benchmarking helpers (serial and parallel)

Notes
- Method based on our paper "Statistical Inferences on Non-stationary Self-similar Processes via Lamperti Transformations" by W. Wu, Q. Peng 
- Simulators are imported from the `fractal_analysis` package (Ding, Peng, Wu).
"""

# estimate hurst values using Halley's method
from hurst_estimation import estimate_hurst_method1, estimate_hurst_method2

# self-similar processes simulation
# imported from fractal-analaysis package by Ding, Peng, Wu
from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwFbmSimulator as DPW_FBM
from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwSubFbmSimulator as DPW_SFBM
from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwBiFbmSimulator as DPW_BFBM
from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwTriFbmSimulator as DPW_TFBM
from fractal_analysis.simulator.wood_chan.wood_chan_fractal_simulator import WoodChanFbmSimulator as WC_FBM
from fractal_analysis.estimator.hurst_estimator import QvHurstEstimator

# Other imports
import numpy as np
import matplotlib.pyplot as plt

# for parallel processing, opens multiple cores if available
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Estimate H for a single simulated path given path length, self-similarity index H, process type, K (bi factor or tri factor) and Lamperti multiplier
def single_trial(N=1024, H=0.5, process_type=DPW_FBM, K=1, Lamperti_multiplier=5, sigma=1):
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
    
    if sigma != 1:
        return estimate_hurst_method2(hss_path, process_type=process_type)
    return estimate_hurst_method1(hss_path, process_type=process_type, H0=H, K=K, sigma=sigma)

# For showcasing sigma estimation
def remove_outliers_iqr(data, k=1.5):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return data[(data >= lower) & (data <= upper)]

# Test multiplate trials of a certan process in parallel (fast method)
# Prints average estimated H and mean squared error
# H_list consists of all H values to be tested
# Progress can be chosen to be printed or not
def hurst_tester(N=1024, *, H_list=[0.5], trials=1, process_type=DPW_FBM, speed="fast", progress=True, K=1, Lamperti_multiplier=5, sigma=1):
    workers = 1
    if speed == "fast" and trials > 1:
        workers = 12
        
    max_workers = min(workers, multiprocessing.cpu_count() - 2)
    H_estimates = []
    
    for H in H_list:
        H_estimates = []
        sigma2_estimates = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_trial, N, H, process_type, K, Lamperti_multiplier, sigma) for _ in range(trials)]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    estimate_H = future.result()[0] if type(future.result()) is tuple else future.result()
                    H_estimates.append(estimate_H)
                    sigma2_estimates.append(future.result()[1] if type(future.result()) is tuple else None)
                    if progress:
                        print(f"H={H} estimate {i}: {estimate_H}")
                except Exception as e:
                    print(f"Trial {i} failed: {e}")
        avg = np.mean(H_estimates)
        if len(H_estimates) == 0:
            print(f"No successful trials for H={H}")
            continue
        mse = sum((H * K - H_estimates[i])**2 for i in range(len(H_estimates))) / len(H_estimates)
        print(f"H: {H} K = {K} Average: {avg} MSE: {mse} trials: {trials} type: {process_type}")
        # Displays the estimated sigma values in a histogram
        if sigma != 1:
            try:
                sigma2_estimates = remove_outliers_iqr(sigma2_estimates, k=1.5)
                print(f"Sigma^2 Average: {np.mean(sigma2_estimates)}")
                plt.hist(sigma2_estimates, bins=20, density=True)
                plt.xlabel(f"Estimated Sigma Squared. Median = {np.median(sigma2_estimates):.4f}")
                plt.ylabel("Frequency")
                plt.show()
            except Exception as e:
                print(f"Error in plotting sigma estimates: {e}")

# Example testing of self-similarity parameter estimation
if __name__ == "__main__":
    hurst_tester(N=128, H_list=[0.2,0.5,0.7,0.8], trials=1, process_type=DPW_SFBM, speed="fast", progress=False, K=1, Lamperti_multiplier=5)
    hurst_tester(N=128, H_list=[0.2], trials=1, process_type=DPW_BFBM, speed="fast", progress=False, K=0.8, Lamperti_multiplier=5, sigma=2)