# Statistical Inferences on Non-stationary Increments Self-similar Processes via Lamperti Transformations

Including

- Estimation of self-similarity index via Lamperti transformations with applications for
    - Fractional Brownian motion
    - Sub-fractional Brownian motion
    - Bifractioanl Brownian motion
    - Trifractional Brownian motion
- Estimation algorithm for the self-similarity index for when the scaling parameter is known and unknown
- Additional implementation for estimating the scaling parameter

## Estimation
Estimation of the self-similarity index of a given H-self-similar process. 

The main idea is to use a modified version of the Lamperti transformation to trasnform the self-similar process to a stationary process, and then exploit the properities of stationarity to estimate the value of H. Proof of convergence and error control along with the exact method can be found in our paper 

>W. Wu, Q. Peng "Statistical Inferences on Non-stationary Increments Self-similar Processes via Lamperti Transformations"

Given the estimation of the self-similarity index, the scaling parameter may also be estimated. 

## To Install
Use the code from the Github repository with 
```
git clone https://github.com/william11074/Detection-and-Estimation-of-Self-similarity
```
The estimation can be found with 
```
from estimation_of_self_similarity.hurst_estimation import hurst_estimation_method1, hurst_estimation_method2
```
The testing code is found in
```
from estimation_of_self_similarity.hurst_estimation_testing.py import hurst_tester
```

## Example Usage
To estimate the hurst parameter on a single path taken in as a list, run 
```
estimate_hurst_method1(hss_path, process_type=DPW_FBM)
```
Where hss_path is the sample path and process type dictates which self-similar process the estimation is used on. 

To run testing using simulated sample paths, run
```
hurst_tester(N=1024, *, H_list=[0.5], trials=1, process_type=DPW_FBM, speed="fast", progress=True, K=1, Lamperti_multiplier=5, sigma=1)
```
