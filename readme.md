# Statistical Inferences on Non-stationary Increments Self-similar Processes via Lamperti Transformations

Including

- Estimation of self-similarity index via Lamperti transformations with applications for
    - Fractional Brownian motion
    - Sub-fractional Brownian motion
    - Bi-fractioanl Brownian motion
    - Tri-fractional Brownian motion
- Testing of self-similarity on generated self-similar processes based on fractal-analysis package
- Detection of self-similarity with applications to 
    - Fractional Brownian motion

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

## Example Usage
To estimate the hurst parameter on a single path taken in as a list, run 
```
estimate_hurst_method1(hss_path, process_type=DPW_FBM)
```
Where hss_path is the sample path and process type dictates which self-similar process the estimation is used on. 
