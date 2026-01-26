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

>W. Wu, Q. Peng "Detection and Estimation of Self-similarity via Lamperti Transformations"

## Detection
Detection of self-similarity for a given sample path. 

The main idea relies on a modified version of the method given in  this paper

>Michał Balcerek, Krzysztof Burnecki. (2020)  
Testing of fractional Brownian motion in a noisy environment.  
Chaos, Solitons & Fractals, Volume 140, 110097.  
https://doi.org/10.1016/j.chaos.2020.110097

## To Install
Use the code from the Github repository with 
```
git clone https://github.com/william11074/Detection-and-Estimation-of-Self-similarity
```
The estimation can be found with 
```
from Detection-and-Estimation-of-Self-similarity import esimtate_hurst, hurst_tester
```

## Example Usage
To estimate the hurst parameter on a single path taken in as a list, run 
```
estimate_hurst(hss_path, process_type=DPW_FBM)
```
Where hss_path is the sample path and process type dictates which self-similar process the estimation is used on. 
