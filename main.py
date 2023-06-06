import numpy as np
from utils import *
from algos import *


def main():

    # run parameters

    n_runs = 1000 #Number of total runs
    n_outer_samples = 1000 #Number of outer samples to generate random samples of comparison for theta
    n_inner_samples = 50 #Number of inner samples to generate relevant normalizers
    n_burn = 500 #Number of burned samples from initial sampling
   

    # performance parameters
    error_theta = [0, 0, 0, 0]
    total_reward = [0, 0, 0, 0]
    # main loop
    for run in range(n_runs):

        # get expert human correction
        t = np.random.rand() * np.pi/2 #number from 0 to pi/2
        theta = np.array([np.cos(t), np.sin(t)]) # varies randomly from 0 to 1 dependent on whether t is leaning the sampler toward a right or left bias
        a_h, _ = human_action(theta, n_samples=1000) #theoretical ideal human action given a belief of theta

        # naive
        theta_naive = mcmc_naive(a_h, n_outer_samples, n_burn)
        theta_naive = np.mean(theta_naive, axis=0) #average value from MH sampling
    
        # mean
        theta_mean = mcmc_mean(a_h, n_outer_samples, n_inner_samples, n_burn)
        theta_mean = np.mean(theta_mean, axis=0)
    
        # max
        theta_max = mcmc_max(a_h, n_outer_samples, n_inner_samples, n_burn)
        theta_max = np.mean(theta_max, axis=0)
    
        # double
        theta_double = mcmc_double(a_h, n_outer_samples, n_inner_samples, n_burn)
        theta_double = np.mean(theta_double, axis=0)

        # metric1 : error in theta
        error_theta[0] += np.linalg.norm(theta_naive - theta)
        error_theta[1] += np.linalg.norm(theta_mean - theta)
        error_theta[2] += np.linalg.norm(theta_max - theta)
        error_theta[3] += np.linalg.norm(theta_double - theta)

        # # metric2 : regret
        # xi_naive, _ = human_corrections(xi, theta_naive, n_iter=10000)
        # xi_mean, _ = human_corrections(xi, theta_mean, n_iter=10000)
        # xi_max, _ = human_corrections(xi, theta_max, n_iter=10000)
        # total_reward[0] += reward(xi_naive, theta)
        # total_reward[1] += reward(xi_mean, theta)
        # total_reward[2] += reward(xi_max, theta)
        
        # report the progress
        e1 = round(error_theta[0], 1)
        e2 = round(error_theta[1], 1)
        e3 = round(error_theta[2], 1)
        e4 = round(error_theta[3], 1)
        print("[*] iter", run, "error", [e1, e2, e3, e4])
main()
