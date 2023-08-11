import numpy as np
from utils import *
import time
#each algorith is given the ideal human action using the outer loop to take a Monte Carlo Sampling method to approximate theta 
#from the results of that theta belief with the ideal human action taking the inner loop to create a normalizer for sample estimation

#Naive method is one such that it ignores the presence of any given normalizer simply using a ratio comparison of the results instead.
def mcmc_naive(state,a_h,  n_outer_samples, n_burn):
    
    #Generate random theta
    t = np.random.rand()*np.pi/2
    theta = np.array([np.cos(t), np.sin(t)]) #Running Baseline theta
    #test theta reward for ideal human action
    p_theta = reward(state,a_h,  theta)
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        t1 = t + 0.5*(np.random.rand()*2-1) #redo random sample with scaled adjustment
        t1 = np.clip(t1, 0, np.pi/2)
        theta1 = np.array([np.cos(t1), np.sin(t1)]) #Comparison theta
        p_theta1 = reward(state,a_h,  theta1) #reward comparison
        if p_theta1 / p_theta > np.random.rand(): #change sample theta dependent on MC sample progression
            t = t1
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]

#Mean approach uses a normalizer made from a inner sampled mean result for a given theta.
def mcmc_mean(state,a_h,  n_outer_samples, n_inner_samples, n_burn):
    #Generate random theta
    
    t = np.random.rand()*np.pi/2
    theta = np.array([np.cos(t), np.sin(t)])
    
    Z_theta = Z_mean(state,theta, n_inner_samples) #normalizer found by mean reward found in inner sampler
   
    #test theta reward for ideal human action
    p_theta = reward(state,a_h, theta) / Z_theta 
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        t1 = t + 0.5*(np.random.rand()*2-1) 
        t1 = np.clip(t1, 0, np.pi/2)
        theta1 = np.array([np.cos(t1), np.sin(t1)])

        Z_theta1 = Z_mean(state,theta1, n_inner_samples)

        p_theta1 = reward(state,a_h, theta1) / Z_theta1
        if p_theta1 / p_theta > np.random.rand():
            #new running baseline
            t = t1
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    
    return theta_samples[-n_burn:,:]

#Max approach uses a normalizer made from a inner sampled max result for a given theta.
def mcmc_max(state,a_h, n_outer_samples, n_inner_samples, n_burn):
    #Generate random theta
    
    t = np.random.rand()*np.pi/2
    theta = np.array([np.cos(t), np.sin(t)])
    
    Z_theta = Z_max(state,theta, n_inner_samples) #normalizer found by max reward found in inner sampler
    
    #test theta reward for ideal human action
    p_theta = reward(state,a_h, theta) / Z_theta
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        t1 = t + 0.5*(np.random.rand()*2-1) #
        t1 = np.clip(t1, 0, np.pi/2)
        theta1 = np.array([np.cos(t1), np.sin(t1)])

        Z_theta1 = Z_max(state,theta1, n_inner_samples)

        p_theta1 = reward(state,a_h, theta1) / Z_theta1
        if p_theta1 / p_theta > np.random.rand(): 
            t = t1
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    
    return theta_samples[-n_burn:,:]

#This DMH approach uses a normalizer found an internally sampled action to test the variance for interal actions generated by an interal MC loop in a similar fashion to the naive appraoch
def mcmc_double(state,a_h, n_outer_samples, n_inner_samples, n_burn):
    #Generate random theta
    t = np.random.rand()*np.pi/2
    theta = np.array([np.cos(t), np.sin(t)])
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        t1 = t + 0.5*(np.random.rand()*2-1) #redo random sample with scaled adjustment
        t1 = np.clip(t1, 0, np.pi/2)
        theta1 = np.array([np.cos(t1), np.sin(t1)])

        y = inner_sampler(state,theta1, n_inner_samples) #find new action to create DMH normalizer
        
        x_theta = reward(state,a_h, theta)
        x_theta1 = reward(state,a_h, theta1)
        #reward for sampled inner action to compare based off ideal inner sample
        y_theta = reward(state,y, theta) 
        y_theta1 = reward(state,y, theta1)
        ratio = (x_theta1 * y_theta) / (x_theta * y_theta1)

        if ratio > np.random.rand():
            t = t1
            theta = np.copy(theta1)
    theta_samples = np.array(theta_samples)
    
    return theta_samples[-n_burn:,:] 

#Mean approach normalizer
def Z_mean(state,theta, n_samples):
    mean_reward = 0.
    for _ in range(n_samples):
        a = rand_action()
        mean_reward += reward(state,a, theta)
    return mean_reward / n_samples

#Max approach normalizer
def Z_max(state,theta, n_samples):
    max_reward = -np.inf
    for _ in range(n_samples):
        a = rand_action()
        r = reward(state,a, theta)
        if r > max_reward:
            max_reward = r
    return max_reward

#Inner sampler for DMH normalizer approach
def inner_sampler(state,theta, n_samples):
    a = rand_action()
    a_score = reward(state,a, theta) #baseline action and score for theta 
    for _ in range(n_samples):
        a1 = rand_action() #redo random sample with new random action for random theta sample
        a1_score = reward(state,a1, theta)
        if a1_score / a_score > np.random.rand():
            #update theoretical action
            a = a1
            a_score = a1_score
    return a