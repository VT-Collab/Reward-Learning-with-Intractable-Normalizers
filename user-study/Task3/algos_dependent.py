import numpy as np
import random


def p_D_theta(env, D, theta, beta=10.0):
    Rtotal = 0.
    for xi in D:
        f = env.feature_count(xi)
        R = env.reward(f, theta)
        Rtotal = R + 0.9 * Rtotal
    return np.exp(beta * Rtotal)

    #Note: Though some CD reward equations rely on action magnitude, several iterations have proven the attribtution not helpful for our context given the
    #mix of small and large equally necesary actions found within the pilot versions of the User Study we choose to remove it as it did not fit our contextual application
    #and is not the highlight of the conditional dependent application.
    #if one wanted to implement that action magnitude into this equation, they would take hte difference between each trajectory and subtract a fraction of that
    #value from the total.
    #def p_D_theta(D, theta, beta=10.0):
    #ActMag = 0.
    #Rtotal = 0.
    #n,m = np.shape(D)
    # for i in range(n):
    #   xi = D[i]
	# 	f = feature_count(xi)
	# 	R = reward(f, theta)
	# 	Rtotal = R + 0.9 * Rtotal
    #   if i > 0:
    #       ActMag -= (xi - D[i-1])^2
    #return np.exp(beta*(Rtotal+ActMag))
    
def rand_demo(xi):
    D = [xi]
    n, m = np.shape(xi)
    xi1 = np.copy(xi)
    for idx in range(1, n):
        xi1[idx:, 0:2] += 0.5 * (np.random.random(2)-0.5)
        xi1[:,0] = np.clip(xi1[:,0], 0.3, 0.7)
        xi1[:,1] = np.clip(xi1[:,1], -0.4, 0.1)
        D.append(np.copy(xi1))
    return D[1:]


def mcmc_naive_d(env,D, n_outer_samples, n_burn, len_theta):
    theta = np.random.rand(len_theta)
    p_theta = p_D_theta(env,D, theta)
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        p_theta1 = p_D_theta(env,D, theta1)
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def mcmc_mean_d(env,D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    Z_theta = Z_mean_d(env,xi0, theta, n_inner_samples)
    p_theta = p_D_theta(env,D, theta) / Z_theta
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        Z_theta1 = Z_mean_d(env,xi0, theta1, n_inner_samples)
        p_theta1 = p_D_theta(env,D, theta1) / Z_theta1
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def mcmc_max_d(env,D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    Z_theta = Z_max_d(env,xi0, theta, n_inner_samples)
    p_theta = p_D_theta(env,D, theta) / Z_theta
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        Z_theta1 = Z_max_d(env,xi0, theta1, n_inner_samples)
        p_theta1 = p_D_theta(env,D, theta1) / Z_theta1
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def mcmc_double_d(env,D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    theta_samples = []
    y_prev = (False, None)
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        y = inner_sampler_d(env,D, xi0, theta1, n_inner_samples, y_prev)
        p_theta1 = p_D_theta(env,D, theta1) / p_D_theta(env,y, theta1)
        p_theta = p_D_theta(env,D, theta) / p_D_theta(env,y, theta)
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            y_prev = (True, y)
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def Z_mean_d(env,xi, theta, n_samples):
    mean_reward = 0.
    for _ in range(n_samples):
        D1 = rand_demo(xi)
        expR = p_D_theta(env,D1, theta)
        mean_reward += expR
    return mean_reward / n_samples


def Z_max_d(env,xi, theta, n_samples):
    max_reward = -np.inf
    for _ in range(n_samples):
        D1 = rand_demo(xi)
        expR = p_D_theta(env,D1, theta)        
        if expR > max_reward:
            max_reward = expR
    return max_reward


def inner_sampler_d(env,D, xi0, theta, n_samples, y_init=(False, None)):
    y = np.copy(D)
    y_score = p_D_theta(env,y, theta)        
    for _ in range(n_samples):
        y1 = rand_demo(xi0)
        y1_score = p_D_theta(env,y1, theta)
        if y1_score / y_score > np.random.rand():
            y = np.copy(y1)
            y_score = y1_score
    return y