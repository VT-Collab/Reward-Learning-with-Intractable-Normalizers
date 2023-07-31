import numpy as np
import random


def p_xi_theta(env,xi, theta, beta=10.0):
		f = env.feature_count(xi)
		R = env.reward(f, theta)
		return np.exp(beta * R)


def rand_demo(xi):
    D = [xi]
    n, m = np.shape(xi)
    xi1 = np.copy(xi)
    for idx in range(1, n):
        xi1[idx:, 0:2] += 0.5 * (np.random.random(2)-0.5)
        xi1[:,0] = np.clip(xi1[:,0], 0.3, 0.7)
        xi1[:,1] = np.clip(xi1[:,1], -0.4, 0.1)
        D.append(np.copy(xi1))
    return D[-1]


def mcmc_naive_i(env,D, n_outer_samples, n_burn, len_theta):
	theta = np.random.rand(len_theta)
	p_theta = 1.0
	for xi in D:
		expR = p_xi_theta(env,xi, theta)
		p_theta *= expR
	theta_samples = []
	for _ in range(n_outer_samples):
		theta_samples.append(theta)
		theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
		theta1 = np.clip(theta1, 0, 1)
		p_theta1 = 1.0
		for xi in D:
			expR1 = p_xi_theta(env,xi, theta1)
			p_theta1 *= expR1
		if p_theta1 / p_theta > np.random.rand():
			theta = np.copy(theta1)
			p_theta = p_theta1
	theta_samples = np.array(theta_samples)
	return theta_samples[-n_burn:,:]


def mcmc_mean_i(env,D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
	theta = np.random.rand(len_theta)
	Z_theta = Z_mean_i(env,xi0, theta, n_inner_samples)
	p_theta = 1.0
	for xi in D:
		expR = p_xi_theta(env,xi, theta)
		p_theta *= expR / Z_theta
	theta_samples = []
	for _ in range(n_outer_samples):
		theta_samples.append(theta)
		theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
		theta1 = np.clip(theta1, 0, 1)
		Z_theta1 = Z_mean_i(env,xi0, theta1, n_inner_samples)
		p_theta1 = 1.0
		for xi in D:
			expR1 = p_xi_theta(env,xi, theta1)
			p_theta1 *= expR1 / Z_theta1
		if p_theta1 / p_theta > np.random.rand():
			theta = np.copy(theta1)
			p_theta = p_theta1
	theta_samples = np.array(theta_samples)
	return theta_samples[-n_burn:,:]


def mcmc_max_i(env,D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
	theta = np.random.rand(len_theta)
	
	Z_theta = Z_max_i(env,xi0, theta, n_inner_samples)
	p_theta = 1.0
	for xi in D:
		expR = p_xi_theta(env,xi, theta)
		p_theta *= expR / Z_theta
	theta_samples = []
	for _ in range(n_outer_samples):
		theta_samples.append(theta)
		theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
		theta1 = np.clip(theta1, 0, 1)
		Z_theta1 = Z_max_i(env,xi0, theta1, n_inner_samples)
		p_theta1 = 1.0
		for xi in D:
			expR1 = p_xi_theta(env,xi, theta1)
			p_theta1 *= expR1 / Z_theta1
		if p_theta1 / p_theta > np.random.rand():
			theta = np.copy(theta1)
			p_theta = p_theta1
	theta_samples = np.array(theta_samples)
	return theta_samples[-n_burn:,:]


def mcmc_double_i(env,D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
	theta = np.random.rand(len_theta)
	theta_samples = []
	y_prev = (False, None)
	for _ in range(n_outer_samples):
		theta_samples.append(theta)
		theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
		theta1 = np.clip(theta1, 0, 1)
		y = inner_sampler_i(env,D, xi0, theta1, n_inner_samples, y_prev)
		p_theta1 = 1.0
		for xi in D:
			expRx1 = p_xi_theta(env,xi, theta1)
			expRy1 = p_xi_theta(env,y, theta1)
			p_theta1 *= expRx1 / expRy1
		p_theta = 1.0
		for xi in D:
			expRx = p_xi_theta(env,xi, theta)
			expRy = p_xi_theta(env,y, theta)
			p_theta *= expRx / expRy
		if p_theta1 / p_theta > np.random.rand():
			theta = np.copy(theta1)
			y_prev = (True, y)
	theta_samples = np.array(theta_samples)
	return theta_samples[-n_burn:,:]


def Z_mean_i(env,xi, theta, n_samples):
	mean_reward = 0.
	for _ in range(n_samples):
		xi1 = rand_demo(xi)
		expR = p_xi_theta(env,xi1, theta)
		mean_reward += expR
	return mean_reward / n_samples


def Z_max_i(env,xi, theta, n_samples):
	max_reward = -np.inf
	
	for _ in range(n_samples):
		xi1 = rand_demo(xi)
		expR = p_xi_theta(env,xi1, theta)        
		if expR > max_reward:
			max_reward = expR
	return max_reward


def inner_sampler_i(env,D, xi0, theta, n_samples, y_init=(False, None)):
	if y_init[0]:
		y = y_init[1]
	else:
		y = random.choice(D)
	y_score = p_xi_theta(env,y, theta)        
	for _ in range(n_samples):
		y1 = rand_demo(y)
		y1_score = p_xi_theta(env,y1, theta)
		if y1_score / y_score > np.random.rand():
			y = np.copy(y1)
			y_score = y1_score
	return y