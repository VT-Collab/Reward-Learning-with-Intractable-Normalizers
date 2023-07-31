import numpy as np
import random


# def feature_count(xi):
# 	n_waypoints, _ = np.shape(xi)
# 	obs1 = np.array([0.39, +0.16])
# 	obs2 = np.array([0.61, -0.17])
# 	goal = np.array([0.7, 0.3, 0.1])
# 	length = 0.
# 	obs1_reward = np.inf
# 	obs2_reward = np.inf
# 	goal_reward = -np.linalg.norm(goal - xi[-1, :])
# 	for idx in range(1, n_waypoints):
# 		length -= np.linalg.norm(xi[idx, :] - xi[idx-1, :])
# 		dist1 = np.linalg.norm(obs1 - xi[idx, :2])
# 		dist2 = np.linalg.norm(obs2 - xi[idx, :2])
# 		if dist1 < obs1_reward:
# 			obs1_reward = dist1
# 		if dist2 < obs2_reward:
# 			obs2_reward = dist2
# 	return np.array([goal_reward, obs1_reward, obs2_reward, length])


def feature_count(xi):
	n, m = np.shape(xi)
	length_reward = 0
	obs_reward = np.inf
	button_reward = -.25
	#start = np.array([0.35, 0, 0.5])
	goal = np.array([0.7, 0.3, 0.1])	
	obs = np.array([.5,.05])
	button = np.array([0.6, -.1, 0.15])
	for idx in range(1, n):
		length_reward -= np.linalg.norm(xi[idx, :] - xi[idx-1, :])**2
		dist1 = np.linalg.norm(obs - xi[idx, 0:2])
		dist2 = np.linalg.norm(xi[idx, :] - button) 
		if dist1 < obs_reward:
			obs_reward = dist1
		if dist2 < .05:
			button_reward = .25
	end_goal = .25 -np.linalg.norm(xi[idx, :] - goal)
	f = np.array([end_goal,button_reward,obs_reward*1.5,(length_reward)])
	return f

def reward(f, theta):
	return f[0] + theta[0]*f[1] + theta[1]*f[2] + theta[2]*f[3]


def p_xi_theta(xi, theta, beta=50.0):
    f = feature_count(xi)
    R = reward(f, theta)
    return np.exp(beta * R)

def rand_demo_i(xi):
	n, m = np.shape(xi)
	xi1 = np.copy(xi)
	for idx in range(1, n):
		xi1[idx:, :] += 0.2 * (np.random.random(3)-0.5)
	return xi1


def mcmc_naive_i(D, n_outer_samples, n_burn, len_theta):
	theta = np.random.rand(len_theta)
	p_theta = 1.0
	for xi in D:
		expR = p_xi_theta(xi, theta)
		p_theta *= expR
	theta_samples = []
	for _ in range(n_outer_samples):
		theta_samples.append(theta)
		theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
		theta1 = np.clip(theta1, 0, 1)
		p_theta1 = 1.0
		for xi in D:
			expR1 = p_xi_theta(xi, theta1)
			p_theta1 *= expR1
		if p_theta1 / p_theta > np.random.rand():
			theta = np.copy(theta1)
			p_theta = p_theta1
	theta_samples = np.array(theta_samples)
	return theta_samples[-n_burn:,:]


def mcmc_mean_i(D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
	theta = np.random.rand(len_theta)
	Z_theta = Z_mean_i(xi0, theta, n_inner_samples)
	p_theta = 1.0
	for xi in D:
		expR = p_xi_theta(xi, theta)
		p_theta *= expR / Z_theta
	theta_samples = []
	for _ in range(n_outer_samples):
		theta_samples.append(theta)
		theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
		theta1 = np.clip(theta1, 0, 1)
		Z_theta1 = Z_mean_i(xi0, theta1, n_inner_samples)
		p_theta1 = 1.0
		for xi in D:
			expR1 = p_xi_theta(xi, theta1)
			p_theta1 *= expR1 / Z_theta1
		if p_theta1 / p_theta > np.random.rand():
			theta = np.copy(theta1)
			p_theta = p_theta1
	theta_samples = np.array(theta_samples)
	return theta_samples[-n_burn:,:]


def mcmc_max_i(D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
	theta = np.random.rand(len_theta)
	Z_theta = Z_max_i(xi0, theta, n_inner_samples)
	p_theta = 1.0
	for xi in D:
		expR = p_xi_theta(xi, theta)
		p_theta *= expR / Z_theta
	theta_samples = []
	for _ in range(n_outer_samples):
		theta_samples.append(theta)
		theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
		theta1 = np.clip(theta1, 0, 1)
		Z_theta1 = Z_max_i(xi0, theta1, n_inner_samples)
		p_theta1 = 1.0
		for xi in D:
			expR1 = p_xi_theta(xi, theta1)
			p_theta1 *= expR1 / Z_theta1
		if p_theta1 / p_theta > np.random.rand():
			theta = np.copy(theta1)
			p_theta = p_theta1
	theta_samples = np.array(theta_samples)
	return theta_samples[-n_burn:,:]


def mcmc_double_i(D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
	theta = np.random.rand(len_theta)
	theta_samples = []
	y_prev = (False, None)
	for _ in range(n_outer_samples):
		theta_samples.append(theta)
		theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
		theta1 = np.clip(theta1, 0, 1)
		y = inner_sampler_i(D, xi0, theta1, n_inner_samples, y_prev)
		p_theta1 = 1.0
		for xi in D:
			expRx1 = p_xi_theta(xi, theta1)
			expRy1 = p_xi_theta(y, theta1)
			p_theta1 *= expRx1 / expRy1
		p_theta = 1.0
		for xi in D:
			expRx = p_xi_theta(xi, theta)
			expRy = p_xi_theta(y, theta)
			p_theta *= expRx / expRy
		if p_theta1 / p_theta > np.random.rand():
			theta = np.copy(theta1)
			y_prev = (True, y)
	theta_samples = np.array(theta_samples)
	return theta_samples[-n_burn:,:]


def Z_mean_i(xi, theta, n_samples):
	mean_reward = 0.
	for _ in range(n_samples):
		xi1 = rand_demo_i(xi)
		expR = p_xi_theta(xi1, theta)
		mean_reward += expR
	return mean_reward / n_samples


def Z_max_i(xi, theta, n_samples):
	max_reward = -np.inf
	for _ in range(n_samples):
		xi1 = rand_demo_i(xi)
		expR = p_xi_theta(xi1, theta)        
		if expR > max_reward:
			max_reward = expR
	return max_reward


def inner_sampler_i(D, xi0, theta, n_samples, y_init=(False, None)):
	if y_init[0]:
		y = y_init[1]
	else:
		y = random.choice(D)
	y_score = p_xi_theta(y, theta)        
	for _ in range(n_samples):
		y1 = rand_demo_i(y)
		y1_score = p_xi_theta(y1, theta)
		if y1_score / y_score > np.random.rand():
			y = np.copy(y1)
			y_score = y1_score
	return y