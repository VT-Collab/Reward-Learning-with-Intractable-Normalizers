import numpy as np
import random


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


def p_D_theta(D, theta, beta=50.0):
	Rtotal = 0
	for xi in D:
		f = feature_count(xi)
		R = reward(f, theta)
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

def rand_demo_d(xi):
	n, m = np.shape(xi)
	D = [xi]
	for idx in range(1, n):
		xi1 = np.copy(D[-1])
		xi1[idx:, :] += 0.2 * (np.random.random(3)-0.5)
		D.append(xi1)
	return D[1:]

def rand_demo_i(xi):
	n, m = np.shape(xi)
	xi1 = np.copy(xi)
	for idx in range(1, n):
		xi1[idx:, :] += 0.2 * (np.random.random(3)-0.5)
	return xi1






def mcmc_mean_d(D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    Z_theta = Z_mean_d(xi0, theta, n_inner_samples)
    p_theta = p_D_theta(D, theta) / Z_theta
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        Z_theta1 = Z_mean_d(xi0, theta1, n_inner_samples)
        p_theta1 = p_D_theta(D, theta1) / Z_theta1
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def mcmc_max_d(D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    Z_theta = Z_max_d(xi0, theta, n_inner_samples)
    p_theta = p_D_theta(D, theta) / Z_theta
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        Z_theta1 = Z_max_d(xi0, theta1, n_inner_samples)
        p_theta1 = p_D_theta(D, theta1) / Z_theta1
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]

def mcmc_naive_d(D, n_outer_samples, n_burn, len_theta):
    theta = np.random.rand(len_theta)
    p_theta = p_D_theta(D, theta)
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        p_theta1 = p_D_theta(D, theta1)
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]

def mcmc_double_d(D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    theta_samples = []
    y_prev = (False, None)
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        y = inner_sampler_d(D, xi0, theta1, n_inner_samples, y_prev)
        p_theta1 = p_D_theta(D, theta1) / p_D_theta(y, theta1)
        p_theta = p_D_theta(D, theta) / p_D_theta(y, theta)
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            y_prev = (True, y)
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def Z_mean_d(xi, theta, n_samples):
    mean_reward = 0.
    for _ in range(n_samples):
        D1 = rand_demo_d(xi)
        expR = p_D_theta(D1, theta)
        mean_reward += expR
    return mean_reward / n_samples


def Z_max_d(xi, theta, n_samples):
    max_reward = -np.inf
    for _ in range(n_samples):
        D1 = rand_demo_d(xi)
        expR = p_D_theta(D1, theta)        
        if expR > max_reward:
            max_reward = expR
    return max_reward


def inner_sampler_d(D, xi0, theta, n_samples, y_init=(False, None)):
    y = np.copy(D)
    y_score = p_D_theta(y, theta)        
    for _ in range(n_samples):
        y1 = rand_demo_d(xi0)
        y1_score = p_D_theta(y1, theta)
        if y1_score / y_score > np.random.rand():
            y = np.copy(y1)
            y_score = y1_score
    return y