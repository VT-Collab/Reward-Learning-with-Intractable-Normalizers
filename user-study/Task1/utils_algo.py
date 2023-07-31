import numpy as np
import pickle
import random


def process_demo(name, xi0):
	n_waypoints, _ = np.shape(xi0)
	data = pickle.load(open(name, "rb"))
	data = np.array(data)[:,1:]
	waypoints = np.linspace(0, len(data)-1, n_waypoints)
	data1 = []
	delta_prev = np.array([0.]*3)
	for t in waypoints:
		if t > 0:
			delta = data[round(t), :]
			data1.append(list(delta - delta_prev))
			delta_prev = np.copy(delta)
	data1 = np.array(data1)
	corrections = [xi0]
	for t, delta in enumerate(data1):
		xi = np.copy(corrections[-1])
		xi[t+1:,:] += delta
		corrections.append(xi)
	return corrections[1:]


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
	return theta[0]*f[0] + theta[1]*f[1] + theta[2]*f[2] + theta[3]*f[3]


def p_D_theta(D, theta, beta=50.0):
	Rtotal = 0.
	for xi in D:
		f = feature_count(xi)
		R = reward(f, theta)
		Rtotal = R + 0.9 * Rtotal
	return np.exp(beta * Rtotal)


def rand_demo_d(xi):
	n, m = np.shape(xi)
	D = [xi]
	for idx in range(1, n):
		xi1 = np.copy(D[-1])
		xi1[idx:, :] += 0.2 * (np.random.random(3)-0.5)
		D.append(xi1)
	return D[1:]


def human_demo_d(xi, theta, n_samples):
	Ds = []
	P = np.array([0.] * n_samples)
	for idx in range(n_samples):
		D = rand_demo_d(xi)
		
		expR = p_D_theta(D, theta)
		P[idx] = expR
		Ds.append(D)
	P /= np.sum(P)
	sample_idx = np.random.choice(n_samples, p=P)
	best_idx = np.argmax(P)
	return Ds[sample_idx], Ds[best_idx]
