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


def feature_count( xi):
	goal = np.array([.5,.4,.3])
	n, m = np.shape(xi)
	length_reward = 0
	coffee_score = -.5
	for idx in range(1, n):
		length_reward -= np.linalg.norm(xi[idx, :] - xi[idx-1, :])
		dist1= np.linalg.norm(xi[idx, 0:2] - goal[0:2])
		if dist1< .05:
			coffee_score = .5 	
	#proximity_score = np.linalg.norm(xi[idx, 0:2] - goal[0:2])**2 #if negative, good to go to goal, if positive go away.
	#proximity_score = np.linalg.norm([.1*(xi[idx, 0] - goal[0]) , 1*(xi[idx, 1] - goal[1]) ])**2
	proximity_score = abs( 1*(xi[-1, 1] - goal[1]) ) #- abs(.15*(xi[-1, 0] - goal[0]))

	f = np.array([ coffee_score,(proximity_score),np.exp(length_reward*(16/n))])
	#print(f)
	return f
	
def reward(f, theta):
	return theta[0]*f[0] + theta[1]*f[1] + theta[2]*f[2] 


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
		add= 0.3 * (np.random.random(3)-0.5)
		xi1[idx:, :] += add
		if xi1[idx,0] < 0.3:
			xi1[idx,0] = 0.3 + abs(add[0])	
		if xi1[idx,0] > 0.75:
			xi1[idx,0] = 0.75 - abs(add[0])	
		if xi1[idx,1] < -0.4:
			xi1[idx,1] = -0.4 + abs(add[1])	
		if xi1[idx,1] > 0.4:
			xi1[idx,1] = 0.4- abs(add[1])	
		if xi1[idx,2] < 0.1:
			xi1[idx,2] = 0.1 + abs(add[2])	
		if xi1[idx,2] > 0.45:
			xi1[idx,2] = 0.5 - abs(add[2])	
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
