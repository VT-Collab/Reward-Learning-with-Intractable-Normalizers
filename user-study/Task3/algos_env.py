import numpy as np
import random


def p_D_theta(env, D, theta, beta=10.0):
	Rtotal = 0.
	for xi in D:
		f = env.feature_count(xi)
		R = env.reward(f, theta)
		Rtotal = R + 0.9 * Rtotal
	return np.exp(beta * Rtotal)



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



def human_demo_env(env, xi0, theta, n_samples):
	Ds = []
	P = np.array([0.] * n_samples)
	for idx in range(n_samples):
		D = rand_demo(xi0)	
		expR = p_D_theta(env, D, theta)
		P[idx] = expR
		Ds.append(D)
	P /= np.sum(P)
	sample_idx = np.random.choice(n_samples, p=P)
	best_idx = np.argmax(P)
	return Ds[sample_idx], Ds[best_idx]
