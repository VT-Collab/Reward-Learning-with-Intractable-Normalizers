import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils_algo import *


# total reward of a sequence of corrections
def get_D_reward(D, theta):
	Rtotal = 0.
	for xi in D:
		f = feature_count(xi)
		R = reward(f, theta)
		Rtotal = R + 0.9 * Rtotal
	return Rtotal

# find the best demo given theta
def search(xi0, theta, D, n_samples):
	best_demo = D
	best_reward = get_D_reward(D, theta)
	for _ in range(n_samples):
		D1 = rand_demo_d(xi0)
		R1 = get_D_reward(D1, theta)
		if R1 > best_reward:
			best_reward = R1
			best_demo = D1
	return best_demo

# thetas used in task
# for task 1 only three different thetas were used
t1 = np.array([0.76193932, 0., 0.64764842, 0.])
t2 = np.array([1., 0., 0., 0.])
t3 = np.array([0.74278135, 0.55708601, 0.37139068, 0.])

# initial trajectory
x_start = np.array([0.35, 0, 0.5])
x_goal = np.array([0.7, 0.3, 0.1])	
xi0 = np.linspace(x_start, x_goal, 8)

# get ideal demos for the true thetas
_, D1 = human_demo_d(xi0, t1, n_samples=10000)
_, D2 = human_demo_d(xi0, t2, n_samples=10000)
_, D3 = human_demo_d(xi0, t3, n_samples=10000)
Rstar1 = get_D_reward(D1, t1)
Rstar2 = get_D_reward(D2, t2)
Rstar3 = get_D_reward(D3, t3)


# convert data to array
data = pickle.load(open("error.pkl", "rb"))
datalist = []
for item in data:
	row = []
	for array in item:
		row.append(list(array))
	datalist.append(row)
data = np.array(data)


# compute the regret
regret = []
for count, item in enumerate(data):
	t = item[0]
	i, s, m, d = item[5], item[6], item[7], item[8]
	if np.linalg.norm(t - t1) < 1e-3:
		Di = search(xi0, i, D1, n_samples=100)
		Ds = search(xi0, s, D1, n_samples=100)
		Dm = search(xi0, m, D1, n_samples=100)
		Dd = search(xi0, d, D1, n_samples=100)
		Ri = get_D_reward(Di, t1)
		Rs = get_D_reward(Ds, t1)
		Rm = get_D_reward(Dm, t1)
		Rd = get_D_reward(Dd, t1)
		regret.append([Rstar1-Ri, Rstar1-Rs, Rstar1-Rm, Rstar1-Rd])
	elif np.linalg.norm(t - t2) < 1e-3:
		Di = search(xi0, i, D2, n_samples=100)
		Ds = search(xi0, s, D2, n_samples=100)
		Dm = search(xi0, m, D2, n_samples=100)
		Dd = search(xi0, d, D2, n_samples=100)
		Ri = get_D_reward(Di, t2)
		Rs = get_D_reward(Ds, t2)
		Rm = get_D_reward(Dm, t2)
		Rd = get_D_reward(Dd, t2)
		regret.append([Rstar2-Ri, Rstar2-Rs, Rstar2-Rm, Rstar2-Rd])
	elif np.linalg.norm(t - t3) < 1e-3:
		Di = search(xi0, i, D3, n_samples=100)
		Ds = search(xi0, s, D3, n_samples=100)
		Dm = search(xi0, m, D3, n_samples=100)
		Dd = search(xi0, d, D3, n_samples=100)
		Ri = get_D_reward(Di, t3)
		Rs = get_D_reward(Ds, t3)
		Rm = get_D_reward(Dm, t3)
		Rd = get_D_reward(Dd, t3)
		regret.append([Rstar3-Ri, Rstar3-Rs, Rstar3-Rm, Rstar3-Rd])
	print(count)
regret = np.array(regret)
pickle.dump(regret, open("regret.pkl", "wb"))