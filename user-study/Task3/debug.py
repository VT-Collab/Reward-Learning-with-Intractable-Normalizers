import numpy as np
from env_1b import Env1	
from algos_env import *
from algos_dependent import *

# set params
n_runs = 100
n_outer_samples = 50
n_inner_samples = 10
n_burn = 25

# initial trajectory
x_start = np.array([0.35, 0, 0.12])
x_block = np.array([0.6, 0.0, 0.12])
goal_pos = np.array([.45,-.25,.1])
xi0 = np.linspace(x_start, np.array([0.5, 0.0, 0.1]), 3)
env = Env1(block_position=x_block, goal_position=goal_pos, visualize=False)

# performance parameters
error_theta = [0, 0, 0, 0, 0, 0, 0, 0]

for run in range(n_runs):

	# random theta
	print("generating demonstration")
	theta = np.random.rand(2)
	_, D = human_demo_env(env, xi0, theta, n_samples=100)
		
	# naive
	print("naive - dependent")
	theta_naive_d = mcmc_naive_d(env,D, n_outer_samples, n_burn, len(theta))
	theta_naive_d = np.mean(theta_naive_d, axis=0)

	# mean
	print("mean - dependent")
	theta_mean_d = mcmc_mean_d(env,D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
	theta_mean_d = np.mean(theta_mean_d, axis=0)

	# max
	print("max - dependent")
	theta_max_d = mcmc_max_d(env,D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
	theta_max_d = np.mean(theta_max_d, axis=0)

	# double
	print("double - dependent")
	theta_double_d = mcmc_double_d(env,	D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
	theta_double_d = np.mean(theta_double_d, axis=0)

	# metric1 : error in theta
	error_theta[0] += np.linalg.norm(theta_naive_d - theta)
	error_theta[1] += np.linalg.norm(theta_mean_d - theta)
	error_theta[2] += np.linalg.norm(theta_max_d - theta)
	error_theta[3] += np.linalg.norm(theta_double_d - theta)

	# report the progress
	e1 = round(error_theta[0], 1)
	e2 = round(error_theta[1], 1)
	e3 = round(error_theta[2], 1)
	e4 = round(error_theta[3], 1)
	print("[*] iter", run, "error", [e1, e2, e3, e4])
