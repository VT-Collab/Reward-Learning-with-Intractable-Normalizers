from algos_dependent import *
from algos_independent import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils_algo import *
from env_1b import Env1	
import time
from algos_env import human_demo_env

def main():

	n_runs = 10*3*10
	n_outer_samples = 50
	n_inner_samples = 10
	n_burn = 25
	Beta = 50

	# trajectory parameters
	x_start = np.array([0.35, 0, 0.12])
	x_block = np.array([0.6, 0.0, 0.12])
	goal_pos =[.45,-.25,.1]
	xi0 = np.linspace(x_start, np.array([0.5, 0.0, 0.12]), 3)
	env = Env1(block_position=x_block,goal_position = goal_pos,visualize=False)

	# performance parameters
	error_theta = [0, 0, 0, 0, 0, 0, 0, 0]
	datasave = []

	# main loop
	for run in range(n_runs):

		choice = run % (10 * 3)
		user = int(np.floor(choice / 3.0))
		demo = choice - user * 3
		user += 1
		demo += 1
		demoname = "data/user" + str(user) + "/demo" + str(demo) + ".pkl"
		thetaname = "data/user" + str(user) + "/theta" + str(demo) + ".pkl"
		D = process_demo(demoname, xi0)
		theta = pickle.load(open(thetaname, "rb"))
		print("[*] run #: ", run, "loaded user: ", user, "demo: ", demo)

		# naive
		print("naive - independent")
		theta_naive_i = mcmc_naive_i(env, D, n_outer_samples, n_burn, len(theta))
		theta_naive_i = np.mean(theta_naive_i, axis=0)

		# mean
		print("mean - independent")
		theta_mean_i = mcmc_mean_i(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
		theta_mean_i = np.mean(theta_mean_i, axis=0)

		# max
		print("max - independent")
		theta_max_i = mcmc_max_i(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
		theta_max_i = np.mean(theta_max_i, axis=0)

		# double
		print("double - independent")
		theta_double_i = mcmc_double_i(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
		theta_double_i = np.mean(theta_double_i, axis=0)


		# naive
		print("naive - dependent")
		theta_naive_d = mcmc_naive_d(env, D, n_outer_samples, n_burn, len(theta))
		theta_naive_d = np.mean(theta_naive_d, axis=0)

		# mean
		print("mean - dependent")
		theta_mean_d = mcmc_mean_d(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
		theta_mean_d = np.mean(theta_mean_d, axis=0)

		# max
		print("max - dependent")
		theta_max_d = mcmc_max_d(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
		theta_max_d = np.mean(theta_max_d, axis=0)

		# double
		print("double - dependent")
		theta_double_d = mcmc_double_d(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
		theta_double_d = np.mean(theta_double_d, axis=0)

		# metric1 : error in theta
		error_theta[0] += np.linalg.norm(theta_naive_i - theta)/len(theta)
		error_theta[1] += np.linalg.norm(theta_mean_i - theta)/len(theta)
		error_theta[2] += np.linalg.norm(theta_max_i - theta)/len(theta)
		error_theta[3] += np.linalg.norm(theta_double_i - theta)/len(theta)

		error_theta[4] += np.linalg.norm(theta_naive_d - theta)/len(theta)
		error_theta[5] += np.linalg.norm(theta_mean_d - theta)/len(theta)
		error_theta[6] += np.linalg.norm(theta_max_d - theta)/len(theta)
		error_theta[7] += np.linalg.norm(theta_double_d - theta)/len(theta)	

		# report the progress
		e1 = round(error_theta[0], 2)
		e2 = round(error_theta[1], 2)
		e3 = round(error_theta[2], 2)
		e4 = round(error_theta[3], 2)

		e5 = round(error_theta[4], 2)
		e6 = round(error_theta[5], 2)
		e7 = round(error_theta[6], 2)
		e8 = round(error_theta[7], 2)

		print("[*] iter", run, "error", [e1, e2, e3, e4, e5, e6, e7, e8])
	
		# save the progress
		datasave.append([theta, theta_naive_i, theta_mean_i, theta_max_i, theta_double_i,\
						theta_naive_d, theta_mean_d, theta_max_d, theta_double_d])

		pickle.dump(datasave, open("error.pkl", "wb"))

main()