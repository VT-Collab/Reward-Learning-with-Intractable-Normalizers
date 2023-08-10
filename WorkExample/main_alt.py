import numpy as np
import argparse
from utils import *
from algos import *
import time


def main():

	# args parse for run parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--runs', type=int, default=1000)
	parser.add_argument('--outer', type=int, default=1000)
	parser.add_argument('--inner', type=int, default=50)
	args = parser.parse_args()
	# run parameters
	n_runs = args.runs #Number of total runs
	n_outer_samples = args.outer #Number of outer samples to generate random samples of comparison for theta
	n_inner_samples = args.inner #Number of inner samples to generate relevant normalizers
	n_burn = np.int8(round(n_outer_samples/2,0)) #Number of burned samples from initial sampling
   
	#Time comparisons
	temp_naive = 0
	temp_mean = 0
	temp_max = 0
	temp_dmh = 0
	tempcomp = 0
	# performance parameters
	error_theta = [0, 0, 0, 0]
	total_reward = [0, 0, 0, 0]
	time_total = [0,0,0,0]
	inner_time_total = [0,0,0,0]
	# main loop
	for run in range(n_runs):

		# get expert human correction
		t = np.random.rand() * np.pi/2 #number from 0 to pi/2
		theta = np.array([np.cos(t), np.sin(t)]) # varies randomly from 0 to 1 dependent on whether t is leaning the sampler toward a right or left bias
		a_h, _ = human_action(theta, n_samples=1000) #theoretical ideal human action given a belief of theta

		# naive
		temp_naive = time.time()
		theta_naive = mcmc_naive(a_h, n_outer_samples, n_burn)
		tempcomp = time.time()
		Time_naive = abs(temp_naive - tempcomp)
		theta_naive = np.mean(theta_naive, axis=0) #average value from MH sampling
	
		# mean
		temp_mean = time.time()
		theta_mean,inner_time_mean = mcmc_mean(a_h, n_outer_samples, n_inner_samples, n_burn)
		tempcomp = time.time()
		Time_mean = abs(temp_mean - tempcomp)
		theta_mean = np.mean(theta_mean, axis=0)
	
		# max
		temp_max = time.time()
		theta_max,inner_time_max  = mcmc_max(a_h, n_outer_samples, n_inner_samples, n_burn)
		tempcomp = time.time()
		Time_max = abs(temp_max - tempcomp)
		theta_max = np.mean(theta_max, axis=0)
	
		# double
		temp_dmh = time.time()
		theta_double,inner_time_dmh = mcmc_double(a_h, n_outer_samples, n_inner_samples, n_burn)
		#print("ALABASTER",inner_time_dmh)
		tempcomp = time.time()
		Time_dmh = abs(temp_dmh - tempcomp)
		theta_double = np.mean(theta_double, axis=0)

		# metric1 : error in theta
		error_theta[0] += np.linalg.norm(theta_naive - theta)
		error_theta[1] += np.linalg.norm(theta_mean - theta)
		error_theta[2] += np.linalg.norm(theta_max - theta)
		error_theta[3] += np.linalg.norm(theta_double - theta)

		# # metric2 : regret
		# xi_naive, _ = human_corrections(xi, theta_naive, n_iter=10000)
		# xi_mean, _ = human_corrections(xi, theta_mean, n_iter=10000)
		# xi_max, _ = human_corrections(xi, theta_max, n_iter=10000)
		# total_reward[0] += reward(xi_naive, theta)
		# total_reward[1] += reward(xi_mean, theta)
		# total_reward[2] += reward(xi_max, theta)
		time_total = [time_total[0] + Time_naive, time_total[1] + Time_mean, time_total[2] + Time_max, time_total[3] + Time_dmh]
		inner_time_total = [0, inner_time_total[1] + inner_time_mean, inner_time_total[2] + inner_time_max, inner_time_total[3] + inner_time_dmh]
		# report the progress
		e1 = round(error_theta[0], 1)
		e2 = round(error_theta[1], 1)
		e3 = round(error_theta[2], 1)
		e4 = round(error_theta[3], 1)
		print("[*] iter", run, "error", [e1, e2, e3, e4],"time",[Time_naive,Time_mean,Time_max,Time_dmh])
	time2 = []
	for i in time_total:
		time2.append(i/n_runs)
	time3 = []
	for j in inner_time_total:
		time3.append(j/n_runs)
	print("END TIME TOTAL:",time_total)
	print("END TIME TOTAL:",inner_time_total)
	print("END TIME AVG:",time2)
	print("END INNER TIME AVG:",time3)
main()
