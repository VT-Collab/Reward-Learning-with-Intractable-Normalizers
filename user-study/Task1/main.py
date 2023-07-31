import numpy as np
from utils import * #Panda specific commands for robot operation
from utils_algo import *
import pickle
import time
import datetime
import argparse
import pygame
import os

import matplotlib.pyplot as plt





def main():

	# get the user number
	parser = argparse.ArgumentParser()
	parser.add_argument('--user', type=int, default=0)
	parser.add_argument('--demo', type=int, default=0)
	parser.add_argument('--show', type=int, default=1)
	args = parser.parse_args()
	if not os.path.exists("data/user" + str(args.user)):
				os.makedirs("data/user" + str(args.user))
	savename = "data/user" + str(args.user) + "/demo" + str(args.demo) + ".pkl"
	thetaname = "data/user" + str(args.user) + "/theta" + str(args.demo) + ".pkl"


	# generate the ideal trajectory (?)
	x_start = np.array([0.35, 0, 0.5])
	x_goal = np.array([0.7, 0.3, 0.1])	
	obs = np.array([.5,.05])
	button = np.array([0.6, -.1, 0.15])
	xi0 = np.linspace(x_start, x_goal, 8)
	
	t_start = 0.0
	t_end = 20.0	
	
	if args.show == 0:
		traj = Trajectory(xi0, t_start, t_end)
	else:
		# can choose random theta	
		theta = np.random.rand(4)
		##Preset defined human beliefs
		#theta = [1,0,.85,0	] #Go to goal, avoid the column
		#theta = [1,0,0,0] #Go to goal
		#theta = [1,.75,.5,0]#Go to button, avoid obstacle. Take as long as you want
		
		theta = theta / np.linalg.norm(theta)
		print("Theta Value is :", theta)
		pickle.dump(theta, open(thetaname, "wb"))
		_, D_star = human_demo_d(xi0, theta, n_samples=100000)
		xi_star = D_star[-1] 	
		traj = Trajectory(xi_star, t_start, t_end)
		print("Feat:",(feature_count(xi_star)))
		print("FeatBase:",(feature_count(xi0)))
		x =xi_star[:,0]
		y =xi_star[:,1]

		#Visual Representation of Generated Version of Task
		fig, ax = plt.subplots() 
		plt.plot(x,y,'b')
		plt.plot(x_goal[0],x_goal[1],'rx')
		plt.plot(x_start[0],x_start[1],'go')
		circle1 = plt.Circle((button[0],button[1]), 0.05, color='b')
		ax.add_patch(circle1	)
		circle2 = plt.Circle((obs[0],obs[1]), 0.01, color='r')
		ax.add_patch(circle2	)
		plt.xlim([0.2,.8])
		plt.ylim([-.2,.4])
		plt.show()

	# print(xi_star)
	#obs1 = np.array([0.39, +0.16])
	#obs2 = np.array([0.61, -0.17])
	# plt.plot(xi_star[:,0], xi_star[:,1], 'bo-')
	# plt.plot(obs1[0], obs1[1], 'ko')
	# plt.plot(obs2[0], obs2[1], 'ko')
	# plt.axis("equal")
	# plt.show()



	# # joystick setup
	# joystick = Joystick()
		# panda setup



	#CONNECTING TO ROBOT CODE HERE



	delta = np.array([0.]*3)
	print("[*] Starting Interaction")


	start_time = time.time()
	last_time = time.time()
	dataset = []
	while True:

		# measure state
		curr_time = time.time() - start_time
		full_state = #Read State Code here
		xyz, R = joint2pose(full_state["q"]) #transfer from joint angle values to position and Rotation values

		if time.time() - last_time > 0.1:
			dataset.append([curr_time] + list(delta))
			last_time = time.time()

		if curr_time > t_end:
			pickle.dump(dataset, open(savename, "wb"))
			return True
		
		# # get joystick input
		# press_A, press_B = joystick.input()
		# if press_A:
		# 	break

		# is there a correction?
		correction = False
		applied_force = np.array([0., 0., 0.])
		if np.linalg.norm(full_state["tau"]) > 5.0:
			correction = True
			F_wrench = full_state["O_F"]
			if abs(F_wrench[0]) > 10.0:
				applied_force[0] = -np.sign(F_wrench[0]) * 1.0
			if abs(F_wrench[1]) > 10.0:
				applied_force[1] = -np.sign(F_wrench[1]) * 1.0
			if abs(F_wrench[2]) > 10.0:
				applied_force[2] = -np.sign(F_wrench[2]) * 1.0


		# if so, modify trajectory xi
		if correction and curr_time < t_end:
			delta += 0.001 * applied_force

		# get desired position
		desired_xyz = traj.get_waypoint(curr_time) + delta
		# safety checks for joint limits
		if desired_xyz[0] < 0.3:
			desired_xyz[0] = 0.3
		if desired_xyz[0] > 0.75:
			desired_xyz[0] = 0.75
		if desired_xyz[1] < -0.4:
			desired_xyz[1] = -0.4
		if desired_xyz[1] > 0.4:
			desired_xyz[1] = 0.4
		if desired_xyz[2] < 0.1:
			desired_xyz[2] = 0.1
		if desired_xyz[2] > 0.6:
			desired_xyz[2] = 0.6

		# tell robot to follow xi
		x_error = desired_xyz - xyz
		#print(np.linalg.norm(x_error))
		x_error = [x_error[0], x_error[1], x_error[2], 0, 0, 0]
		q_error = xdot2qdot(x_error, full_state)
		#Sending command for robot movement
		#send2robot(conn, q_error, "v", limit=0.3)




main()