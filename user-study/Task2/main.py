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
import matplotlib.patches as pat





def main():

	# get the user number
	parser = argparse.ArgumentParser()
	parser.add_argument('--user', type=int, default=0)
	parser.add_argument('--demo', type=int, default=0)
	parser.add_argument('--show', type=int, default=1)
	parser.add_argument('--cup', type=int, default=0)
	parser.add_argument('--traj', type=int, default=0)
	args = parser.parse_args()
	if not os.path.exists("data/user" + str(args.user)):
				os.makedirs("data/user" + str(args.user))
	savename = "data/user" + str(args.user) + "/demo" + str(args.demo) + ".pkl"
	thetaname = "data/user" + str(args.user) + "/theta" + str(args.demo) + ".pkl"

	
	# generate the ideal trajectory (?)
	x_start = np.array([0.35, 0, 0.5])
	x_goal = np.array([0.5, 0.1, 0.3])
	x_goal2 = np.array([0.5, 0.4, 0.3])
	t_start = 0.0
	t_end = 30.0	
	xi0 = np.linspace(x_start, x_goal, 8)
	if args.show == 0:
		traj = Trajectory(xi0, t_start, t_end)
	else:
		if (args.traj > 0	):
			#Re-run ideal trajectory if the user needs to see it again
			trajname = "data/user" + str(args.user) + "/traj" + str(args.demo) + ".pkl"
			traj = pickle.load(open(trajname, "rb"))
			thetaname = "data/user" + str(args.user) + "/theta" + str(args.demo) + ".pkl"
			theta = pickle.load(open(thetaname, "rb"))
			print(traj)
		else:
			# can choose random theta	
			theta = [.1,1,.6] #Run away from teh scary human
			#theta = [.8 , .8,.1] #Deliver and leave
			#theta = [.6,.1,1.5] #deliver coffee and wait for a pat on the head
			#theta = np.random.rand(3)
			#print(theta)
			theta = theta / np.linalg.norm(theta)
			thetaname = "data/user" + str(args.user) + "/theta" + str(args.demo) + ".pkl"
			print("Theta Value is :", theta)
			pickle.dump(theta, open(thetaname, "wb"))
			_, D_star = human_demo_d(xi0, theta, n_samples=200000)
			xi_star = D_star[-1] 	
			print(D_star)
			traj = Trajectory(xi_star, t_start, t_end)
			trajname = "data/user" + str(args.user) + "/traj" + str(args.demo) + ".pkl"
			pickle.dump(traj, open(trajname, "wb"))

			y =xi_star[:,0]
			x =xi_star[:,1]
			#print("Traj:",xi_star)
			fig, ax = plt.subplots() 
			plt.plot(x,y,'b')
			plt.plot(xi0[:,1],xi0[:,0],'g')
			plt.plot(x_start[1],x_start[0],'go')
			circle1 = plt.Circle((x_goal2[1],x_goal2[0]), 0.05, color='r')
			ax.add_patch(circle1	)
			plt.ylim([0.2,.8])
			plt.xlim([-.4,.5])
			plt.show()


	
	print("[*] Connecting to Robot")
	#connect to robot here

	full_state = readState(conn)
	xyz, R = joint2pose(full_state["q"])
	print("START",xyz)
	delta = np.array([0.]*3)
	print("[*] Starting Interaction")


	start_time = time.time()
	last_time = time.time()
	dataset = []
	while True:

		# measure state
		curr_time = time.time() - start_time
		full_state = readState(conn)
		xyz, R = joint2pose(full_state["q"])

		if time.time() - last_time > 0.1:
			dataset.append([curr_time] + list(delta))
			last_time = time.time()

		if curr_time > t_end:
			print(xyz)
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
		# safety checks
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
		if desired_xyz[2] > 0.5:
			desired_xyz[2] = 0.5

		# tell robot to follow xi
		x_error = desired_xyz - xyz
		#print(np.linalg.norm(x_error))
		x_error = [x_error[0], x_error[1], x_error[2], 0, 0, 0]
		q_error = xdot2qdot(x_error, full_state)	
		#send command to robot
		send2robot(conn, q_error, "v", limit=0.3)
	




main()