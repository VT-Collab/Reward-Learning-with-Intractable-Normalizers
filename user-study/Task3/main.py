import numpy as np
from utils import *
from utils_algo import *
import pickle
import time
import datetime
import argparse
import pygame
import os
from algos_env import human_demo_env
import matplotlib.pyplot as plt
from env_1b import Env1





def main():

	# get the user number
	parser = argparse.ArgumentParser()
	parser.add_argument('--user', type=int, default=0)
	parser.add_argument('--demo', type=int, default=0)
	parser.add_argument('--show', type=int, default=1)
	parser.add_argument('--traj', type=int, default=0)
	args = parser.parse_args()
	if not os.path.exists("data/user" + str(args.user)):
				os.makedirs("data/user" + str(args.user))
	savename = "data/user" + str(args.user) + "/demo" + str(args.demo) + ".pkl"
	thetaname = "data/user" + str(args.user) + "/theta" + str(args.demo) + ".pkl"
	

	print("[*] Connecting to Robot")

	#Read robot state
	full_state = readState(conn)
	start_pos, R = joint2pose(full_state["q"])


	x,y = get_target() #Get where real world object is
	print("BOX IS AT:", x,y)
	x = x# push slight further back than boxes current center
	y2 = start_pos[1]
	box_push = [x,y,start_pos[2]]
	#goal_position =[start_pos[0],start_pos[1]-.25,start_pos[2]]
	goal_pos =[.45,-.25,.1]#goal
	box_pos =[x-.1,y,.1] #close to box
	#xi0 = np.linspace(start_pos, box_push, 4)
	xi0 = np.linspace(start_pos, box_pos, 3)
	
	env = Env1(block_position=[x,y,.12],goal_position = goal_pos,visualize=False) #robot enviroment for processing object interaction

	t_start = 0.0
	t_end = 20.0
			
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
			
			theta = [1.0,1.0] 
			#theta = [0.9,0.1] 
			#theta = np.random.rand(2)
			#print(theta)
			theta = theta / np.linalg.norm(theta)
			thetaname = "data/user" + str(args.user) + "/theta" + str(args.demo) + ".pkl"
			print("Theta Value is :", theta)
			pickle.dump(theta, open(thetaname, "wb"))
			_, D_star = human_demo_env(env,xi0, theta, n_samples=1000)
			xi_star = D_star[-1] 	
			print(D_star)
			traj = Trajectory(xi_star, t_start, t_end)
			trajname = "data/user" + str(args.user) + "/traj" + str(args.demo) + ".pkl"
			pickle.dump(traj, open(trajname, "wb"))

			print("Feat:",(env.feature_count(xi_star)))
			xplt =xi_star[:,0]
			yplt =xi_star[:,1]	
			xplt2 =xi0[:,0]
			yplt2 =xi0[:,1]	
			#print("Traj	:",xi_star)
			fig, ax = plt.subplots() 
			plt.plot(xplt,yplt,'b')
			plt.plot(xplt2,yplt2,'r')
			plt.plot(start_pos[0],start_pos[1],'b+')
			plt.plot(goal_pos[0],goal_pos[1],'rx')
			plt.plot(box_push[0],box_push[1],'go')
			plt.plot(box_pos[0],box_pos[1],'go')
			plt.show()
	


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
		if desired_xyz[2] > 0.6:
			desired_xyz[2] = 0.6

		# tell robot to follow xi
		x_error = desired_xyz - xyz
		#print(np.linalg.norm(x_error))
		x_error = [x_error[0], x_error[1], x_error[2], 0, 0, 0]
		q_error = xdot2qdot(x_error, full_state)
		send2robot(conn, q_error, "v", limit=0.3)




main()