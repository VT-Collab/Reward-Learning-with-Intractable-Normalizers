import numpy as np
import cv2
from imutils.video import VideoStream
import time 
import pickle
import socket
import sys
from scipy.interpolate import interp1d
import pygame
import torch
import copy
from torch.optim import Adam
from torch.nn.utils.convert_parameters import parameters_to_vector
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize


"""Home Position for Panda for all tasks"""
HOME = [0.8385, -0.0609, 0.2447, -1.5657, 0.0089, 1.5335, 1.8607]

class Joystick(object):

	def __init__(self):
		pygame.init()
		self.gamepad = pygame.joystick.Joystick(0)
		self.gamepad.init()
		self.deadband = 0.1
		self.timeband = 0.5
		self.lastpress = time.time()

	def input(self):
		pygame.event.get()
		curr_time = time.time()
		A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
		B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
		X_pressed = self.gamepad.get_button(2) and (curr_time - self.lastpress > self.timeband)
		START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
		if A_pressed or START_pressed or B_pressed:
			self.lastpress = curr_time
		return A_pressed, B_pressed, X_pressed, START_pressed


class Trajectory(object):

	def __init__(self, xi, T):
		""" create cublic interpolators between waypoints """
		self.xi = np.asarray(xi)
		self.T = T
		self.n_waypoints = xi.shape[0]
		timesteps = np.linspace(0, self.T, self.n_waypoints)
		self.f1 = interp1d(timesteps, self.xi[:,0], kind='cubic')
		self.f2 = interp1d(timesteps, self.xi[:,1], kind='cubic')
		self.f3 = interp1d(timesteps, self.xi[:,2], kind='cubic')
		self.f4 = interp1d(timesteps, self.xi[:,3], kind='cubic')
		self.f5 = interp1d(timesteps, self.xi[:,4], kind='cubic')
		self.f6 = interp1d(timesteps, self.xi[:,5], kind='cubic')
		# self.f7 = interp1d(timesteps, self.xi[:,6], kind='cubic')

	def get(self, t):
		""" get interpolated position """
		if t < 0:
			q = [self.f1(0), self.f2(0), self.f3(0), self.f4(0), self.f5(0), self.f6(0)]
		elif t < self.T:
			q = [self.f1(t), self.f2(t), self.f3(t), self.f4(t), self.f5(t), self.f6(t)]
		else:
			q = [self.f1(self.T), self.f2(self.T), self.f3(self.T), self.f4(self.T), self.f5(self.T), self.f6(self.T)]
		return np.asarray(q)


"""Obtain the location of target"""
def get_target():
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

	frame = vs.read()
	frame = cv2.flip(frame, 1)

	minx = 50
	maxx = 480-minx
	miny = 50
	maxy =640 -  miny
	roi = (minx, miny, maxx, maxy)
	clone = frame.copy()
	image = clone[int(roi[1]):int(roi[1] + roi [3]), \
				int(roi[0]):int(roi[0] + roi[2])]

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# hsv = image

	# upper_blue = np.array([140, 140 , 245])
	# lower_blue = np.array([100, 100 , 200])
	upper = np.array([160, 210 , 250])
	lower = np.array([100, 150 , 190])
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	#rgb =image
	mask = cv2.inRange(rgb, lower, upper)
	kernal = np.ones ((15, 15), "uint8")
	red = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)    
	result = cv2.bitwise_and(image, image, mask=mask) 
	(contoursred, hierarchy) =cv2.findContours (red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for pic, contourred in enumerate (contoursred):
		area = cv2.contourArea (contourred) 
		if (area > 10):
			x, y, w, h = cv2.boundingRect (contourred)
			img = cv2.rectangle (hsv, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.putText(img,"MARKER",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))

	if len(contoursred) > 0:
		# Find the biggest contour
		biggest_contour = max(contoursred, key=cv2.contourArea)

		# Find center of contour and draw filled circle
		moments = cv2.moments(biggest_contour)
		centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
		cv2.circle(img, centre_of_contour, 2, (0, 0, 255), -1)
		# Save the center of contour so we draw line tracking it
		center_points1 = centre_of_contour
		r1 = center_points1[0]
		c1 = center_points1[1]
		y = r1-150
		x = 425-c1

		x /= 440*.502
		y /= 340*2
		# print(c1, r1)
		print("target_x={}, target_y={}".format(x,y))
	else:
		x = 0
		y = 0
		print("FAILURE")
	# cv2.imshow('hsv', hsv)
	# cv2.imshow('image', image)
	# cv2.imshow('mask', red)
	# cv2.waitKey(0)
	vs.stop()
	return x, y


"""Connecting and Sending commands to robot"""
def connect2robot(PORT):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.bind(('172.16.0.3', PORT))
	s.listen()
	conn, addr = s.accept()
	return conn

def send2robot(conn, qdot, mode, traj_name=None, limit=0.5):
	if traj_name is not None:
		if traj_name[0] == 'q':
			# print("limit increased")
			limit = 1.0
	qdot = np.asarray(qdot)
	scale = np.linalg.norm(qdot)
	if scale > limit:
		qdot *= limit/scale
	send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
	send_msg = "s," + send_msg + "," + mode + ","
	conn.send(send_msg.encode())

def send2gripper(conn):
	send_msg = "o"
	conn.send(send_msg.encode())

def listen2robot(conn):
	state_length = 7 + 7 + 7 + 42
	message = str(conn.recv(2048))[2:-2]
	state_str = list(message.split(","))
	for idx in range(len(state_str)):
		if state_str[idx] == "s":
			state_str = state_str[idx+1:idx+1+state_length]
			break
	try:
		state_vector = [float(item) for item in state_str]
	except ValueError:
		return None
	if len(state_vector) is not state_length:
		return None
	state_vector = np.asarray(state_vector)
	state = {}
	state["q"] = state_vector[0:7]
	state["dq"] = state_vector[7:14]
	state["tau"] = state_vector[14:21]
	state["J"] = state_vector[21:].reshape((7,6)).T

	# get cartesian pose
	xyz_lin, R = joint2pose(state_vector[0:7])
	beta = -np.arcsin(R[2,0])
	alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
	gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
	xyz_ang = [alpha, beta, gamma]
	xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
	state["x"] = np.array(xyz)
	return state

def readState(conn):
	while True:
		state = listen2robot(conn)
		if state is not None:
			break
	return state

def xdot2qdot(xdot, state):
	J_pinv = np.linalg.pinv(state["J"])
	return J_pinv @ np.asarray(xdot)

def joint2pose(q):
	def RotX(q):
		return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
	def RotZ(q):
		return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	def TransX(q, x, y, z):
		return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
	def TransZ(q, x, y, z):
		return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
	H1 = TransZ(q[0], 0, 0, 0.333)
	H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
	H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
	H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
	H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
	H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
	H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
	H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
	H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
	return H[:,3][:3], H[:,:3][:3]


def go2home(conn, h=None):
	if h is None:
		home = np.copy(HOME)
	else:
		home = np.copy(h)
	total_time = 35.0;
	start_time = time.time()
	state = readState(conn)
	current_state = np.asarray(state["q"].tolist())

	# Determine distance between current location and home
	dist = np.linalg.norm(current_state - home)
	curr_time = time.time()
	action_time = time.time()
	elapsed_time = curr_time - start_time

	# If distance is over threshold then find traj home
	while dist > 0.02 and elapsed_time < total_time:
		current_state = np.asarray(state["q"].tolist())

		action_interval = curr_time - action_time
		if action_interval > 0.005:
			# Get human action
			qdot = home - current_state
			# qdot = np.clip(qdot, -0.3, 0.3)
			send2robot(conn, qdot, "v")
			action_time = time.time()

		state = readState(conn)
		dist = np.linalg.norm(current_state - home)
		curr_time = time.time()
		elapsed_time = curr_time - start_time

	# Send completion status
	if dist <= 0.02:
		return True
	elif elapsed_time >= total_time:
		return False

def wrap_angles(theta):
	if theta < -np.pi:
		theta += 2*np.pi
	elif theta > np.pi:
		theta -= 2*np.pi
	else:
		theta = theta
	return theta


"""Play the trajectory on the robot for Soweing Queries/ Provide corrections"""
def play_traj(conn, args, traj_name, algo, iter_count=None):
	total_time = 45.0

	traj = pickle.load(open(traj_name, "rb" ))
	if args.task == 'cup':
		traj[:, 0] = np.clip(traj[:, 0], 0.35, 0.71)
		traj[:, 1] = np.clip(traj[:, 1], -0.4, 0.6)
		traj[:, 2] = np.clip(traj[:, 2], 0.25, 0.6)
	else:
		traj[:, 0] = np.clip(traj[:, 0], 0.2, 0.71)
		traj[:, 1] = np.clip(traj[:, 1], -0.4, 0.6)
		traj[:, 2] = np.clip(traj[:, 2], 0.1, 0.6)
	traj = Trajectory(traj[:, :6], total_time)
	# traj[:, 2] = np.clip(traj[:, 2], 0.1, 0.6)

	print('[*] Connecting to low-level controller...')
	print("RETURNING HOME")
	interface = Joystick()
	go2home(conn)
	print("PRESS START WHEN READY")

	curr_t = 0.0
	start_t = None
	play_traj = False
	dropped = False
	state = readState(conn)
	corrections = []
	C = []
	record = False
	steptime = 0.1
	
	scale = 1.0
	mode = "v"
	while True:

		state = readState(conn)
		A, B, stop, start = interface.input()

		if start and not play_traj:
			# go2home(conn)
			play_traj = True
			start_t = time.time()

		if stop and record:
			record = False
			print("Are you satisfied with the demonstration?")
			print("Enter [yes] to proceed any ANY KEY to scrap it")
			ans = input()
			if ans == 'yes':
				for idx in range (len(corrections)):
					corrections[idx] = corrections[idx] + corrections[-1]
				C.append(corrections)
			
			print("[*] I recorded this many datapoints: ", len(corrections))
			print("Please Release the E-Stop")
			corrections = []
			time.sleep(5)

			if algo == 'ours':
				go2home(conn, stop_point)
			else: 
				go2home(conn)
			
			print("Do you wish to provide another correction?")
			ans = input()
			if ans == 'y':
				record = True
			else:
				if iter_count is None:
					filename = "corrections/" + algo + "/run_" + args.run_name + "/correction.pkl"				
				else:
					filename = "corrections/" + algo + "/run_" + args.run_name + "/corr_" + str(iter_count) + ".pkl"
				print("I have this many corrections:", len(C))
				if algo == 'ours':
					pickle.dump(C[0], open(filename, 'wb'))
				else:
					pickle.dump(C, open(filename, 'wb'))
				break
			

		# if stop and not record:
		# 	if iter_count is None:
		# 		filename = "corrections/" + algo + "/run_" + args.run_name + "/correction.pkl"				
		# 	else:
		# 		filename = "corrections/" + algo + "/run_" + args.run_name + "/corr_" + str(iter_count) + ".pkl"
		# 	print("I have this many corrections:", len(C))
		# 	pickle.dump(C, open(filename, 'wb'))
		# 	break


		if A:
			scale = 0.0
			mode = "k"
			print("Changing to Kinesthetic Control Mode")
			print("please press E-STOP")
			print("Do you wish to provide a Correction?")
			stop_point = state['q']
			if algo == 'ours':
				go2home(conn, stop_point)
			else: 
				go2home(conn)
			line = input()
			if line == 'y':
				record = True
				start_time = time.time()
				print("Recording the correction")
			else:
				break
				
		elif B:
			mode = "v"
			scale = 1.0
			print("Changing to Veclocity Control Mode")

		if play_traj:
			

			curr_t = time.time() - start_t
			x_des = traj.get(curr_t)
			x_curr = state['x']

			# x_des[0] = np.clip(x_des[0], 0.0, 0.76)
			# x_des[1] = np.clip(x_des[0], -0.55, 0.65)
			# x_des[2] = np.clip(x_des[0], 0.08, 0.7)

			# if np.linalg.norm(x_des[:3])>0.76:
			# 	x_des[2] = x_curr[2]

			

			x_des[3] = wrap_angles(x_des[3])
			x_des[4] = wrap_angles(x_des[4])
			x_des[5] = wrap_angles(x_des[5])
			xdot = 1*scale * (x_des - x_curr)
			xdot[3] = wrap_angles(xdot[3])
			xdot[4] = wrap_angles(xdot[4])
			xdot[5] = wrap_angles(xdot[5])
			# print(x_des)
			if x_curr[0] <= 0.15 or x_curr[0] >= 0.7:
				xdot[0] = 0
			if x_curr[1] <= -0.4 or x_curr[1] >= 0.65:
				xdot[1] = 0
			if x_curr[2] <= 0.08 or x_curr[2] >= 0.6:
				xdot[2] = 0
			qdot = xdot2qdot(xdot, state)
			q_curr = state['q']
			if (q_curr[6] > 2.7 and qdot[6] > 0) or (q_curr[6] < -2.7 and qdot[6] < 0):
				qdot[6] = 0
			if (q_curr[3] > -0.1 and qdot[3] > 0) or (q_curr[3] < -2.7 and qdot[3] < 0):
				qdot[3] = 0
			if (q_curr[5] > 3.6 and qdot[5] > 0) or (q_curr[5] < 0.1 and qdot[5] < 0):
				qdot[5] = 0
			send2robot(conn, qdot, mode, traj_name)

		curr_time = time.time()
		if record and curr_time - start_time >= steptime:
			corrections.append(state["x"].tolist())
			start_time = curr_time

"""Play the final trajectory of a method and record it"""
def final_traj(conn, args, traj_name, algo, iter_count=None):
	total_time = 45.0

	traj = pickle.load(open(traj_name, "rb" ))
	if algo=='demo':
		traj = np.array(traj)[0]
	if args.task == 'cup':
		traj[:, 0] = np.clip(traj[:, 0], 0.45, 0.71)
		traj[:, 1] = np.clip(traj[:, 1], -0.4, 0.6)
		traj[:, 2] = np.clip(traj[:, 2], 0.25, 0.6)
	else:
		traj[:, 0] = np.clip(traj[:, 0], 0.2, 0.71)
		traj[:, 1] = np.clip(traj[:, 1], -0.4, 0.6)
		traj[:, 2] = np.clip(traj[:, 2], 0.1, 0.6)
	
	traj = Trajectory(traj[:, :6], total_time)
	

	print('[*] Connecting to low-level controller...')
	print("RETURNING HOME")
	interface = Joystick()
	# go2home(conn)
	print("PRESS START WHEN READY")

	curr_t = 0.0
	start_t = None
	play_traj = False
	dropped = False
	state = readState(conn)
	final_traj = []
	record = False
	steptime = 0.1
	
	scale = 1.0
	mode = "v"
	while True:

		state = readState(conn)
		A, B, stop, start = interface.input()

		if start and not play_traj:
			go2home(conn)
			play_traj = True
			record = True
			print("Recording the final trajectory")
			start_t = time.time()
			start_time = time.time()

		if stop and record:
			filename = "final_trajs/" + algo + "/run_" + args.run_name + ".pkl"
			for idx in range (len(final_traj)):
				final_traj[idx] = final_traj[idx] + final_traj[-1]

			pickle.dump(final_traj, open( filename, "wb"))
			print("[*] Done!")
			print("[*] I recorded this many datapoints: ", len(final_traj))
			break


		if A:
			scale = 0.0
			mode = "k"
			line = input()
			if line == 'y':
				record = True
			else:
				break
				
		elif B:
			mode = "v"
			scale = 1.0
			print("Changing to Veclocity Control Mode")

		if play_traj:

			curr_t = time.time() - start_t
			# for idx in range (len(traj)-1):
			# 	while np.linalg.norm(traj[idx+1, :6] - traj[idx, :6]) > 0.02:
			# 		xdot = traj[idx+1, :6] - traj[idx, :6]
			x_des = traj.get(curr_t)
			x_curr = state['x']

			# x_des[0] = np.clip(x_des[0], 0.0, 0.76)
			# x_des[1] = np.clip(x_des[0], -0.55, 0.65)
			# x_des[2] = np.clip(x_des[0], 0.08, 0.7)

			# if np.linalg.norm(x_des[:3])>0.76:
			# 	x_des[2] = x_curr[2]
			

			x_des[3] = wrap_angles(x_des[3])
			x_des[4] = wrap_angles(x_des[4])
			x_des[5] = wrap_angles(x_des[5])
			xdot = 1*scale * (x_des - x_curr)
			xdot[3] = wrap_angles(xdot[3])
			xdot[4] = wrap_angles(xdot[4])
			xdot[5] = wrap_angles(xdot[5])
			# print("XDES=",x_des)
			# print("XCUR=", x_curr)
			# if x_curr[0] <= 0.15 or x_curr[0] >= 0.7:
			# 	xdot[0] = 0
			# if x_curr[1] <= -0.4 or x_curr[1] >= 0.65:
			# 	xdot[1] = 0
			# if x_curr[2] <= 0.08 or x_curr[2] >= 0.6:
			# 	xdot[2] = 0
			qdot = xdot2qdot(xdot, state)
			q_curr = state['q']
			if (q_curr[6] > 2.7 and qdot[6] > 0) or (q_curr[6] < -2.7 and qdot[6] < 0):
				qdot[6] = 0
			if (q_curr[3] > -0.1 and qdot[3] > 0) or (q_curr[3] < -2.7 and qdot[3] < 0):
				qdot[3] = 0
			if (q_curr[5] > 3.6 and qdot[5] > 0) or (q_curr[5] < 0.1 and qdot[5] < 0):
				qdot[5] = 0
			send2robot(conn, qdot, mode)

		curr_time = time.time()
		if record and curr_time - start_time >= steptime:
			final_traj.append(state["x"].tolist())
			start_time = curr_time


"""Collect Physical Human Demonstrations"""
def collect_demos(conn, args):

	print("RETURNING HOME")
	interface = Joystick()
	# go2home(conn)
	print("PRESS START WHEN READY")

	state = readState(conn)
	print(state['x'])
	qdot = [0.0]*7
	demonstration = []
	record = False
	steptime = 0.1
	XI = []
	scale = 1.0
	mode = "k"
	while True:

		state = readState(conn)
		# print(state['x'])
		
		A, B, stop, start = interface.input()

		if A:
			record = False
			print("Are you satisfied with the demonstration?")
			print("Enter [yes] to proceed any ANY KEY to scrap it")
			ans = input()
			if ans == 'yes':
				for idx in range (len(demonstration)):
					demonstration[idx] = demonstration[idx] + demonstration[-1]
				XI.append(demonstration)
				print("[*] Done!")
				print("[*] I recorded this many datapoints: ", len(demonstration))
			demonstration = []
			print("Please release the E-Stop")
			time.sleep(5)
			go2home(conn)
			print("Press START for another demonstration or X to save the dataset")
		
		
		if stop:
			pickle.dump(XI, open('../demos/run_' + args.run_name + '/demo_expert.pkl', "wb"))

			demos = pickle.load(open('../demos/run_' + args.run_name + '/demo_expert.pkl', "rb"))
			XI = []
			print(len(demos))
			for idx in range(len(demos)): 
				demo = []
				demo1 = np.array(demos[idx])
				for d_idx in range (len(demo1)-1):
					if np.linalg.norm(demo1[d_idx,:6] - demo1[d_idx+1, :6]) > 0.001:
						demo.append(demo1[d_idx,:])
						
				XI.append(demo)
				print(len(XI))

			pickle.dump(XI, open('../demos/run_' + args.run_name + '/demo_train.pkl', "wb"))
			return True


		if start and not record:
			record = True
			start_time = time.time()
			print('[*] Recording the demonstration...')

		curr_time = time.time()
		if record and curr_time - start_time >= steptime:
			demonstration.append(state["x"].tolist())
			start_time = curr_time

		send2robot(conn, qdot, mode)


# get_target()