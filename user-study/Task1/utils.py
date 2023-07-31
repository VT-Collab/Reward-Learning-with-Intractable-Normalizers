import numpy as np
import time
import pickle
import socket
import sys
import pygame
import serial
from scipy.interpolate import interp1d


HOME = [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]


class Trajectory():

    def __init__(self, xi, t_start, t_end):
        self.t_start = t_start
        self.t_end = t_end
        self.n, self.m = np.shape(xi)
        self.traj = []
        xi = np.asarray(xi)
        timesteps = np.linspace(self.t_start, self.t_end, self.n)
        for idx in range(self.m):
            self.traj.append(interp1d(timesteps, xi[:, idx], kind='linear'))
        
    def get_waypoint(self, t):
        if t < self.t_start:
            t = self.t_start
        if t > self.t_end:
            t = self.t_end
        waypoint = np.array([0.] * self.m)
        for idx in range(self.m):
            waypoint[idx] = self.traj[idx](t)
        return waypoint


class Joystick(object):

	def __init__(self):
		self.gamepad = pygame.joystick.Joystick(0)		
		self.gamepad.init()
		self.timeband = 1.0
		self.lastpress = time.time()

	def input(self):
		pygame.event.get()
		curr_time = time.time()
		A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
		B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
		if A_pressed or B_pressed:
			self.lastpress = curr_time
		return A_pressed, B_pressed


def connect2robot(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('172.16.0.3', PORT))
    s.listen()
    conn, addr = s.accept()
    return conn


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


def listen2robot(conn):
	state_length = 7 + 7 + 7 + 6 + 42
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
	state["O_F"] = state_vector[21:27]
	state["J"] = state_vector[27:].reshape((7,6)).T
	return state


def readState(conn):
	while True:
		state = listen2robot(conn)
		if state is not None:
			break
	return state


def go2home(conn, home=HOME):
	target = np.asarray(home)
	state = readState(conn)
	error = target - state["q"]
	while np.linalg.norm(error) > 0.02:
		send2robot(conn, error, "v", limit=0.3)
		state = readState(conn)
		error = target - state["q"]


def send2robot(conn, qdot, mode, limit=0.3):
	qdot = np.asarray(qdot)
	scale = np.linalg.norm(qdot)
	if scale > limit:
		qdot *= limit/scale
	send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
	send_msg = "s," + send_msg + ","
	conn.send(send_msg.encode())


def xdot2qdot(xdot, state):
	J_pinv = np.linalg.pinv(state["J"])
	return J_pinv @ np.asarray(xdot)