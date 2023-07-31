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

from essentials  import *
from utils import *
from panda import *


PORT = 8080
print("[*] Connecting to Robot")
conn = connect2robot(PORT)
# generate the ideal trajectory (?)
full_state = readState(conn)
start_pos, R = joint2pose(full_state["q"])
print(start_pos)

