import numpy as np
import pickle
import random
import cv2	
from imutils.video import VideoStream
import time 

def process_demo(name, xi0):
	n_waypoints, _ = np.shape(xi0)
	data = pickle.load(open(name, "rb"))
	data = np.array(data)[:,1:]
	waypoints = np.linspace(0, len(data)-1, n_waypoints)
	data1 = []
	delta_prev = np.array([0.]*3)
	for t in waypoints:
		if t > 0:
			delta = data[round(t), :]
			data1.append(list(delta - delta_prev))
			delta_prev = np.copy(delta)
	data1 = np.array(data1)
	corrections = [xi0]
	for t, delta in enumerate(data1):
		xi = np.copy(corrections[-1])
		xi[t+1:,:] += delta
		corrections.append(xi)
	return corrections[1:]

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

		x /= 440*.68
		y /= 340*2
		# print(c1, r1)
		#print("target_x={}, target_y={}".format(x,y))
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