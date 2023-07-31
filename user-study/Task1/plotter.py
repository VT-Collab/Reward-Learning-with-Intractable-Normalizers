import numpy as np
import pickle
import matplotlib.pyplot as plt


# convert data to array
data = pickle.load(open("error.pkl", "rb"))
datalist = []
for item in data:
	row = []
	for array in item:
		row.append(list(array))
	datalist.append(row)
data = np.array(data)

# get error
error = np.zeros((len(data), 8))
for idx, item in enumerate(data):
	theta = item[0]
	for jdx in range(8):
		error[idx, jdx] = np.linalg.norm(theta - item[jdx+1])
mean = np.mean(error, axis=0)
sem = np.std(error, axis=0) / np.sqrt(10)

# plot result
x = range(8)
plt.bar(x, mean)
plt.errorbar(x, mean, sem)
plt.show()