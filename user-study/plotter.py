import numpy as np
import pickle
import matplotlib.pyplot as plt


def data2array(name):
	data = pickle.load(open(name, "rb"))

	datalist = []
	for item in data:
		row = []
		for array in item:
			row.append(list(array))
		datalist.append(row)
	return np.array(data)

def get_error(data):
	error = np.zeros((len(data), 8))
	for idx, item in enumerate(data):
		theta = item[0]
		for jdx in range(8):
			error[idx, jdx] = np.linalg.norm(theta - item[jdx+1])
	return list(error)

data1 = data2array("error1.pkl")
data2 = data2array("error2.pkl")
data3 = data2array("error3.pkl")

error1 = get_error(data1)
error2 = get_error(data2)
error3 = get_error(data3)

error = error1 + error2 + error3
error = np.array(error)
np.savetxt("error.csv", error, delimiter=",")

# confirm all data is here
print(np.shape(error))

# get metrics
mean = np.mean(error, axis=0)
sem = np.std(error, axis=0) / np.sqrt(30)

# plot result
x = range(8)
plt.bar(x, mean)
plt.errorbar(x, mean, sem)
plt.show()


# regret processing
data1 = pickle.load(open("regret1.pkl", "rb"))
regret = data1
np.savetxt("regret.csv", regret, delimiter=",")

# confirm all data is here
print(np.shape(regret))

# get metrics
mean = np.mean(regret, axis=0)
sem = np.std(regret, axis=0) / np.sqrt(30)

# plot result
x = range(4)
plt.bar(x, mean)
plt.errorbar(x, mean, sem)
plt.show()