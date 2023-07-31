import numpy as np
import pickle
import matplotlib.pyplot as plt


def data2array(name):
	data = pickle.load(open(name, "rb"))

	datalist = []
	#print("test1")
	#print(np.shape(data[0]))
	for item in data[0]:
		row = []
		for array in item:
			row.append(list(array))
		datalist.append(row)
	
	#print(np.shape(data[1]))
	datalist2 = []
	for item in data[1]:		
		datalist2.append(item)
	return np.array(datalist),np.array(datalist2)

def get_error(data):
	error = np.zeros((len(data), 4))
	for idx, item in enumerate(data):
		theta = item[0]
		for jdx in range(4):
			error[idx, jdx] = np.linalg.norm(theta - item[jdx+1])
	return list(error)

filename = "sim1/data/error.pkl"
data1,data1b = data2array(filename)
filename = "sim2/data/error.pkl"
data2,data2b = data2array(filename)
filename = "sim3/data/error.pkl"
data3,data3b = data2array(filename)

error1 = get_error(data1)
error2 = get_error(data2)
error3 = get_error(data3)

error = error1 + error2 + error3
error = np.array(error)
#np.savetxt("error.csv", error, delimiter=",")

# confirm all data is here
print("ShapeTest",np.shape(error))

# get metrics
mean = np.mean(error, axis=0)
sem = np.std(error, axis=0) / np.sqrt(30)

# plot result
x = range(4)
plt.bar(x, mean)
plt.errorbar(x, mean, sem)
plt.title("Error in Theta")
plt.xlabel("Method:Naive, Mean, Max, DMH")
plt.show()


# regret processing
#data11 = pickle.load(open("regret1.pkl", "rb"))
regret1 = data1b
regret2 = data2b
regret3 = data3b
regret = regret1 + regret2 + regret3
regret = np.array(regret)
#np.savetxt("regret.csv", regret, delimiter=",")

# confirm all data is here
print(np.shape(regret))


#Total Regret
# get metrics
mean = np.mean(regret, axis=0)
sem = np.std(regret, axis=0) / np.sqrt(30)

# plot result
x = range(4)
plt.bar(x, mean)
plt.errorbar(x, mean, sem)
plt.title("Regret Total")
plt.xlabel("Method:Naive, Mean, Max, DMH")
plt.show()

##Indvidual Sims

# #sim1
# # get metrics
# mean = np.mean(regret1, axis=0)
# sem = np.std(regret1, axis=0) / np.sqrt(30)

# # plot result
# x = range(4)
# plt.bar(x, mean)
# plt.errorbar(x, mean, sem)
# plt.title("Regret Sim1")
# plt.xlabel("Method:Naive, Mean, Max, DMH")
# plt.show()

# #sim2
# # get metrics
# mean = np.mean(regret2, axis=0)
# sem = np.std(regret2, axis=0) / np.sqrt(30)

# # plot result
# x = range(4)
# plt.bar(x, mean)
# plt.errorbar(x, mean, sem)
# plt.title("Regret Sim2")
# plt.xlabel("Method:Naive, Mean, Max, DMH")
# plt.show()

# #sim3
# # get metrics
# mean = np.mean(regret3, axis=0)
# sem = np.std(regret3, axis=0) / np.sqrt(30)

# # plot result
# x = range(4)
# plt.bar(x, mean)
# plt.errorbar(x, mean, sem)
# plt.title("Regret Sim3")
# plt.xlabel("Method:Naive, Mean, Max, DMH")
# plt.show()

