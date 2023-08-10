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

filename = "Push/data/error_push_25_50.pkl"
data1,data1b = data2array(filename)
filename = "Close/data/error_pour_25_50.pkl"
data2,data2b = data2array(filename)
filename = "Pour/data/error_close_25_50.pkl"
data3,data3b = data2array(filename)

error1 = get_error(data1)
error2 = get_error(data2)
error3 = get_error(data3)

error = error1 + error2 + error3
error = np.array(error)
#np.savetxt("error.csv", error, delimiter=",")

# confirm all data is here
print("ShapeTest",np.shape(error))

colors = [(.4, .4, .4),(.701,.701, .701),(0.627,0.832 , .643),(1, 0.6, 0)]
colorsc = {'Ignore':(.4, .4, .4), 'Sampling':(.701,.701, .701),'Maximum':(0.627,0.832 , .643),'Double MH':(1, 0.6, 0)}   
labels = list(colorsc.keys())
handles = [plt.Rectangle((0,0),1,1, color=colorsc[label]) for label in labels]

# get metrics
mean = np.mean(error1, axis=0)
sem = np.std(error1, axis=0) / np.sqrt(30)
print("Error1",mean,sem)
# plot result
x = range(4)

plt.bar(x, mean,color = colors)

plt.legend(handles, labels)
plt.errorbar(x, mean, sem,xerr = None,ls='none')
plt.title("Error in Theta - Push")
plt.xlabel("Method: Ignore, Sampling, Maximum, Double MH")
plt.ylabel("Error in Theta")
plt.show()

# get metrics
mean = np.mean(error2, axis=0)
sem = np.std(error2, axis=0) / np.sqrt(30)
print("Error2",mean,sem)
# plot result
x = range(4)
plt.bar(x, mean,color = colors)

plt.legend(handles, labels)
plt.errorbar(x, mean, sem,xerr = None,ls='none')
plt.title("Error in Theta - Close")
plt.xlabel("Method: Ignore, Sampling, Maximum, Double MH")
plt.ylabel("Error in Theta")
plt.show()

# get metrics
mean = np.mean(error3, axis=0)
sem = np.std(error3, axis=0) / np.sqrt(30)
print("Error3",mean,sem)
# plot result
x = range(4)
plt.bar(x, mean,color = colors)
plt.legend(handles, labels)
plt.errorbar(x, mean, sem,xerr = None,ls='none')
plt.title("Error in Theta - Pour")
plt.xlabel("Method: Ignore, Sampling, Maximum, Double MH")
plt.ylabel("Error in Theta")
plt.show()

# get metrics
mean = np.mean(error, axis=0)
sem = np.std(error, axis=0) / np.sqrt(30)
print("Error",mean,sem)
# plot result
x = range(4)
plt.bar(x, mean,color = colors)

plt.legend(handles, labels)
plt.errorbar(x, mean, sem,xerr = None,ls='none')
plt.title("Error in Theta - Total")
plt.xlabel("Method: Ignore, Sampling, Maximum, Double MH")
plt.ylabel("Error in Theta")
plt.show()

# regret processing
#data11 = pickle.load(open("regret1.pkl", "rb"))
regret1 = list(data1b)
regret2 = list(data2b)
regret3 = list(data3b)

# confirm all data is here
print(np.shape(regret1))
print(np.shape(regret2))
print(np.shape(regret3))


regret = regret1 + regret2 + regret3
#np.savetxt("regret.csv", regret, delimiter=",")



# get metrics
mean = np.mean(regret, axis=0)
sem = np.std(regret, axis=0) / np.sqrt(30)
print("Reg",mean,sem)
# plot result
x = range(4)
plt.bar(x, mean,color = colors)

plt.legend(handles, labels)
plt.errorbar(x, mean, sem,xerr = None,ls='none')
plt.title("Regret - Total")
plt.xlabel("Method: Ignore, Sampling, Maximum, Double MH")
plt.ylabel("Regret")
plt.ylim([0,.3])
plt.show()


# get metrics
mean = np.mean(regret1, axis=0)
sem = np.std(regret1, axis=0) / np.sqrt(30)
print("Reg1",mean,sem)

# plot result
x = range(4)
plt.bar(x, mean,color = colors)

plt.legend(handles, labels)
plt.errorbar(x, mean, sem,xerr = None,ls='none')
plt.title("Regret - Push")
plt.xlabel("Method: Ignore, Sampling, Maximum, Double MH")
plt.ylabel("Regret")
plt.show()
#sim2
# get metrics
mean = np.mean(regret2, axis=0)
sem = np.std(regret2, axis=0) / np.sqrt(30)
print("Reg2",mean,sem)

# plot result
x = range(4)
plt.bar(x, mean,color = colors)

plt.legend(handles, labels)
plt.errorbar(x, mean, sem,xerr = None,ls='none')
plt.title("Regret - Close")
plt.xlabel("Method: Ignore, Sampling, Maximum, Double MH")
plt.ylabel("Regret")
plt.show()
#sim3
# get metrics
mean = np.mean(regret3, axis=0)
sem = np.std(regret3, axis=0) / np.sqrt(30)
print("Reg3",mean,sem)
# plot result
x = range(4)
plt.bar(x, mean,color = colors)
plt.legend(handles, labels)
plt.errorbar(x, mean, sem,xerr = None,ls='none')
plt.title("Regret - Pour")
plt.xlabel("Method: Ignore, Sampling, Maximum, Double MH")
plt.ylabel("Regret")
plt.ylim([0,.3])
plt.show()

# filename = "sim1/data/rebuttal/error_s1_time325.pkl"
# data = pickle.load(open(filename, "rb"))
# # plot result
# use2 = []
# for use in data:
# 	use2.append(use/50)
# x = range(4)
# plt.bar(x,use2,color= [(.4, .4, .4),(.701,.701, .701),(0.627,0.832 , .643),(1, 0.6, 0)])
# plt.title("Avg Cycle Time per Method")
# plt.xlabel("Method:Naive, Mean, Max, DMH")
# plt.show()
