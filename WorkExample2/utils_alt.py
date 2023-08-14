import numpy as np

#The environment in this code is just a belief of where a marker should be in between two goals (1 and 2)
#Where the user has a belief of how much to value the distance from each goal relative to the marker.

# parameters of environment
goal1 = np.array([0.1,0.50])
goal2 = np.array([0.9,0.50])


# compute the number line reward
def reward(position, theta, beta=10.0): 
    dist1 = np.linalg.norm([abs(position[0] - goal1[0]),abs(position[1] - goal1[1])])
    dist2 = np.linalg.norm([abs(position[0] - goal2[0]),abs(position[1] - goal2[1])])
    f = -theta[0] * dist1 - theta[1] * dist2
    return np.exp(beta * f)

def Qfun(state,a_h,theta): 
    position = np.array(state) + np.array(a_h) #Base to first state modification
    #Rcurr = reward(pos2,theta)
    #V = 100
    #ideal = ( ( ((2*theta[0])*goal1) + ((theta[1])*goal2) ) / (theta[0]+theta[1]) ) #ideal position 
    #act2 = np.array(ideal-state)*.2

    #position = np.array(pos2)+np.array(act2) #Second state modification
    dist1 = np.linalg.norm([abs(position[0] - goal1[0]),abs(position[1] - goal1[1])])**2
    dist2 = np.linalg.norm([abs(position[0] - goal2[0]),abs(position[1] - goal2[1])])**2
    f = -theta[0] * dist1 - theta[1] * dist2
    Q = np.exp(10.0 * f)

    #Q = Rcurr + V
    return Q

# generate a random position
def rand_action():
    return (np.random.rand(2)-.5)*.4 #restrained magnitude


# sample for an positon
def human_action(state,theta, n_samples):
    A_H = []
    P = np.array([0.] * n_samples)
    for idx in range(n_samples):
        a_h = rand_action()
        a_state = [a_h[0]+state[0], a_h[1]+state[1]]
        P[idx] = reward(a_state, theta)
        A_H.append(a_h)
    P /= np.sum(P)
    sample_idx = np.random.choice(n_samples, p=P) #random sample
    best_idx = np.argmax(P)
    return A_H[sample_idx], A_H[best_idx] #random sample, best sample
