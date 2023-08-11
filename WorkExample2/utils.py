import numpy as np

#The enviroment in this code is just a belief of where a marker should be in between two goals (1 and 2)
#Where the user has a belief of how much to value the distance from each goal relative to the marker.

# parameters of environment
goal1 = [0.1,0.90]
goal2 = [0.9,0.90]


# compute the number line reward
def reward(state,a,theta, beta=50.0):
    position = np.array(state)+np.array(a)
    dist1 = np.linalg.norm([abs(position[0] - goal1[0]),abs(position[1] - goal1[1])])*2
    dist2 = np.linalg.norm([abs(position[0] - goal2[0]),abs(position[1] - goal2[1])])
    f = -theta[0] * dist1 - theta[1] * dist2
    #print(np.exp(theta))
    return np.exp(beta * f)

# # compute the number line reward
# def Qfun(position,action, theta, beta=50.0): 
#     pos2 = position+action
#     Rcurr = reward(pos2,theta)
#     V = -100
#     L = np.linspace(0,1,50)
#     for i in L:
#         for j in L:
#             theta2 = [i,j]
#             V2 = reward(pos2,theta2)
#             if V > V2:
#                 V = V2
#     Q = Rcurr + V
#     return Q


# generate a random position
def rand_action():
    return np.random.rand(2)


# sample for an positon
def human_action(state,theta, n_samples):
    A_H = []
    P = np.array([0.] * n_samples)
    for idx in range(n_samples):
        a_h = rand_action()
        a_state = [a_h[0]+state[0], a_h[1]+state[1]]
        P[idx] = reward(state,a_h, theta)
        A_H.append(a_h)
    P /= np.sum(P)
    sample_idx = np.random.choice(n_samples, p=P) #random sample
    best_idx = np.argmax(P)
    return A_H[sample_idx], A_H[best_idx] #random sample, best sample