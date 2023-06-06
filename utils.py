import numpy as np

#The enviroment in this code is just a belief of where a marker should be in between two goals (1 and 2)
#Where the user has a belief of how much to value the distance from each goal relative to the marker.

# parameters of environment
goal1 = 0.0
goal2 = 1.0


# compute the number line reward
def reward(position, theta, beta=10.0): 
    dist1 = abs(position - goal1)*5
    dist2 = abs(position - goal2)
    f = -theta[0] * dist1 - theta[1] * dist2
    return np.exp(beta * f)


# generate a random position
def rand_action():
    return np.random.rand()


# sample for an positon
def human_action(theta, n_samples):
    A_H = []
    P = np.array([0.] * n_samples)
    for idx in range(n_samples):
        a_h = rand_action()
        P[idx] = reward(a_h, theta)
        A_H.append(a_h)
    P /= np.sum(P)
    sample_idx = np.random.choice(n_samples, p=P) #random sample
    best_idx = np.argmax(P)
    return A_H[sample_idx], A_H[best_idx] #random sample, best sample