import numpy as np
import random


def p_xi_theta(env, xi, theta, beta=10.0):
    f = env.feature_count(xi)
    R = env.reward(f, theta)
    return np.exp(beta * R), f, R


def rand_demo(xi):
    n, m = np.shape(xi)
    xi1 = np.copy(xi)
    for idx in range(1, n-1):
        # range of joints
        q1 = np.random.uniform(-np.pi/4, np.pi/4)
        q2 = np.random.uniform(-1.76, 1.76)
        q3 = np.random.uniform(-2.89, 2.89)
        q4 = np.random.uniform(-3.07, -0.1)
        q5 = np.random.uniform(-2.89, 2.89)
        q6 = np.random.uniform(-0.01, 3.75)
        q7 = np.random.uniform(-2.89, 2.89)
        xi1[idx, :] = np.array([q1, q2, q3, q4, q5, q6, q7])
    return xi1


def human_demo(env, xi, theta, n_samples, n_demos):
    XI = []
    Fs = []
    P = np.array([0.] * n_samples)
    for idx in range(n_samples):
        xi1 = rand_demo(xi)
        expR, f, _ = p_xi_theta(env, xi1, theta)
        P[idx] = expR
        Fs.append(f)
        XI.append(xi1)
    P /= np.sum(P)
    demos = []
    for _ in range(n_demos):
        sample_idx = np.random.choice(n_samples, p=P)
        demos.append(XI[sample_idx])
    best_idx = np.argmax(P)
    return demos, XI[best_idx], Fs, Fs[best_idx]


def get_regret(env, feature_set, theta_star, f_star, theta_hat):
    f_hat, reward_hat = None, -np.inf
    for f in feature_set:
        curr_reward = env.reward(f, theta_hat)
        if curr_reward > reward_hat:
            reward_hat = curr_reward
            f_hat = f
    reward_star = env.reward(f_star, theta_star)
    reward_hat = env.reward(f_hat, theta_star)
    return reward_star - reward_hat


def mcmc_naive(env, D, n_outer_samples, n_burn, len_theta):
    theta = np.random.rand(len_theta)
    p_theta = 1.0
    for xi in D:
        expR, _, _, = p_xi_theta(env, xi, theta)
        p_theta *= expR
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        p_theta1 = 1.0
        for xi in D:
            expR1, _, _, = p_xi_theta(env, xi, theta1)
            p_theta1 *= expR1
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def mcmc_mean(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    Z_theta = Z_mean(env, xi0, theta, n_inner_samples)
    p_theta = 1.0
    for xi in D:
        expR, _, _, = p_xi_theta(env, xi, theta)
        p_theta *= expR / Z_theta
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        Z_theta1 = Z_mean(env, xi0, theta1, n_inner_samples)
        p_theta1 = 1.0
        for xi in D:
            expR1, _, _, = p_xi_theta(env, xi, theta1)
            p_theta1 *= expR1 / Z_theta1
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def mcmc_max(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    Z_theta = Z_max(env, xi0, theta, n_inner_samples)
    p_theta = 1.0
    for xi in D:
        expR, _, _, = p_xi_theta(env, xi, theta)
        p_theta *= expR / Z_theta
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        Z_theta1 = Z_max(env, xi0, theta1, n_inner_samples)
        p_theta1 = 1.0
        for xi in D:
            expR1, _, _, = p_xi_theta(env, xi, theta1)
            p_theta1 *= expR1 / Z_theta1
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def mcmc_double(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    theta_samples = []
    y_prev = (False, None)
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        y = inner_sampler(env, D, theta1, n_inner_samples, y_prev)
        p_theta1 = 1.0
        for xi in D:
            expRx1, _, _, = p_xi_theta(env, xi, theta1)
            expRy1, _, _, = p_xi_theta(env, y, theta1)
            p_theta1 *= expRx1 / expRy1
        p_theta = 1.0
        for xi in D:
            expRx, _, _, = p_xi_theta(env, xi, theta)
            expRy, _, _, = p_xi_theta(env, y, theta)
            p_theta *= expRx / expRy
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            y_prev = (True, y)
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]


def Z_mean(env, xi, theta, n_samples):
    mean_reward = 0.
    for _ in range(n_samples):
        xi1 = rand_demo(xi)
        expR, _, _, = p_xi_theta(env, xi1, theta)
        mean_reward += expR
    return mean_reward / n_samples


def Z_max(env, xi, theta, n_samples):
    max_reward = -np.inf
    for _ in range(n_samples):
        xi1 = rand_demo(xi)
        expR, _, _, = p_xi_theta(env, xi1, theta)        
        if expR > max_reward:
            max_reward = expR
    return max_reward


def inner_sampler(env, D, theta, n_samples, y_init=(False, None)):
    if y_init[0]:
        y = y_init[1]
    else:
        y = random.choice(D)
    y_score, _, _, = p_xi_theta(env, y, theta)        
    for _ in range(n_samples):
        y1 = rand_demo(y)
        y1_score, _, _, = p_xi_theta(env, y1, theta)
        if y1_score / y_score > np.random.rand():
            y = np.copy(y1)
            y_score = y1_score
    return y