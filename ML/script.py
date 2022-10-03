#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
import maze
import scipy.stats
import numpy as np

env = maze.GridEnv()
states = env.states
actions = env.actions
N = len(actions)

gamma = 0.8

def greedy(qfunc, state):
    amax = actions[0]
    qmax = qfunc[state, amax]
    for action in actions:
        q = qfunc[state, action]
        if qmax < q:
            qmax = q
            amax = action
    return action


def epsilon_greedy(qfunc, state, epsilon=0.05):
    amax = actions[0]
    qmax = qfunc[state, amax]
    for action in actions:
        q = qfunc[state, action]
        if qmax < q:
            qmax = q
            amax = action
    pro = [epsilon/N for action in actions]
    pro[actions.index(amax)] += 1-epsilon

    a = scipy.stats.rv_discrete(values=(np.arange(N), pro))
    return actions[a.rvs()]


def policy(qfunc, state):
    stateList = [state]
    while True:
        a = epsilon_greedy(qfunc, state, 0.001)
        env.state = state
        next_state, r, t, i = env.step(a)
        if next_state != state:
            stateList.extend([a, next_state])
        state = next_state
        if t:
            break
    return stateList


def qlearning(num_iter=10, alpha=0.5, epsilon=0.05):

    qfunc = {}
    for s in states:
        for a in actions:
            key = s, a
            qfunc[key] = 0
    for _ in range(num_iter):
        s = env.reset()
        env.render()
        a = np.random.choice(actions)
        for _ in range(100):
            key = s, a
            s, r, t, i = env.step(a)
            a1 = epsilon_greedy(qfunc, s, 0.001)
            key1 = s, a1
            v = qfunc[key1]
            qfunc[key] += alpha*(r + gamma * v - qfunc[key])
            a = epsilon_greedy(qfunc, s, epsilon)
            env.render()
            if t:
                break
    env.close()

    return qfunc

        
qfunc = qlearning()
# p = policy(qfunc, states[0])
# print(p)

