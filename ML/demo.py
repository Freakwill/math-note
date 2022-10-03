#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import gym
import numpy as np

gym.register(
    id='GridWorld-v0',
    entry_point='grid_mdp:GridEnv',
    max_episode_steps=200,
    reward_threshold=100.0
    )

env = gym.make('GridWorld-v0')
env.seed()

from sklearn.naive_bayes import *
from sklearn.model_selection import  train_test_split
# 这里是NB模型
model = MultinomialNB()

def transform(x,h,m):
    return m+(x-0.5)*h

def demo():
    # demo of RL
    alpha = 0.1  # learning rate
    Q = {}  # Q table, a dict as {state, action:value}
    V = {}  # V table, a dict as {state:value}
    for i in range(300):
        state = env.reset()
        env.render()
        a = format(i+1)
        t = 0
        grade = 0
        if state not in V:
            V[state] = 0
        k =0
        while state not in env.terminate_states and k<100:
            time.sleep(0.1)
            action = greedy(state, Q)
            state1, reward, done, info = env.step(action)
            t += 1
            # print(state1)
            if done:
                if reward == -1:
                    grade = -(t)-80
                    print('false',grade)
                elif reward == 2:
                    grade = -(t)+80
                    print('win',grade)
                
                print((i+1),t)
            env.render()
            key = state, action
            if key not in Q:
                # 当Q表中没有记录该值（即机器人进入新的坏境，或者已经遗忘的环境）时，需要进行推测
                # 用模型对当前输入做出判断；model.predict(X) # 注意类别值翻译成Q表的分值
                # 若没有嵌入NB，只能预判为0值
                if i > 10:
                    X = [(*state, env.actions.index(action))]
                    Q[key] = transform(model.predict(X)[0], h, ymin)
                else:
                    Q[key] = 0
            Q[key] += alpha * (reward + env.gamma * V.get(state1,0) -Q[key])
            V[state] = max(V.get(state,0), Q.get(key,0))
            state = state1
            k+=1
        # 当Q表过大时，随机删除一部分，因为已经有NB进行泛化了。
        # 
        # 嵌入*NB学习*：
        # 从Q表中采样（机器人通过回忆熟悉的环境来学习、适应新环境） X, Y= sample_from(Q)
        # 训练模型 model.fit(X,Y)
        # model 将用于陌生（以及遗忘的）环境下的预判
        # 否则，预判为0值
        # 注意：在样本够并且Q表发生较大修改的情况下，才值得学习！或者间隔几次动作执行NB学习！
        X, Y, ymin, h = sample_from(Q)
        #print(Q)
        #print(Y)
        
        #x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.5,random_state=0)
        model.fit(X,Y)
        y_pre = model.predict(X)
        #print(y_pre)
    env.close()

import random
def greedy(state, Q, epsilon=0.01):
    # greedy policy
    action = random.choice(env.actions)
    if random.random()<epsilon:
        return action
    key = (state, action)
    q = Q.get(key,0)
    for key, value in Q.items():
        s, a = key
        if s == state:
            if q < value:
                q = value
                action = a
    return action

def sample_from(Q):
    """从 Q表中生成学习样本，输入变量：状态，动作
    输出变量：回报值
    这些样本是NB的训练样本

    例:
    Q = {((1,1), 'n'):1, ((1,2),'w'):0}
    X, Y=sample_from(Q)

    [[1 1 0]
    [1 2 3]]

    [1 0]
    
    ---
    Arguments:
        Q {dict} -- Q table
    
    Returns:
        tuple of arrays-- X, Y
    """
    X = np.array([s+(env.actions.index(a),) for s, a in Q.keys()])
    Y = np.array(list(Q.values()))
    ymin, ymax = Y.min(), Y.max()
    h = (ymax - ymin)/5
    Z = 0
    for k in range(1, 6):
        Z += Y >= transform(k,h,ymin)
    #print(ymin,ymax)
    # X, Y 可以进行随机删选
    return X, Z, ymin, h


if __name__ == '__main__':
    demo()
