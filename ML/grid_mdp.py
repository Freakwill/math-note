#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""强化学习演示程序V2.0

在这个格子世界的一个机器人要去寻找金子（黄色圆圈），同时需要避免很多陷阱（黑色圆圈）

主要方法:
step: 每次主体与环境互动的流程
    调用get_reward与nex_state
get_reward: 每次互动获得回报 r=g(s,a,s')
nex_state: 转态迁移 s'=f(s,a)

依赖库:
    numpy
    gym
"""

import logging
import numpy
import random, time
from gym import spaces
import gym

logger = logging.getLogger(__name__)

# 配置
## 格子数目
M, N = 7, 7
## 窗口大小
edge = 100
screen_width = edge*(M+2)
screen_height = edge*(N+2)
## 陷阱
traps = {(1,2),
(5,3),
(6,2),
(2,4),
(3,3)
}
## 金币
gold = (3,1)

## 格子位置与坐标对应关系
def coordinate(position):
    return position[0]*edge+edge//2, position[1]*edge+edge//2


class GridEnv(gym.Env):
    """Grid world 格子世界
    
    A robot playing the grid world, tries to find the golden (yellow circle), meanwhile
    it has to avoid of the traps(black circles)
    在这个格子世界的一个机器人要去寻找金子（黄色圆圈），同时需要避免很多陷阱（黑色圆圈）
    
    Extends:
        gym.Env
    
    Variables:
        metadata {dict} -- configuration of rendering
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.terminate_states = traps | {gold}

        self.actions = ['n','e','s','w']

        self.gamma = 0.8
        self.viewer = None
        self.state = None


    def get_reward(self, state, action, next_state):
        """回报函数
        
        被step方法调用
        
        Arguments:
            state -- 动作之前的状态
            action -- 动作
            next_state -- 动作之后的状态
        
        Returns:
            number -- 回报值
        """
        if next_state in traps:
            return -1
        elif next_state in {gold}:
            return 2
        else:
            return -0.1

    def next_state(self, state, action):
        """状态迁移方法
        
        这是一个确定性迁移
        
        Arguments:
            state-- 当前状态
            action-- 动作
        
        Returns:
            新状态
        
        Raises:
            Exception -- 无效动作
        """
        if action=='e':
            if state[0]<=M-1:
                state = (state[0]+1, state[1])
            return state
        elif action=='w':
            if state[0]>=2:
                state = (state[0]-1, state[1])
            return state
        elif action=='s':
            if state[1]>=2:
                state = (state[0], state[1]-1)
            return state
        elif action=='n':
            if state[1]<=N-1:
                state = (state[0], state[1]+1)
            return state
        else:
            raise Exception('invalid action!')

    def __getitem__(self, key):
        return self.next_state(*key)


    def step(self, action):
        """环境核心程序
        
        主体和环境互动: 主体执行动作，转变状态，并从环境中获得回报
        
        Arguments:
            action -- 主体执行的动作
        
        Returns:
            tuple -- 新状态、回报值、终止判断、其他信息
        """
        # current state
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}

        # state transitating
        next_state = self[state, action]
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        r = self.get_reward(state, action, next_state)

        return next_state, r, is_terminal, {}

    def reset(self):
        self.state = 1, N
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # grid world           
            for k in range(M+1):
                line = rendering.Line(((1+k)*edge, edge), ((1+k)*edge, (N+1)*edge))
                line.set_color(0,0,0)
                self.viewer.add_geom(line)
            for k in range(N+1):
                line = rendering.Line((edge, (1+k)*edge), ((M+1)*edge, (1+k)*edge))
                line.set_color(0,0,0)
                self.viewer.add_geom(line)

            # traps
            for trap in traps:
                self.trap = rendering.make_circle(30)
                self.trap.add_attr(rendering.Transform(translation=coordinate(trap)))
                self.trap.set_color(0, 0, 0)
                self.viewer.add_geom(self.trap)
            # gold
            self.gold = rendering.make_circle(30)
            self.circletrans = rendering.Transform(translation=coordinate(gold))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)
            self.gold_hole = rendering.make_circle(15)
            self.gold_hole.add_attr(rendering.Transform(translation=coordinate(gold)))
            self.gold_hole.set_color(1, 1, 1)
            self.viewer.add_geom(self.gold);
            self.viewer.add_geom(self.gold_hole)

            # robot
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None
        self.robotrans.set_translation(edge*self.state[0]+50, edge*self.state[1]+50)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
