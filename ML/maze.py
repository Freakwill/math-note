#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy
from gym import spaces
import gym
from gym.envs.classic_control import rendering

class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.states = (1,2,3,4,5,6,7,8)
        self.terminate_states = {6,7,8}

        self.actions = ['n','e','s','w']

        self.rewards = {}
        self.rewards[1, 's'] = -1
        self.rewards[3, 's'] = 1
        self.rewards[5, 's'] = -1

        self.t = {}
        self.t[1, 's'] = 6
        self.t[1, 'e'] = 2
        self.t[2, 'w'] = 1
        self.t[2, 'e'] = 3
        self.t[3, 's'] = 7
        self.t[3, 'w'] = 2
        self.t[3, 'e'] = 4
        self.t[4, 'w'] = 3
        self.t[4, 'e'] = 5
        self.t[5, 's'] = 8
        self.t[5, 'w'] = 4

        self.viewer = None
        self.state = None


    def step(self, action):
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        key = state, action

        next_state = self.t.get(key, state)
        self.state = next_state

        is_terminal = next_state in self.terminate_states
        r = self.rewards.get(key, 0)

        return next_state, r, is_terminal, {}

    def reset(self):
        self.state = self.states[0]
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            self.viewer = Grid(screen_width, screen_height)
            self.viewer.draw()

            self.kulo1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(140,150))
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(0,0,0)

            self.kulo2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(460, 150))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(0, 0, 0)

            self.gold = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(300, 150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)

            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None
        else:
            self.robotrans.set_translation(*state2xy(self.state))
            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

x=[140,220,300,380,460,140,300,460]
y=[250,250,250,250,250,150,150,150]
def state2xy(state):
    return x[state-1], y[state-1]


class Grid(rendering.Viewer):
    def __init__(self, *args, **kwargs):
        super(Grid, self).__init__(*args, **kwargs)
        self.line1 = rendering.Line((100, 300), (500, 300))
        self.line2 = rendering.Line((100, 200), (500, 200))
        self.line3 = rendering.Line((100, 300), (100, 100))
        self.line4 = rendering.Line((180, 300), (180, 100))
        self.line5 = rendering.Line((260, 300), (260, 100))
        self.line6 = rendering.Line((340, 300), (340, 100))
        self.line7 = rendering.Line((420, 300), (420, 100))
        self.line8 = rendering.Line((500, 300), (500, 100))
        self.line9 = rendering.Line((100, 100), (180, 100))
        self.line10 = rendering.Line((260, 100), (340, 100))
        self.line11 = rendering.Line((420, 100), (500, 100))

        self.line1.set_color(0, 0, 0)
        self.line2.set_color(0, 0, 0)
        self.line3.set_color(0, 0, 0)
        self.line4.set_color(0, 0, 0)
        self.line5.set_color(0, 0, 0)
        self.line6.set_color(0, 0, 0)
        self.line7.set_color(0, 0, 0)
        self.line8.set_color(0, 0, 0)
        self.line9.set_color(0, 0, 0)
        self.line10.set_color(0, 0, 0)
        self.line11.set_color(0, 0, 0)

        self.add_geom(self.line1)
        self.add_geom(self.line2)
        self.add_geom(self.line3)
        self.add_geom(self.line4)
        self.add_geom(self.line5)
        self.add_geom(self.line6)
        self.add_geom(self.line7)
        self.add_geom(self.line8)
        self.add_geom(self.line9)
        self.add_geom(self.line10)
        self.add_geom(self.line11)
