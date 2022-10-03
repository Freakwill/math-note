# Reinforcement Learning

[TOC]

Keywords:

- model

  - transit
  
  - reward
  
  - policy
- value function
  - state function
  - state-action function

- Bellman eq.


- TD algo.
  
  - Sara
  
  - QL
  - on/off policy
  
  
  

## [Dynamic_programming](https://en.wikipedia.org/wiki/Dynamic_programming)

### Model

- functional model

  $(S, A,f, \pi, r, \gamma)$ 

- graphic model

  $(G=(V,E),f:V\to V, r:E\to\mathbb{R})$



#### transition

$$
f(a,s)=s':S\times A\to S
$$

#### Policy

policy: seq. of actions $p=a_1a_2\cdots a_n:W(A)$ or function
$$
\pi(s)=a:S\to A,\\
v(s,a_1a_2\cdots)=r(s,a_1)+r(s_1,a_2)+\cdots, s_1=f(a,s),\cdots
$$
*opt. policy* $v(s):=\max_pv(s,p)$

 recursive formula of state-value function $v$
$$
v(s)=\max_a\{r(s,a)+\gamma v(s')\}, s\overset{a}{\to} s'\\
\pi(s)=\arg\max_a\{r(s,a)+\gamma v(s')\}
$$
or
$$
p(s)=\arg\max_a\{r(s,a)+v(s')\}p(s')
$$
*Bellman principle*

If $v(s,pq)=\max_u v(s,u)$ then $v(s,p)=\max_u v(s,u,s'),v(s,q)=\max_u v(s',u)$



#### Value function

- State-Behavior Value function

$$
q(s,a):= r(s,a) + \gamma v(s)
$$

- State Value function
  $$
  v(s):=\max_a q(s,a)
  $$
  

### Bellman eq.

$s\overset{a/r}\to s'\overset{a'}{\to}s''$:
$$
\begin{cases}
v(s)=r(s,a)+\gamma v(s)\\
q(s,a)=r(s,a)+\gamma q(s',a')
\end{cases}
$$



### Algorithm

```
function v(s)
    v=-M
    for a in actions(s)
        s' <- f(s, a)
        w q <- v(s')  # recursion
        r <- reword(s,a)
        v0 = w + r
        if v0>v
            v = v0
            p=aq
    return v, p
```



## Markov Decision

### Model

$(P_{ss'}^a, \pi, R_t, S, A)$, parameter: $\gamma $ 

#### transition

$$
P_{ss'}^a=p(S_{t+1}=s'|A_t=a,S_t=s)
$$



#### Policy

$$
\pi(a|s)=p(A_t=a|S_t=s)
$$



#### Reward

$$
G_t:=\sum_k\gamma^kR_{t+k+1}
$$



#### State-Value function

$$
v_\pi(s):=E(G_t|S_t=s)=E(\sum_k\gamma^kR_{t+k+1}|S_t=s)\\
= \sum_{a\in A}\pi(a|s)(R^a_s+\gamma\sum_{s'}P_{ss'}^av_\pi(s'))
$$

 

#### State-Behavior Value function

$$
q_\pi(s,a):=E(G_{t}|S_t=s,A_t=a)=E(\sum_k\gamma^kR_{t+k+1}|S_t=s,A_t=a)
$$



### Bellman eq.

$$
\begin{cases}
v(s)=E(R_{t+1}+\gamma v(S_{t+1})|S_t=s)\\
v(S_t)=E(R_{t+1}+\gamma v(S_{t+1}))\\
q_\pi(s,a)=E(R_{t+1}+\gamma q(S_{t+1},A_{t+1})|S_t=s,A_t=a)\\
q_\pi(S_{t},A_{t})=E(R_{t+1}+\gamma q(S_{t+1},A_{t+1}))
\end{cases}
$$



### Based on model

#### Policy Iteration algo.

$\pi\mapsto \pi', q_{\pi}\leq q_{\pi'}$

##### Policy evaluation algo.

$\pi\mapsto v$ (see (11))

##### Policy improvement algo.

$v\mapsto \pi$

## RL

### Based on value function (model is unknown)

MC: $v(s)\sim \frac{1}{N}\sum_i G_i(s)$

Algo.

- initalize: $S, A, Q,\pi, R$  ($\epsilon$-soft wrt $Q$)

- iteration:
  - sampling: generate seq. of $s,a$ by  $\pi$
  - evaluation:$G_i(s)$, then $Q(s,a),v(s)$
  - improvement: $\pi'$

  

On-policy/Off-policy

covering cond.



*Thought*: If model is unknown, then do sampling

### TD

TD-formula
$$
V(S_t)=V(S_t)+\alpha\delta_t, \delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)\\
Q(S_t, A_t)=Q(S_t,A_t)+\alpha\delta_t, \delta_t=R_{t+1}+\gamma V(S_{t+1})-Q(S_t,a_t)
$$

#### Sarsa($\lambda$)

on-policy

see QLearning

#### QLearning(Watkins1989)

off-policy

Algo.

1. initalize: $Q,s$

2. repeat:(update $Q$)

   - $a\in A(s)$

   - iteration:
     - select $a_t$ at $s_t$ by $\epsilon$-greedy
     - $Q(s_t,a_t) :=... $
     - go to $s_{t+1}$

   - until $s$ is terminal

   until $Q$ conv.

3. output terminal policy $\pi$



## Value Approx.

*Goal of traing*
$$
\arg\min_\theta (q(s,a)-\hat{q}(s,a,\theta))^2\\
\arg\min_\theta (v(s)-\hat{v}(s,\theta))^2
$$


SGD:
$$
\theta_{t+1}=\theta_t+\alpha(U_t-\hat{v}(S_t,\theta_t))\nabla \hat{v}(S_t,\theta_t)
$$


Semi-GD:
$$
\theta_{t+1}=\theta_t+\alpha(R_t+\gamma \hat{v}(S_{t+1},\theta)-\hat{v}(S_t,\theta_t))\nabla \hat{v}(S_t,\theta_t)
$$

*linear model*:
$$
\hat{v}(s,\theta)=\theta^T\phi(s)\\
\Delta \theta=\alpha\delta\phi(s)
$$



batch method:
$$
\min \sum_t(v_t-\hat{v}_t(s_t,\theta))^2
$$


#### Sarsa Algo. (based on semi-GD)

$\hat{q}:S\times A\times \R^n\to \R$

- for each episode

  - init s, a

  - repeat

    - select a, get r and s'

    - if s' is terminal: 

      $\theta = \theta+\alpha(r-\hat{q})\nabla \hat{q}$, next episode

  - select a' estimate $\hat{q}(s',a',\theta)$

  - $\theta = \theta+\alpha(r+\gamma \hat{q}(s',a',\theta)-\hat{q}(s,a,\theta))\nabla \hat{q}$

  - $s=s', a=a'$



## policy grad.

$\tau=s_tu_ts_{t+1}\cdots$
$$ {\\}
\max U(\theta)=E(\sum_tR(s_t,u_t)|\theta)=\sum_\tau P(\tau|\theta)R(\tau)\\
\nabla U=1/m\sum_i\nabla \log P(\tau^i|\theta)R(\tau^i),\nabla \log P(\tau^i|\theta)=\sum_t\nabla \log \pi_\theta(u_t^i|s_t^i)
$$


*baseline* $R-b$



## REPO

S. Kakade formula
$$
\eta(\tilde{\pi})=\eta(\pi)+E_{\tilde{\pi}}(\sum\gamma^tA(s_t,a_t))\\
A(s,a):=q(s,a)-v(s)=E_{s'}(r(s,a)+\gamma v(s')-v(s))
$$


## Gym

1. create env, if save it in `gym`

   - copy env file to `/gym/gym/envs/classic_control`
   - append `import GridEnv` to `__init__` file

   

2. register the env

   ```python
   gym.register(
       id='GridWorld',
       entry_point='grid_mdp:GridEnv',
       max_episode_steps=200,
       reward_threshold=100.0
       )
   ```

   

3. demo

   ```python
   import gym
   env = gym.make('GridWorld')
   for i_episode in range(20):
       observation = env.reset(env.action_space.sample())
       for t in range(100):
           env.render(env.action_space.sample())
           print(observation)
           action = env.action_space.sample()
           observation, reward, done, info = env.step(action)
           if done:
               print("Episode finished after {} timesteps".format(t+1))
               break
   env.close()
   ```

### main APIs

1. reset: -> state
2. render
3. step: action -> observant(i.e. state), reward, done, info(dict)

#### code

`maze.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy
from gym import spaces
import gym

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

        self.gamma = 0.8
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

        x=[140,220,300,380,460,140,300,460]
        y=[250,250,250,250,250,150,150,150]

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.line1 = rendering.Line((100,300),(500,300))
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

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None: return None
        self.robotrans.set_translation(x[self.state-1], y[self.state- 1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

```



```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
import maze
import scipy.stats
import numpy as np

env = maze.GridEnv()
states = env.states
actions = env.actions
gamma = env.gamma
N = len(actions)

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


def qlearning(num_iter=20, alpha=0.5, epsilon=0.05):

    qfunc = {}
    for s in states:
        for a in actions:
            key = s, a
            qfunc[key] = 0
    for _ in range(num_iter):
        s = env.reset()
        a = np.random.choice(actions)
        for _ in range(100):
            key = s, a
            s, r, t, i = env.step(a)
            a1 = epsilon_greedy(qfunc, s, 0.001)
            key1 = s, a1
            v = qfunc[key1]
            qfunc[key] += alpha*(r + gamma * v - qfunc[key])
            a = epsilon_greedy(qfunc, s, epsilon)
            if t:
                break

    return qfunc

        
qfunc = qlearning()
p = policy(qfunc, states[0])
print(p)
```

