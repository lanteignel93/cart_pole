import random
import gym
import sys
import numpy as np
from collections import deque,namedtuple
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
plt.style.use('seaborn')

class DQN(nn.Module):
    def __init__(self,hidden_sz,state_sz, action_sz):
        super().__init__()
        self.hidden_sz = hidden_sz

        self.fc1 = nn.Linear(state_sz,self.hidden_sz)
        self.fc2 = nn.Linear(self.hidden_sz,self.hidden_sz)
        self.fc3 = nn.Linear(self.hidden_sz,action_sz)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

class Agent():
    def __init__(self,env,target_update_frequency=100,eps=1):

        self.env = env
        self.action_sz = self.env.action_space.n
        self.state_sz = self.env.observation_space.shape[0]
        self.eps = eps
        self.target_update_frequency = target_update_frequency
        self.target_update_counter = 0
        self.rewards = []
        self.train_time = None
        self.n_episodes = None
        self.batch_size = None
        self.gamma = None
        self.lr = None
        self.decay = None
        self.replay_buffer = deque(maxlen=10000)
        self.transition = namedtuple('transition',['s_prime','reward','s','action','done'])
        self.network = DQN(256,self.state_sz, self.action_sz)
        self.target_network = DQN(256,self.state_sz, self.action_sz)
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def print_env_settings(self):
        print('State space: ',self.state_sz)
        print('Action space: ',self.action_sz)

    def init_hyperparameters(self, n_episodes,batch_size,gamma,lr,decay):
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.decay = decay
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)

    def select_action(self,state,eps):

        t = np.random.random()
        if t < eps:
            a = np.random.choice(range(self.action_sz))
        else:
            q = self.network(torch.FloatTensor(state))
            a = q.argmax().item()
        return a

    def store(self,transition):
        self.replay_buffer.append(transition)

    def update(self):

        if len(self.replay_buffer)< self.batch_size:
            return

        batch = random.sample(self.replay_buffer,self.batch_size)

        s = torch.FloatTensor([t.s for t in batch])
        r = torch.FloatTensor([t.reward for t in batch])
        s_prime = torch.FloatTensor([t.s_prime for t in batch])
        a = torch.LongTensor([t.action for t in batch]).unsqueeze(1)
        done = torch.FloatTensor([t.done for t in batch])

        target = (r + self.gamma*self.target_network(s_prime).max(dim=1)[0]*(1-done))

        prediction = self.network(s).gather(1,a)


        self.optimizer.zero_grad()

        loss = self.loss_fn(target.unsqueeze(1),prediction)

        loss.backward()

        self.optimizer.step()

    def get_train_time(self):
        return self.train_time

    def run_episode(self,render,k):

        s = self.env.reset()
        done = False
        total_reward = 0.0
        self.eps = self.eps * self.decay
        transition_count = 0

        while not done:
            if render:
                self.env.render()

            self.target_update_counter += 1

            if self.eps > 0.01:
                eps = self.eps
            else:
                eps = 0.01

            action = self.select_action(s,eps)
            s_prime,reward,done,_ = self.env.step(action)

            self.store((self.transition(s_prime,reward,s,action,done)))

            total_reward += reward

            s = s_prime

            done = done

            self.update()

            transition_count+=1
        if k % 100 == 0 and k > 1:
            print('Transition Count: ',transition_count)
            print('Episode Reward: ',total_reward)
        self.rewards.append(total_reward)

    def run_episode2(self,render,k):

        s = self.env.reset()
        done = False
        total_reward = 0.0
        self.eps = self.eps * self.decay
        transition_count = 0

        while not done:
            if render:
                self.env.render()

       #     eps = 0.0
            transition_count+=1
            self.target_update_counter += 1

            if self.eps > 0.01:
                eps = self.eps
            else:
                eps = 0.01

            action = self.select_action(s,eps)

            s_prime,reward,done,_ = self.env.step(action)

            next_state = np.reshape(s_prime, [1, self.state_sz])
            s_ = np.reshape(s, [1, self.state_sz])

            # We want to encourage swing moves
            if next_state[0][0] > s_[0][0] and next_state[0][0]>-0.4 and s_[0][0]>-0.4:
                reward += 20
            elif next_state[0][0] < s_[0][0] and next_state[0][0]<=-0.6 and s_[0][0]<=-0.6:
                reward += 20
            # Massive reward to reach flag
            if done and transition_count != 200:
                reward = reward + 10000
            else:
                # put extra penalty if not done
                reward = reward - 10

            self.store(self.transition(s_prime,reward,s,action,done))
            total_reward += reward
            s = s_prime

            done = done

            self.update()
        if k % 100 == 0 and k > 1:
            print('Transition Count: ',transition_count)
            print('Episode Reward: ',total_reward)
        self.rewards.append(total_reward)


    def train(self):
        t1 = time.time()
        for k in range(self.n_episodes):
            if k == self.n_episodes - 1:
                self.train_time = time.time() - t1

            render = False

#             if k % 100 <= 10:
#                 render = True
            if k % 100 == 0 and k > 1:
                print('Episode: ',k)
            self.run_episode(render,k)

            if self.target_update_counter >= self.target_update_frequency:

                self.target_update_counter = 0
                self.target_network.load_state_dict(self.network.state_dict())

    def train2(self):
        t1 = time.time()
        for k in range(self.n_episodes):
            if k == self.n_episodes - 1:
                self.train_time = time.time() - t1

            render = False

#             if k % 100 <= 10:
#                 render = True
            if k % 100 == 0 and k > 1:
                print('Episode: ',k)
            self.run_episode2(render,k)

            if self.target_update_counter >= self.target_update_frequency:

                self.target_update_counter = 0
                self.target_network.load_state_dict(self.network.state_dict())
