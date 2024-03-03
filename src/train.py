from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

path = './qnetwork.pt'
config = {'nb_actions': 4,
            'learning_rate': 0.001,
            'gamma':0.99,
            'buffer_size': 1000000,
            'epsilon_min': 0.1,
            'epsilon_max': 1.,
            'epsilon_decay_period': 40000,
            'epsilon_delay_decay': 1200,
            'batch_size': 1024,
            'gradient_steps': 5,
            'update_target_strategy': 'ema',#'replace', # or 'ema'
            'update_target_freq': 200,
            'update_target_tau': 0.001,
            'criterion': torch.nn.SmoothL1Loss(),
            'nb_neurons': 256,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'memory_length':5}

class ProjectAgent:
    def __init__(self):
        self.device = config['device']
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = torch.nn.Sequential(nn.Linear(env.observation_space.shape[0], config['nb_neurons']),
                          nn.ReLU(),
                          nn.Linear(config['nb_neurons'], config['nb_neurons']),
                          nn.ReLU(), 
                          nn.Linear(config['nb_neurons'], config['nb_neurons']),
                          nn.ReLU(), 
                          nn.Linear(config['nb_neurons'], config['nb_neurons']),
                          nn.ReLU(), 
                          nn.Linear(config['nb_neurons'], config['nb_neurons']),
                          nn.ReLU(), 
                          nn.Linear(config['nb_neurons'], env.action_space.n)).to(self.device) 
        
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        #
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
    

    def greedy_action(self, state):
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def act(self, observation, use_random=False):
        if use_random:
            action = env.action_space.sample()
        else:
            action = self.greedy_action(observation)
        return action
    
    
    def save(self, path):
        torch.save(self.model.to('cpu').state_dict(), path)
        self.model.to(self.device)

    def load(self):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        # dqn_state_dict = torch.load('C:/Users/nhcoh/Documents/MVA/S2/RL/rl-class-assignment-NathanielCohen/agent.pt')
        # DQN.load_state_dict(dqn_state_dict)
        # return DQN
    
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            action = self.act(state, np.random.rand() < epsilon)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)

            # next transition
            step += 1
            
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()

                # Save the model if it is the best so far
                if len(episode_return) == 0 or episode_cum_reward >= max(episode_return):
                    print("New best model saved")
                    now = datetime.now().strftime("%Y%m%d-%H%M%S")
                    self.save(f"best_model_{now}.pt")

                episode_return.append(episode_cum_reward)

                episode_cum_reward = 0
                #
                if episode % 10 == 0:
                    plt.plot(episode_return)
                    plt.xlabel('Episode')
                    plt.ylabel('Return')
                    plt.title('Return per Episode')
                    plt.show()
            else:
                state = next_state
        return episode_return

if __name__ == "__main__":
    agent = ProjectAgent()
    episode_return = agent.train(env, 100)
    agent.save(path)
    print(episode_return)
    env.close()