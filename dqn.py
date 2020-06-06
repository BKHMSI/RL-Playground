import os 
import gym
import yaml
import random 
import argparse
import torch as T
import numpy as np

from time import sleep
from tqdm import tqdm
from collections import namedtuple

from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = []
        self.position = 0
        self.capacity = capacity

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Q_Network(nn.Module):
    def __init__(self, layers):
        super(Q_Network, self).__init__()
        network = []
        
        for idx in range(len(layers)-1):
            network += [nn.Linear(layers[idx], layers[idx+1])]
            if idx+2 < len(layers):
                network += [nn.ReLU()]
       
        self.network = nn.Sequential(*network)
    
    def forward(self, state):
        return self.network(state)

class DQN:
    def __init__(self, config):
        self.writer = SummaryWriter() 
        self.device = config["device"]
        self.dqn_type = config["type"]
        self.run_title = config["run-title"]
        self.env = gym.make(config["environment"])


        self.num_states  = np.prod(self.env.observation_space.shape)
        self.num_actions = self.env.action_space.n

        layers = [
            self.num_states, 
            *config["architecture"], 
            self.num_actions
        ]

        self.policy_net = Q_Network(layers).to(self.device)
        self.target_net = Q_Network(layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.buffer = ReplayBuffer(config["max-experiences"])

        self.decay_epsilon = lambda e: max(config["epsilon-min"], e * config["epsilon-decay"])

        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch-size"]

        self.optim = T.optim.AdamW(self.policy_net.parameters(), lr=config["lr-init"], weight_decay=config["weight-decay"])
        self.lr_scheduler = T.optim.lr_scheduler.StepLR(self.optim, step_size=config["lr-step"], gamma=config["lr-gamma"])
        self.criterion = nn.SmoothL1Loss() # Huber Loss
        self.min_experiences = max(config["min-experiences"], config["batch-size"])

        self.save_path = config["save-path"]

    def get_action(self, state, epsilon):
        """
            Get an action using epsilon-greedy
        """
        if np.random.sample() < epsilon:
            return int(np.random.choice(np.arange(self.num_actions)))
        else:
            return self.policy_net(T.tensor(state, device=self.device).float()).argmax().item()

    def soft_update(self):
        """
            Polyak averaging: soft update model parameters. 
            θ_target = τ*θ_current + (1 - τ)*θ_target
        """
        for target_param, current_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau*target_param.data + (1.0-self.tau)*current_param.data)

    def optimize(self):
        if len(self.buffer) < self.min_experiences:
            return 

        transitions = self.buffer.sample(self.batch_size)
        # transpose the batch --> transition of batch-arrays
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = T.tensor(tuple(map(lambda state: state is not None, batch.next_state)), 
                                                                device=self.device, dtype=T.bool)  
        non_final_next_states = T.cat([T.tensor([state]).float() for state in batch.next_state if state is not None]).to(self.device)

        state_batch  = T.tensor(batch.state,  device=self.device).float()
        action_batch = T.tensor(batch.action, device=self.device).long()
        reward_batch = T.tensor(batch.reward, device=self.device).float()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
        next_state_values = T.zeros(self.batch_size, device=self.device)
        if self.dqn_type == "DDQN":
            self.policy_net.eval()
            action_next_state = self.policy_net(non_final_next_states).max(1)[1]
            self.policy_net.train()
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, action_next_state.unsqueeze(1)).squeeze().detach()
        else:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optim.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        # Update target network
        self.soft_update()

    def run_episode(self, epsilon):
        total_reward, done = 0, False
        state = self.env.reset()
        while not done:
            # Using epsilon-greedy to get an action
            self.policy_net.eval()
            action = self.get_action(state, epsilon)
            # Caching the information of current state
            prev_state = state
            # Take action
            state, reward, done, _ = self.env.step(action)
            # Accumulate reward
            total_reward += reward
            # Store the transition in buffer
            if done: state = None 
            self.buffer.push(prev_state, action, state, reward)
            # Optimize model
            self.policy_net.train()
            self.optimize()

        return total_reward

    def train(self, episodes, epsilon, solved_reward):
        total_rewards = np.zeros(episodes)
        for episode in range(episodes):

            reward = self.run_episode(epsilon)
            epsilon = self.decay_epsilon(epsilon)
            self.lr_scheduler.step()

            total_rewards[episode] = reward
            avg_reward = total_rewards[max(0, episode-100):(episode+1)].mean()
            last_lr = self.lr_scheduler.get_last_lr()[0]

            self.writer.add_scalar(f'{self.run_title}/reward', reward, episode)
            self.writer.add_scalar(f'{self.run_title}/reward_100', avg_reward, episode)
            self.writer.add_scalar(f'{self.run_title}/lr', last_lr, episode)
            self.writer.add_scalar(f'{self.run_title}/epsilon', epsilon, episode)

            print(f"Episode: {episode} | Last 100 Average Reward: {avg_reward:.5f} | Learning Rate: {last_lr:.5E} | Epsilon: {epsilon:.5E}", end='\r')

            if avg_reward > solved_reward:
                break
        
        self.writer.close()
        print(f"Environment solved in {episode} episodes")
        T.save(self.policy_net.state_dict(), os.path.join(self.save_path, f"{self.run_title}.pt"))

    def visualize(self, load_path=None):
        done = False
        state = self.env.reset()

        if load_path is not None:
            self.policy_net.load_state_dict(T.load(load_path, map_location=self.device))
        self.policy_net.eval()
        
        while not done:
            self.env.render()
            action = self.get_action(state, -1)
            state, _, done, _ = self.env.step(int(action))
            sleep(0.01) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="configs/dqn.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    agent = DQN(config)

    if config["train"]:
        agent.train(episodes=config["episodes"], epsilon=config["epsilon-start"], solved_reward=config["solved-criterion"])
    
    if config["visualize"]:
        for _ in range(config["vis-episodes"]):
            agent.visualize(config["load-path"])