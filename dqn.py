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

from common.schedules import LinearSchedule
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Transition

class Q_Network(nn.Module):
    def __init__(self, dqn_type, layers):
        super(Q_Network, self).__init__()
        self.dqn_type = dqn_type
        
        network = []
        for idx in range(len(layers)-3):
            network += [nn.Linear(layers[idx], layers[idx+1])]
            network += [nn.ReLU()]
       
        self.body = nn.Sequential(*network)

        if self.dqn_type == "dueling":
            self.adv_stream = nn.Sequential(
                nn.Linear(layers[-3], layers[-2]),
                nn.ReLU(),
                nn.Linear(layers[-2], layers[-1])
            )

            self.val_stream = nn.Sequential(
                nn.Linear(layers[-3], layers[-2]),
                nn.ReLU(),
                nn.Linear(layers[-2], 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(layers[-3], layers[-2]*2),
                nn.ReLU(),
                nn.Linear(layers[-2]*2, layers[-1])
            )

    def forward(self, state):
        feats = self.body(state)
        if self.dqn_type == "dueling":
            values = self.val_stream(feats)
            advantages = self.adv_stream(feats)
            return values + (advantages - advantages.mean(dim=1, keepdims=True))
        else:
            return self.head(feats)

class DQN:
    def __init__(self, config):
        self.writer = SummaryWriter() 
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.dqn_type = config["dqn-type"]
        self.run_title = config["run-title"]
        self.env = gym.make(config["environment"])

        self.num_states  = np.prod(self.env.observation_space.shape)
        self.num_actions = self.env.action_space.n

        layers = [
            self.num_states, 
            *config["architecture"], 
            self.num_actions
        ]

        self.policy_net = Q_Network(self.dqn_type, layers).to(self.device)
        self.target_net = Q_Network(self.dqn_type, layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        capacity = config["max-experiences"]
        self.p_replay_eps = config["p-eps"]
        self.prioritized_replay = config["prioritized-replay"]
        self.replay_buffer = PrioritizedReplayBuffer(capacity, config["p-alpha"]) if self.prioritized_replay \
                        else ReplayBuffer(capacity)

        self.beta_scheduler = LinearSchedule(config["episodes"], initial_p=config["p-beta-init"], final_p=1.0)
        self.epsilon_decay = lambda e: max(config["epsilon-min"], e * config["epsilon-decay"])

        self.train_freq = config["train-freq"]
        self.use_soft_update = config["use-soft-update"]
        self.target_update = config["target-update"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch-size"]
        self.time_step = 0

        self.optim = T.optim.AdamW(self.policy_net.parameters(), lr=config["lr-init"], weight_decay=config["weight-decay"])
        self.lr_scheduler = T.optim.lr_scheduler.StepLR(self.optim, step_size=config["lr-step"], gamma=config["lr-gamma"])
        self.criterion = nn.SmoothL1Loss(reduction="none") # Huber Loss
        self.min_experiences = max(config["min-experiences"], config["batch-size"])

        self.save_path = config["save-path"]

    def act(self, state, epsilon=0):
        """
            Act on environment using epsilon-greedy policy
        """
        if np.random.sample() < epsilon:
            return int(np.random.choice(np.arange(self.num_actions)))
        else:
            self.policy_net.eval()
            return self.policy_net(T.tensor(state, device=self.device).float().unsqueeze(0)).argmax().item()

    def _soft_update(self, tau):
        """
            Polyak averaging: soft update model parameters. 
            θ_target = τ*θ_current + (1 - τ)*θ_target
        """
        for target_param, current_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*target_param.data + (1.0-tau)*current_param.data)

    def update_target(self, tau):
        if self.use_soft_update:
            self._soft_update(tau)
        elif self.time_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self, beta=None):
        if len(self.replay_buffer) < self.min_experiences:
            return None, None 

        self.policy_net.train()

        if self.prioritized_replay:
            transitions, (is_weights, t_idxes) = self.replay_buffer.sample(self.batch_size, beta)
        else:
            transitions = self.replay_buffer.sample(self.batch_size)
            is_weights, t_idxes = np.ones(self.batch_size), None

        # transpose the batch --> transition of batch-arrays
        batch = Transition(*zip(*transitions))
        # compute a mask of non-final states and concatenate the batch elements
        non_final_mask = T.tensor(tuple(map(lambda state: state is not None, batch.next_state)), 
                                                                device=self.device, dtype=T.bool)  
        non_final_next_states = T.cat([T.tensor([state]).float() for state in batch.next_state if state is not None]).to(self.device)

        state_batch  = T.tensor(batch.state,  device=self.device).float()
        action_batch = T.tensor(batch.action, device=self.device).long()
        reward_batch = T.tensor(batch.reward, device=self.device).float()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
        next_state_values = T.zeros(self.batch_size, device=self.device)
        if self.dqn_type == "vanilla":
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        else:
            self.policy_net.eval()
            action_next_state = self.policy_net(non_final_next_states).max(1)[1]
            self.policy_net.train()
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, action_next_state.unsqueeze(1)).squeeze().detach()

        # compute the expected Q values (RHS of the Bellman equation)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # compute temporal difference error
        td_error = T.abs(state_action_values.squeeze() - expected_state_action_values).detach().cpu().numpy()

        # compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = T.mean(loss * T.tensor(is_weights, device=self.device))
      
        # optimize the model
        self.optim.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        return td_error, t_idxes

    def run_episode(self, epsilon, beta):
        total_reward, done = 0, False
        state = self.env.reset()
        while not done:
            # use epsilon-greedy to get an action
            action = self.act(state, epsilon)
            # caching the information of current state
            prev_state = state
            # take action
            state, reward, done, _ = self.env.step(action)
            # accumulate reward
            total_reward += reward
            # store the transition in buffer
            if done: state = None 
            self.replay_buffer.push(prev_state, action, state, reward)
            # optimize model
            if self.time_step % self.train_freq == 0:
                td_error, t_idxes = self.optimize(beta=beta)
                # update priorities 
                if self.prioritized_replay and td_error is not None:
                    self.replay_buffer.update_priorities(t_idxes, td_error + self.p_replay_eps)
            # update target network
            self.update_target(self.tau)
            # increment time-step
            self.time_step += 1

        return total_reward

    def train(self, episodes, epsilon, solved_reward):
        total_rewards = np.zeros(episodes)
        for episode in range(episodes):
            
            # compute beta using linear scheduler
            beta = self.beta_scheduler.value(episode)
            # run episode and get rewards
            reward = self.run_episode(epsilon, beta)
            # exponentially decay epsilon
            epsilon = self.epsilon_decay(epsilon)
            # reduce learning rate by
            self.lr_scheduler.step()

            total_rewards[episode] = reward
            avg_reward = total_rewards[max(0, episode-100):(episode+1)].mean()
            last_lr = self.lr_scheduler.get_last_lr()[0]

            # log into tensorboard
            self.writer.add_scalar(f'dqn-{self.dqn_type}/reward', reward, episode)
            self.writer.add_scalar(f'dqn-{self.dqn_type}/reward_100', avg_reward, episode)
            self.writer.add_scalar(f'dqn-{self.dqn_type}/lr', last_lr, episode)
            self.writer.add_scalar(f'dqn-{self.dqn_type}/epsilon', epsilon, episode)

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
            action = self.act(state)
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