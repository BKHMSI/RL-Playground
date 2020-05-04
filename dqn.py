import gym
import yaml
import argparse
import torch as T
import numpy as np

from time import sleep
from tqdm import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

class DQN(nn.Module):
    def __init__(self, layers):
        super(DQN, self).__init__()
        network = []
        
        for idx in range(len(layers)-1):
            network += [nn.Linear(layers[idx], layers[idx+1])]
            if idx+2 < len(layers):
                network += [nn.ReLU()]
       
        self.network = nn.Sequential(*network)
    
    def forward(self, state):
        return self.network(state)


class DQNAgent:
    def __init__(self, config, num_states, num_actions):
        self.batch_size = config["batch-size"]
        self.gamma = config["gamma"]
        self.model = DQN([
            num_states, 
            *config["architecture"], 
            num_actions
        ]).to(config["device"])
        self.num_actions = num_actions
        self.device = config["device"]
        self.optim = T.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config["weight-decay"])
        self.criterion = nn.MSELoss()
        self.experience = {'s': [], 'a': [], 'r': [], 's_bar': [], 'done': []} # the buffer
        self.max_experiences = config["max-experiences"]
        self.min_experiences = config["min-experiences"]

    def predict(self, inputs):
        return self.model(T.from_numpy(inputs).float().to(self.device))

    def train(self, target_net):
        if len(self.experience['s']) < self.min_experiences:
            # Only start the training process when we have enough experiences in the buffer
            return 0

        # Randomly select n experience in the buffer, where n is the batch-size
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])

        # Prepare labels for training process
        states_next = np.asarray([self.preprocess(self.experience['s_bar'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(target_net.predict(states_next).detach().cpu().numpy(), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        actions = np.expand_dims(actions, axis=1)
        actions_one_hot = T.FloatTensor(self.batch_size, self.num_actions).zero_()
        actions_one_hot = actions_one_hot.scatter_(1, T.LongTensor(actions), 1).to(self.device)
        selected_action_values = T.sum(self.predict(states) * actions_one_hot, dim=1)
        actual_values = T.FloatTensor(actual_values).to(self.device)

        self.optim.zero_grad()
        loss = self.criterion(selected_action_values, actual_values)
        loss.backward()
        self.optim.step()

    def get_action(self, state, epsilon):
        """
            Get an action using epsilon-greedy
        """
        if np.random.random() < epsilon:
            return int(np.random.choice(np.arange(self.num_actions)))
        else:
            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().cpu().numpy()
            return int(np.argmax(prediction))

    def add_experience(self, exp):
        """
            Used to manage buffer
        """
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, train_net):
        self.model.load_state_dict(train_net.model.state_dict())

    def save_weights(self, path):
        T.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(T.load(path))

    def preprocess(self, state):
        return state


class Trainer:
    def __init__(self, config):
        self.writer = SummaryWriter() 
        self.run_title = config["run-title"]
        self.env = gym.make(config["environment"])

        num_states  = np.prod(self.env.observation_space.shape)
        num_actions = self.env.action_space.n

        self.train_net  = DQNAgent(config, num_states, num_actions)
        self.target_net = DQNAgent(config, num_states, num_actions)

        self.decay_epsilon = lambda e: max(config["min-epsilon"], e * config["decay"])

    def play_game(self, epsilon, copy_step):
        rewards, idx = 0, 0
        done = False
        state = self.env.reset()
        while not done:
            # Using epsilon-greedy to get an action
            action = self.train_net.get_action(state, epsilon)

            # Caching the information of current state
            prev_state = state

            # Take action
            state, reward, done, _ = self.env.step(action)

            # Apply new rules
            # if done:
            #     if reward == 1: # Won
            #         reward = 20
            #     elif reward == 0: # Lost
            #         reward = -20
            #     else: # Draw
            #         reward = 10
            # else:
            #     # reward = -0.05 # Try to prevent the agent from taking a long move
            #     # Try to promote the agent to "struggle" when playing against negamax agent
            #     # as Magolor's (@magolor) idea
            #     reward = 0.5

            rewards += reward

            # Add experience into buffer
            self.train_net.add_experience({
                's': prev_state,
                'a': action,
                'r': reward,
                's_bar': state,
                'done': done
            })

            # Train the training model by using experiences in buffer and the target model
            self.train_net.train(self.target_net)

            idx += 1

            if idx % copy_step == 0:
                # Update the weights of the target model when reaching enough "copy step"
                self.target_net.copy_weights(self.train_net)

        return rewards

    def train(self, episodes, epsilon, copy_step):
        total_rewards = np.zeros(episodes)
        for episode in range(episodes):
            epsilon = self.decay_epsilon(epsilon)
            reward = self.play_game(epsilon, copy_step)
            
            total_rewards[episode] = reward
            avg_reward = total_rewards[max(0, episode-100):(episode+1)].mean()

            self.writer.add_scalar(f'{self.run_title}/reward', reward, episode)
            self.writer.add_scalar(f'{self.run_title}/epsilon', epsilon, episode)
            self.writer.add_scalar(f'{self.run_title}/reward_100', avg_reward, episode)

            print(f"Episode: {episode} | Last 100 Average Reward: {avg_reward}", end='\r')

            if avg_reward > 195:
                self.visualize()

    def visualize(self):
        done = False
        state = self.env.reset()
        while not done:
            self.env.render()
            action = self.train_net.get_action(state, -1)
            state, _, done, _ = self.env.step(int(action))
            sleep(0.01) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="configs/dqn.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    trainer = Trainer(config)
    trainer.train(episodes=config["episodes"], epsilon=config["epsilon"], copy_step=config["copy-step"])