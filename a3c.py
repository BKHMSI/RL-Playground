import os
import yaml
import pickle
import argparse
import numpy as np

import torch as T
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 
from collections import namedtuple

import common.shared_optim as shared_optim
from common.atari_envs import create_atari_env
from common.weight_inits import weights_init, normalized_columns_initializer

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_actions)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))

        val_estimate = self.critic_linear(hx)
        action_logits = self.actor_linear(hx)

        return action_logits, val_estimate, (hx, cx)

    def get_init_states(self, device):
        h0 = T.zeros(1, self.lstm.hidden_size).to(device)
        c0 = T.zeros(1, self.lstm.hidden_size).to(device)
        return (h0, c0)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        

def train(config, shared_model, rank, optimizer):
    T.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    T.random.manual_seed(config["seed"] + rank)
    device = config["device"]

    print(f"> Running Trainer {rank}")
    env = create_atari_env(config["environment"])
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    agent.to(device)
    agent.train()

    if rank % 4 == 0:
        writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"] + f"_{rank}"))
    
    save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")
    save_interval = config["save-interval"]
    
    episode_length = 0
    episode_reward = 0
    update_counter = 0
    total_rewards = []

    state = env.reset()
    state = T.from_numpy(state)
    done = True

    while True:
        # Sync with the shared model
        agent.load_state_dict(shared_model.state_dict())
        if done:
            hx, cx = agent.get_init_states(device)
        else:
            hx, cx = hx.detach(), cx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for _ in range(config["num-steps"]):
            episode_length += 1

            logit, value, (hx, cx) = agent((state.unsqueeze(0).to(device), (hx, cx)))
                        
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies += [entropy]

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(int(action))
            done = done or episode_length >= config["max-episode-length"]
            reward = max(min(reward, 1), -1)

            episode_reward += reward

            state = T.from_numpy(state)
            log_probs += [log_prob]
            values += [value]
            rewards += [reward]

            if done:
                episode_length = 0
                state = T.from_numpy(env.reset())
                total_rewards += [episode_reward]

                if rank % 4 == 0:
                    avg_reward_100 = np.array(total_rewards[-100:]).mean()
                    writer.add_scalar("perf/reward_t", episode_reward, len(total_rewards))
                    writer.add_scalar("perf/avg_reward_100", avg_reward_100, len(total_rewards))
                
                episode_reward = 0
                if len(total_rewards) % save_interval == 0 and rank == 0:
                    T.save({
                        "state_dict": shared_model.state_dict(),
                        "avg_reward_100": avg_reward_100,
                    }, save_path.format(epi=len(total_rewards)) + ".pt")

                break

        R = T.zeros(1, 1).to(device)
        if not done:
            _, value, _ = agent((state.unsqueeze(0).to(device), (hx, cx)))
            R = value.detach()

        values += [R]
        policy_loss = 0
        value_loss = 0
        gae = T.zeros(1, 1).to(device)
        for i in reversed(range(len(rewards))):
            R = config["gamma"] * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + config["gamma"] * values[i + 1] - values[i]
            gae = gae * config["gamma"] * config["gae-lambda"] + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - config["entropy-coef"] * entropies[i]

        loss = policy_loss + config["val-loss-coef"] * value_loss
        optimizer.zero_grad()

        loss.backward()
        T.nn.utils.clip_grad_norm_(agent.parameters(), config["max-grad-norm"])

        ensure_shared_grads(agent, shared_model)
        optimizer.step()

        update_counter += 1
        if rank % 4 == 0:
            writer.add_scalar("losses/total_loss", loss.item(), update_counter)

if __name__ == "__main__":
    
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="configs/a3c.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    save_path = os.path.join(config["save-path"], config["run-title"])
    if not os.path.isdir(save_path): 
        os.mkdir(save_path)

    env = create_atari_env(config["environment"])
    
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    shared_model.share_memory()
    shared_model.to(config["device"])

    optimizer = shared_optim.SharedAdam(shared_model.parameters(), lr=config["init-lr"])
    optimizer.share_memory()
  
    processes = []

    T.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    T.random.manual_seed(config["seed"])

    if config["resume"]:
        filepath = os.path.join(
            config["save-path"], 
            config["run-title"], 
            f"{config['run-title']}_{config['start-episode']}.pt"
        )
        print(f"> Loading Checkpoint {filepath}")
        shared_model.load_state_dict(T.load(filepath)["state_dict"])

    for rank in range(config["num-workers"]):
        p = mp.Process(target=train, args=(config, shared_model, rank, optimizer))
        p.start()
        processes += [p]

    for p in processes:
        p.join()



    