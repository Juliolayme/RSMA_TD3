"""
td3.py - Twin Delayed DDPG (TD3) cho toi uu RSMA

Tham khao truc tiep tu: folder 14 (td3.py)
Cai tien:
  - Code sach hon, bo hardcoded path
  - Giu nguyen kien truc 4-layer voi LayerNorm
  - Giu nguyen OUActionNoise va AWGNActionNoise
  - Giu nguyen ReplayBuffer, CriticNetwork, ActorNetwork, Agent
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# =============================================================================
# Noise Classes (giong folder 14)
# =============================================================================

class OUActionNoise:
    """Ornstein-Uhlenbeck noise cho exploration"""

    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class AWGNActionNoise:
    """Additive White Gaussian Noise cho exploration"""

    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        x = np.random.normal(size=self.mu.shape) * self.sigma
        return x


# =============================================================================
# Replay Buffer (giong folder 14)
# =============================================================================

class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# =============================================================================
# Critic Network - Twin Q-Networks (giong folder 14, 4-layer + LayerNorm)
# =============================================================================

class CriticNetwork(nn.Module):
    """
    Critic Network voi 4 hidden layers + LayerNorm
    Input: state -> 4 FC layers -> ket hop voi action -> Q-value
    """

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims,
                 n_actions, name, chkpt_dir='tmp/TD3'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_TD3')

        # State pathway: 4 layers
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        f3 = 1. / np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(fc3_dims)

        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        f4 = 1. / np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(fc4_dims)

        # Action pathway
        self.action_value = nn.Linear(n_actions, fc4_dims)

        # Output Q-value
        self.q = nn.Linear(fc4_dims, 1)
        f5 = 0.003
        T.nn.init.uniform_(self.q.weight.data, -f5, f5)
        T.nn.init.uniform_(self.q.bias.data, -f5, f5)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = self.bn3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc4(state_value)
        state_value = self.bn4(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_file=''):
        print('... loading checkpoint ...')
        if T.cuda.is_available():
            self.load_state_dict(T.load(load_file))
        else:
            self.load_state_dict(T.load(load_file, map_location=T.device('cpu')))


# =============================================================================
# Actor Network (giong folder 14, 4-layer + LayerNorm + tanh output)
# =============================================================================

class ActorNetwork(nn.Module):
    """
    Actor Network: state -> 4 FC layers -> tanh -> action ∈ [-1, 1]
    """

    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims,
                 n_actions, name, chkpt_dir='tmp/TD3'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_TD3')

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        f3 = 1. / np.sqrt(self.fc3.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)
        self.bn3 = nn.LayerNorm(fc3_dims)

        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        f4 = 1. / np.sqrt(self.fc4.weight.data.size()[0])
        self.fc4.weight.data.uniform_(-f4, f4)
        self.fc4.bias.data.uniform_(-f4, f4)
        self.bn4 = nn.LayerNorm(fc4_dims)

        f5 = 0.003
        self.mu = nn.Linear(fc4_dims, n_actions)
        self.mu.weight.data.uniform_(-f5, f5)
        self.mu.bias.data.uniform_(-f5, f5)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_file=''):
        print('... loading checkpoint ...')
        if T.cuda.is_available():
            self.load_state_dict(T.load(load_file))
        else:
            self.load_state_dict(T.load(load_file, map_location=T.device('cpu')))


# =============================================================================
# TD3 Agent (giong folder 14, voi twin critics va delayed policy update)
# =============================================================================

class Agent:
    """
    TD3 Agent voi:
    - Twin Critics (2 Q-networks) de giam overestimation
    - Delayed Policy Update (cap nhat Actor cham hon Critic)
    - Target Policy Smoothing (them noise vao target action)
    """

    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000,
                 layer1_size=400, layer2_size=300, layer3_size=256, layer4_size=128,
                 batch_size=64, update_actor_interval=2,
                 noise='AWGN', agent_name='default',
                 # FIX #4: Target policy smoothing params
                 target_noise_std=0.2, target_noise_clip=0.5):
        self.gamma = gamma
        self.tau = tau
        # FIX #4: Target policy smoothing (TD3 paper Section 5.3)
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.update_actor_iter = update_actor_interval

        chkpt_dir = os.path.join('tmp', 'TD3')
        os.makedirs(chkpt_dir, exist_ok=True)

        # Actor va Target Actor
        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, layer3_size, layer4_size,
                                  n_actions=n_actions,
                                  name='Actor_' + agent_name,
                                  chkpt_dir=chkpt_dir)

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, layer3_size, layer4_size,
                                         n_actions=n_actions,
                                         name='TargetActor_' + agent_name,
                                         chkpt_dir=chkpt_dir)

        # Twin Critics va Target Critics
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, layer3_size, layer4_size,
                                      n_actions=n_actions,
                                      name='Critic_1_' + agent_name,
                                      chkpt_dir=chkpt_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, layer3_size, layer4_size,
                                      n_actions=n_actions,
                                      name='Critic_2_' + agent_name,
                                      chkpt_dir=chkpt_dir)

        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, layer3_size, layer4_size,
                                             n_actions=n_actions,
                                             name='TargetCritic_1_' + agent_name,
                                             chkpt_dir=chkpt_dir)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, layer3_size, layer4_size,
                                             n_actions=n_actions,
                                             name='TargetCritic_2_' + agent_name,
                                             chkpt_dir=chkpt_dir)

        # Noise
        if noise == 'OU':
            self.noise = OUActionNoise(mu=np.zeros(n_actions))
        elif noise == 'AWGN':
            self.noise = AWGNActionNoise(mu=np.zeros(n_actions))

        # Copy tham so sang target networks (tau=1 la copy hoan toan)
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, greedy=0.5):
        """
        Chon action: Actor output + noise * greedy factor

        Args:
            observation: state hien tai
            greedy: he so noise (giam dan theo episode de exploit nhieu hon)
        """
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(greedy * self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        """Luu transition vao replay buffer"""
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        """
        Cap nhat TD3:
        1) Sample mini-batch tu replay buffer
        2) Tinh target Q = r + gamma * min(Q1_target, Q2_target)
        3) Cap nhat ca 2 Critics
        4) Moi update_actor_iter buoc, cap nhat Actor va soft-update targets
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)

        # Eval mode cho target networks
        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self.critic_1.eval()
        self.critic_2.eval()

        # FIX #4: Target Policy Smoothing - them noise vao target actions
        # Day la feature QUAN TRONG cua TD3 (paper: "Target Policy Smoothing Regularization")
        # Ngan critic overfit vao deterministic target action
        target_actions = self.target_actor.forward(new_state)
        target_noise = T.clamp(
            T.randn_like(target_actions) * self.target_noise_std,
            -self.target_noise_clip, self.target_noise_clip
        ).to(self.critic_1.device)
        target_actions = T.clamp(target_actions + target_noise, -1.0, 1.0)

        # Twin target Q-values
        critic_value_1_ = self.target_critic_1.forward(new_state, target_actions)
        critic_value_2_ = self.target_critic_2.forward(new_state, target_actions)

        # Current Q-values
        critic_value_1 = self.critic_1.forward(state, action)
        critic_value_2 = self.critic_2.forward(state, action)

        # Lay min cua 2 target critics (chong overestimation)
        critic_value_ = T.min(critic_value_1_, critic_value_2_)

        # Tinh target
        target = reward.unsqueeze(1) + self.gamma * critic_value_ * done.unsqueeze(1)
        target = target.detach()

        # Update Critics
        self.critic_1.train()
        self.critic_2.train()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        critic_1_loss = F.mse_loss(target, critic_value_1)
        critic_2_loss = F.mse_loss(target, critic_value_2)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # Delayed Actor Update
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.critic_1.eval()
        self.critic_2.eval()

        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_q1_loss = self.critic_1.forward(state, mu)
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # Soft update target networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """Soft update: target = tau * online + (1-tau) * target"""
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + \
                (1 - tau) * target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
                (1 - tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        """Luu tat ca models"""
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self, load_file_actor='', load_file_critic_1='', load_file_critic_2=''):
        """Load models tu file"""
        self.actor.load_checkpoint(load_file=load_file_actor)
        self.target_actor.load_checkpoint(load_file=load_file_actor)
        self.critic_1.load_checkpoint(load_file=load_file_critic_1)
        self.critic_2.load_checkpoint(load_file=load_file_critic_2)
        self.target_critic_1.load_checkpoint(load_file=load_file_critic_1)
        self.target_critic_2.load_checkpoint(load_file=load_file_critic_2)
