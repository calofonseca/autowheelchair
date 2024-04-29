from typing import Any, List
import numpy as np
import torch
import torch.nn.functional as F
import random
import numpy.typing as npt
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
from collections import namedtuple, deque
import copy
from base import BaseClass as RL
# conditional imports
try:
    import torch
    from torch.distributions import Normal
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")


class MADDPG(RL):
    def __init__(self, num_agents, agent_observation_size, agent_action_size, actor_units: list = [512, 256, 128, 64, 32], critic_units: list = [1024, 512, 256, 128, 64, 32],
                 buffer_size: int = int(5e5), batch_size: int = 128, gamma: float = 0.95, sigma=0.24,
                 lr_actor: float = 1e-5, lr_critic: float = 1e-4, decay_factor=0.995, tau=1e-2, *args, **kwargs):

        super().__init__(**kwargs)

        # Retrieve number of agents
        self.num_agents = num_agents

        # Discount factor for the MDP
        self.gamma = gamma
        self.decay_factor = decay_factor
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon = 1.0  # Initial value of epsilon
        self.decay_rate = 4.61e-5#9.21e-5 #4.61×10−6  # The decay rate we calculated earlier
        self.min_epsilon = 0.05  # Minimum value of epsilon

        # Replay buffer and batch size
        self.replay_buffer = ReplayBuffer1(capacity=buffer_size, num_agents=self.num_agents)
        self.batch_size = batch_size

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = random.randint(0, 100_000_000)
        self.actor_units = actor_units
        self.critic_units = critic_units
        self.tau = tau
        self.sigma = sigma
        self.agent_observation_size = agent_observation_size
        self.agent_action_size = agent_action_size

        # Initialize actors and critics
        self.actors = [
            Actor(self.agent_observation_size, self.agent_action_size, self.seed, actor_units).to(
                self.device) for i in range(self.num_agents)
        ]
        self.critics = [
            Critic(self.agent_observation_size*self.num_agents, self.agent_action_size*self.num_agents, self.seed, critic_units).to(
                self.device) for i in range(self.num_agents)
        ]

        # Initialize target networks
        self.actors_target = [
            Actor(self.agent_observation_size, self.agent_action_size, self.seed, actor_units).to(
                self.device) for i in range(self.num_agents)
        ]
        self.critics_target = [
            Critic(self.agent_observation_size*self.num_agents, self.agent_action_size*self.num_agents, self.seed, critic_units).to(
                self.device) for i in range(self.num_agents)
        ]

        self.actors_optimizer = [torch.optim.Adam(actor.parameters(), lr=lr_actor, weight_decay=1e-4) for actor in self.actors]
        self.critics_optimizer = [torch.optim.Adam(critic.parameters(), lr=lr_critic, weight_decay=1e-4) for critic in self.critics]

        # Noise process
        self.noise = [OUNoise(size=agent_action_size, sigma=self.sigma, seed=self.seed) for _ in range(num_agents)]

        self.scaler = GradScaler()
        self.exploration_done = False

    def update(self, observations, actions, reward, next_observations, done):

        self.replay_buffer.push(observations, actions, reward, next_observations, done)

        if len(self.replay_buffer) < self.batch_size:
            print("returned due to buffer")
            return

        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.replay_buffer.sample(
            self.batch_size)

        obs_tensors = []
        next_obs_tensors = []
        actions_tensors = []
        reward_tensors = []
        dones_tensors = []

        for agent_num in range(self.num_agents):
            obs_tensors.append(
                torch.stack([torch.FloatTensor(self.get_encoded_observations(agent_num, obs)).to(self.device)
                             for obs in obs_batch[agent_num]]))
            next_obs_tensors.append(
                torch.stack([torch.FloatTensor(self.get_encoded_observations(agent_num, next_obs)).to(self.device)
                             for next_obs in next_obs_batch[agent_num]]))
            actions_tensors.append(
                torch.stack([torch.FloatTensor(action).to(self.device)
                             for action in actions_batch[agent_num]]))
            reward_tensors.append(
                torch.tensor(rewards_batch[agent_num], dtype=torch.float32).to(self.device).view(-1, 1))
            dones_tensors.append(torch.tensor(dones_batch[agent_num], dtype=torch.float32).to(self.device).view(-1, 1))

        obs_full = torch.cat(obs_tensors, dim=1)
        next_obs_full = torch.cat(next_obs_tensors, dim=1)
        action_full = torch.cat(actions_tensors, dim=1)

        for agent_num, (actor, critic, actor_target, critic_target, actor_optim, critic_optim) in enumerate(
                zip(self.actors, self.critics, self.actors_target, self.critics_target, self.actors_optimizer,
                    self.critics_optimizer)):

            with autocast():
                # Update critic
                Q_expected = critic(obs_full, action_full)
                next_actions = [self.actors_target[i](next_obs_tensors[i]) for i in range(self.num_agents)]
                next_actions_full = torch.cat(next_actions, dim=1)
                Q_targets_next = critic_target(next_obs_full, next_actions_full)
                Q_targets = reward_tensors[agent_num] + (self.gamma * Q_targets_next * (1 - dones_tensors[agent_num]))
                critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

            self.scaler.scale(critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1)
            self.scaler.step(critic_optim)
            critic_optim.zero_grad()
            self.scaler.update()

            with autocast():
                # Update actor
                predicted_actions = [self.actors[i](obs_tensors[i]) for i in range(self.num_agents)]
                predicted_actions_full = torch.cat(predicted_actions, dim=1)
                actor_loss = -critic(obs_full, predicted_actions_full).mean()

            self.scaler.scale(actor_loss).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1)
            self.scaler.step(actor_optim)
            actor_optim.zero_grad()
            self.scaler.update()
            
            # Update target networks
            self.soft_update(critic, critic_target, self.tau)
            self.soft_update(actor, actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_deterministic_actions(self, observations):
        with torch.no_grad():
            encoded_observations = [self.get_encoded_observations(i, obs) for i, obs in enumerate(observations)]
            actions = [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                   for actor, obs in zip(self.actors, encoded_observations)]
            print("Deterministic Actions:")
            print(actions)
            return actions

    def get_exploration_prediction(self, states: List[List[float]], step) -> List[float]:
        deterministic_actions = self.get_deterministic_actions(states)
        noisy_actions = []
        # Loop over each action and corresponding noise generator
        for idx, action in enumerate(deterministic_actions):
            # Retrieve the correct noise generator for the current agent
            noise = self.noise[idx].sample()
            # Add noise to the deterministic action
            noisy_action = action + noise
            noisy_actions.append(noisy_action)

        print("Exploration Actions with Noise:")
        print(noisy_actions)
        return noisy_actions

    def predict(self, observations, step, deterministic=False):
        if deterministic:
            actions = self.get_deterministic_actions(observations)
        else:
            actions = self.get_exploration_prediction(observations, step)
        return actions
   
    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(np.array(observations, dtype=float)) if j != None], dtype = float)

    def reset(self):
        super().reset()
        for noise in self.noise:
            noise.reset()

    def save_weights(self, filename_prefix):
        """Save all network weights."""
        for idx, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f'{filename_prefix}_actor_{idx}.pth')
        for idx, critic in enumerate(self.critics):
            torch.save(critic.state_dict(), f'{filename_prefix}_critic_{idx}.pth')
        for idx, actor_target in enumerate(self.actors_target):
            torch.save(actor_target.state_dict(), f'{filename_prefix}_actor_target_{idx}.pth')
        for idx, critic_target in enumerate(self.critics_target):
            torch.save(critic_target.state_dict(), f'{filename_prefix}_critic_target_{idx}.pth')

    def load_weights(self, filename_prefix):
        """Load all network weights."""
        for idx, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(f'{filename_prefix}_actor_{idx}.pth'))
        for idx, critic in enumerate(self.critics):
            critic.load_state_dict(torch.load(f'{filename_prefix}_critic_{idx}.pth'))
        for idx, actor_target in enumerate(self.actors_target):
            actor_target.load_state_dict(torch.load(f'{filename_prefix}_actor_target_{idx}.pth'))
        for idx, critic_target in enumerate(self.critics_target):
            critic_target.load_state_dict(torch.load(f'{filename_prefix}_critic_target_{idx}.pth'))

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[256, 128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list): List of node counts in the hidden layers.
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.max_action = 1

        # Input layer
        self.fc_layers = [nn.Linear(state_size, fc_units[0])]

        # Intermediate layers
        for i in range(1, len(fc_units)):
            self.fc_layers.append(nn.Linear(fc_units[i - 1], fc_units[i]))

        # Output layer
        self.fc_layers.append(nn.Linear(fc_units[-1], action_size))

        # ModuleList to register the layers with PyTorch
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, state):
        x = state
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
        x = self.fc_layers[-1](x)
        return torch.tanh(x) * self.max_action

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[256, 128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list): List of node counts in the hidden layers.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Initial layer
        self.fc1 = nn.Linear(state_size, fc_units[0])

        # Concatenation layer (adding action_size to the width)
        self.fc2 = nn.Linear(fc_units[0] + action_size, fc_units[1] if len(fc_units) > 1 else 1)

        # Additional layers if any
        self.fc_layers = []
        for i in range(1, len(fc_units) - 1):
            self.fc_layers.append(nn.Linear(fc_units[i], fc_units[i + 1]))

        # If there are more than 2 fc_units, the last fc_layer will output the Q-value.
        # Otherwise, fc2 is responsible for that.
        if len(fc_units) > 2:
            self.fc_layers.append(nn.Linear(fc_units[-1], 1))

        # ModuleList to register the layers with PyTorch
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))

        # Concatenate the action values with the output from the previous layer
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))

        for fc in self.fc_layers:
            x = F.relu(fc(x))

        return x


class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, sigma_decay=0.99):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        self.sigma_decay = sigma_decay

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def decay_sigma(self):
        self.sigma *= self.sigma_decay

import numpy as np
from collections import deque
import random

class ReplayBuffer1:
    def __init__(self, capacity, num_agents):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = [deque(maxlen=capacity) for _ in range(num_agents)]

    def push(self, state, action, reward, next_state, done):
        
        for i in range(self.num_agents):
            self.buffer[i].append((state[i], action[i], reward[i], next_state[i], done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = [], [], [], [], []
        for i in range(self.num_agents):
            # For each agent, get a batch of experiences
            batch = random.sample(self.buffer[i], batch_size)

            # For each agent's batch, separate the experiences into state, action, reward, next_state, done
            state_i, action_i, reward_i, next_state_i, done_i = zip(*batch)

                    # Print the shapes of arrays
            
            # Print the shapes of individual elements within state_i
            #print("Individual shapes for agent", i, "state_i:")
            #for j, s in enumerate(action_i):
            #    print(f"State {j} shape:", np.array(s))

            state.append(np.stack(state_i))
            action.append(np.stack(action_i))
            reward.append(np.stack(reward_i))
            next_state.append(np.stack(next_state_i))
            done.append(np.stack(done_i))

        return state, action, reward, next_state, done

    def __len__(self):
        return min(len(self.buffer[i]) for i in range(self.num_agents))







