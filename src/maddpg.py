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

# conditional imports
try:
    import torch
    from torch.distributions import Normal
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")


class MADDPG(RLC):
    def __init__(self, num_agents, agent_observation_size, agent_action_size, actor_units: list = [256, 128], critic_units: list = [256, 128],
                 buffer_size: int = int(1e5), batch_size: int = 128, gamma: float = 0.99, sigma=0.2,
                 lr_actor: float = 1e-5, lr_critic: float = 1e-4, decay_factor=0.96, tau=1e-3, *args, **kwargs):

        super().__init__(**kwargs)

        # Retrieve number of agents
        self.num_agents = num_agents

        # Discount factor for the MDP
        self.gamma = gamma

        # Replay buffer and batch size
        self.replay_buffer = ReplayBuffer1(capacity=buffer_size, num_agents=self.num_agents)
        self.batch_size = batch_size

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("ATTENTION HERE")
        print(self.device)

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
                self.device) for i in range(len(self.num_agents))
        ]
        self.critics = [
            Critic(sum(self.agent_observation_size), sum(self.agent_action_size), self.seed, critic_units).to(
                self.device) for _ in range(len(self.num_agents))
        ]

        # Initialize target networks
        self.actors_target = [
            Actor(self.agent_observation_size, self.agent_action_size, self.seed, actor_units).to(
                self.device) for i in range(len(self.num_agents))
        ]
        self.critics_target = [
            Critic(sum(self.agent_observation_size), sum(self.agent_action_size), self.seed, critic_units).to(
                self.device) for _ in range(len(self.num_agents))
        ]

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_actor) for i in
                                 range(len(self.num_agents))]
        self.critics_optimizer = [torch.optim.Adam(self.critics[i].parameters(), lr=lr_critic) for i in
                                  range(len(self.num_agents))]

        self.noise = [OUNoise(self.action_space[i].shape[0], self.seed, sigma=sigma, decay_factor=decay_factor) for i in range(len(self.action_space))]

        self.scaler = GradScaler()
        self.exploration_done = False

    def update(self, observations, actions, reward, next_observations, done):
        self.replay_buffer.push(observations, actions, reward, next_observations, done)

        if len(self.replay_buffer) < self.batch_size:
            print("returned due to buffer")
            return

        print("training")
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.replay_buffer.sample(
            self.batch_size)

        obs_tensors = []
        next_obs_tensors = []
        actions_tensors = []
        reward_tensors = []
        dones_tensors = []

        for agent_num in range(len(self.num_agents)):
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
            self.scaler.step(critic_optim)
            critic_optim.zero_grad()
            self.scaler.update()

            with autocast():
                # Update actor
                predicted_actions = [self.actors[i](obs_tensors[i]) for i in range(self.num_agents)]
                predicted_actions_full = torch.cat(predicted_actions, dim=1)
                actor_loss = -critic(obs_full, predicted_actions_full).mean()

            self.scaler.scale(actor_loss).backward()
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
            to_return = [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                    for actor, obs in zip(self.actors, encoded_observations)]
            return to_return

    def predict(self, observations, deterministic=False):
        actions_return = None
        if self.time_step > self.end_exploration_time_step or deterministic:
            if deterministic:
                actions_return = self.get_deterministic_actions(observations)
            else:
                actions_return = self.get_exploration_prediction(observations)
        else:
            actions_return = self.get_exploration_prediction(observations)

        return actions_return

    def predict_deterministic(self, encoded_observations):
        actions_return = None
        with torch.no_grad():
            actions_return = [actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                    for actor, obs in zip(self.actors, encoded_observations)]
        return actions_return

    def get_exploration_prediction(self, states: List[List[float]]) -> List[float]:
        deterministic_actions = self.get_deterministic_actions(states)

        random_noises = [noise.sample() for noise in self.noise]

        actions = [action + noise for action, noise in zip(deterministic_actions, random_noises)]
        clipped_actions = [np.clip(action, -1, 1) for action in actions]
        actions_return = [action.tolist() for action in clipped_actions]

        return actions_return

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(np.array(observations, dtype=float)) if j != None], dtype = float)

    def reset(self):
        super().reset()

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
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
        return torch.tanh(self.fc_layers[-1](x))


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
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.25, decay_factor=0.005):
        ...
        self.decay_factor = decay_factor
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.internal_state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.internal_state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.internal_state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for _ in range(len(x))])
        self.internal_state = x + dx
        #self.sigma *= self.decay_factor
        return self.internal_state

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
            state.append(np.stack(state_i))
            action.append(np.stack(action_i))
            reward.append(np.stack(reward_i))
            next_state.append(np.stack(next_state_i))
            done.append(np.stack(done_i))

        return state, action, reward, next_state, done

    def __len__(self):
        return min(len(self.buffer[i]) for i in range(self.num_agents))







