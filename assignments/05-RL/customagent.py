import gymnasium as gym
import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Agent:
#     def __init__(
#         self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box,
#         learning_rate: float = 0.1, discount_factor: float = 0.99, epsilon: float = 1.0,
#         epsilon_decay: float = 0.99, min_epsilon: float = 0.01
#     ):
#         self.action_space = action_space
#         self.observation_space = observation_space
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.min_epsilon = min_epsilon
#         self.q_table = np.zeros((observation_space.shape[0], self.action_space.n))

#     def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
#         if np.random.rand() < self.epsilon:
#             # Choose a random action
#             return self.action_space.sample()
#     # Choose the action with the highest Q-value
#         observation_idx = observation.astype(int)
#         q_values = self.q_table[observation_idx]
#         max_action = np.argmax(q_values)
#         return np.random.choice(max_action)


#     def learn(
#         self,
#         observation: gym.spaces.Box,
#         reward: float,
#         terminated: bool,
#         truncated: bool,
#     ) -> None:

#         observation_idx = observation.astype(int)
#         updated_q_values = self.q_table[observation_idx]

#         if terminated or truncated:
#             target = reward
#         else:
#             next_observation_idx = observation.astype(int)
#             next_q_values = self.q_table[next_observation_idx]
#             target = reward + self.discount_factor * np.max(next_q_values)

#         updated_q_values[self.last_action] += self.learning_rate * (target - updated_q_values[self.last_action])
#         self.q_table[observation_idx] = updated_q_values


#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

# # update internal state
# self.this_action = None if terminated else self.act(observation)
# self.last_action = None if terminated else self.this_action

# ===========================================================================

# class Agent:
#     '''
#     a customized agent that performs deep learning for the lunar landing problem
#     '''
#     def __init__(self, action_space, observation_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay_rate=0.99):
#         self.action_space = action_space
#         self.observation_space = observation_space
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay_rate = exploration_decay_rate
#         self.q_table = np.zeros((self.observation_space.shape[0], self.action_space.n))
#         self.last_action = None
#         self.last_observation = None

#     def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
#         '''
#         makes an action according to the current observation
#         '''

#         if self.last_action is None:
#             action = self.action_space.sample() # take random action on first step
#         else:
#             exploration_rate = self.exploration_rate * self.exploration_decay_rate
#             if np.random.random() < exploration_rate:
#                 action = self.action_space.sample() # explore
#             else:
#                 state = np.argmax(observation)
#                 action = np.argmax(self.q_table[state]) # exploit
#         self.last_observation = observation
#         self.last_action = action
#         return action


#     def learn(
#         self,
#         observation: gym.spaces.Box,
#         reward: float,
#         terminated: bool,
#         truncated: bool,
#     ) -> None:
#         '''
#         learns how to obtain the most reward points with Q-learning
#         '''

#         state = np.argmax(self.last_observation)
#         next_state = np.argmax(observation)
#         action = self.last_action
#         next_action = self.act(observation) # need to know next action for next_state in Q update equation
#         td_error = reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state][action]
#         self.q_table[state][action] += self.learning_rate * td_error


# ========================================================================================


# DQN algo!!!

# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.tensor(state_batch, dtype=torch.float),
            torch.tensor(action_batch, dtype=torch.long),
            torch.tensor(reward_batch, dtype=torch.float),
            torch.tensor(next_state_batch, dtype=torch.float),
            torch.tensor(done_batch, dtype=torch.float),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


class Agent:
    """
    Custom agent for lunar landing problem
    """

    def __init__(
        self,
        action_space,
        observation_space,
        batch_size=64,
        gamma=0.99,
        epsilon=1.0,
        eps_decay=0.9995,
        eps_min=0.01,
        lr=1e-3,
        target_update=10,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.lr = lr
        self.target_update = target_update
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.policy_net = QNetwork(observation_space.shape[0], action_space.n).to(
            self.device
        )
        self.target_net = QNetwork(observation_space.shape[0], action_space.n).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        agent makes an action based on the current observation state
        """
        self.epsilon = max(self.eps_min, self.eps_decay * self.epsilon)
        self.steps_done += 1

        if np.random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(
                    np.array([observation]), dtype=torch.float, device=self.device
                )
                q_values = self.policy_net(state)
                _, action = torch.max(q_values, dim=1)
                action = int(action.item())

        self.last_action = action
        self.last_state = observation
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        agent learns how to score higher with DQN
        """
        self.replay_buffer.push(
            self.last_state, self.last_action, reward, observation, terminated
        )

        if len(self.replay_buffer) < self.batch_size:
            return

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(self.batch_size)
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device).view(-1, 1)
        # done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)
        q_pred = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(-1))
        q_target_actions = self.target_net(next_state_batch)

        next_q_values = self.target_net(next_state_batch).max(dim=1)[0].view(-1, 1)

        q_target = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = F.mse_loss(q_pred, q_target).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
