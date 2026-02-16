import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # State is stored as uint8 to save memory, convert to float only when sampling
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_shape, num_actions, learning_rate=1e-4, gamma=0.99, buffer_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

        self.gamma = gamma
        self.num_actions = num_actions

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) / 255.0
                q_value = self.policy_net(state)
                return q_value.argmax().item()
        else:
            return random.randrange(self.num_actions)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        state, action, reward, next_state, done = self.memory.sample(batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(self.device) / 255.0
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device) / 255.0
        action = torch.tensor(action).unsqueeze(1).to(self.device)
        reward = torch.tensor(reward).unsqueeze(1).to(self.device)
        done = torch.tensor(done).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(state)
        q_value = q_values.gather(1, action)

        # Q_target = r + gamma * max(Q(s', a'))
        with torch.no_grad():
            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Loaded model from {filepath}")
        else:
            print(f"No model found at {filepath}")
