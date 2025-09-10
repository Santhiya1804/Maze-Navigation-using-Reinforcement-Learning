import pybullet as p
import pybullet_data
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time

# === Hyperparameters ===
GAMMA = 0.95
LR = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
MEMORY_SIZE = 5000
BATCH_SIZE = 32
EPISODES = 20
MAX_STEPS = 200  # Allowing for more steps to explore

# === Maze Configuration ===
GRID_SIZE = 0.5
WIDTH = 10
HEIGHT = 10

def generate_maze():
    """Generates a complex maze structure."""
    walls = set()
    
    # Outer walls
    for i in range(WIDTH):
        walls.add((i, 0))  # Bottom
        walls.add((i, HEIGHT - 1))  # Top
    for j in range(HEIGHT):
        walls.add((0, j))  # Left
        walls.add((WIDTH - 1, j))  # Right

    # Internal maze walls
    internal_walls = [
        (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
        (5, 3), (5, 4), (5, 5), (4, 5), (3, 5), (2, 5), (1, 5),
        (7, 1), (7, 2), (7, 3), (7, 4),
        (3, 7), (4, 7), (5, 7), (6, 7),
        (6, 8), (6, 9), (7, 9)
    ]
    
    walls.update(internal_walls)
    return walls

# === Maze Environment ===
class MazeEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.width = WIDTH
        self.height = HEIGHT
        self.goal = (8, 8)

        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[3, 3, 0])

        self.robot = None
        self.wall_positions = generate_maze()
        self._create_walls()
        self.reset()

    def _create_walls(self):
        """Creates walls using bricks in a maze-like structure."""
        for pos in self.wall_positions:
            p.loadURDF("brick.urdf", basePosition=[pos[0] * self.grid_size, pos[1] * self.grid_size, 0.1])

    def reset(self):
        if self.robot:
            p.removeBody(self.robot)
        self.robot = p.loadURDF("r2d2.urdf", [0.5, 0.5, 0.1])
        return self.get_state()

    def step(self, action):
        dx, dy = actions[action]
        x, y, z = p.getBasePositionAndOrientation(self.robot)[0]
        new_x = x + dx
        new_y = y + dy

        # Prevent out-of-bounds movement
        new_x = np.clip(new_x, 0, (self.width - 1) * self.grid_size)
        new_y = np.clip(new_y, 0, (self.height - 1) * self.grid_size)

        # Collision Detection
        if (round(new_x / GRID_SIZE), round(new_y / GRID_SIZE)) in self.wall_positions:
            reward = -5  # Penalty for hitting a wall
            print(f"❌ Penalty of -5: The robot hit a wall at ({round(new_x, 1)}, {round(new_y, 1)})")
            done = False
        else:
            p.resetBasePositionAndOrientation(self.robot, [new_x, new_y, z], [0, 0, 0, 1])
            p.stepSimulation()
            time.sleep(0.1)
            reward = -1  # Small penalty for each step
            print(f"🔹 Step taken: -1 penalty for moving to ({round(new_x, 1)}, {round(new_y, 1)})")
            done = False

        # Check if goal is reached
        if (round(new_x, 1), round(new_y, 1)) == self.goal:
            reward = 10
            print(f"🎉 Goal Reached! +10 reward at ({round(new_x, 1)}, {round(new_y, 1)})")
            done = True

        return self.get_state(), reward, done

    def get_state(self):
        x, y, _ = p.getBasePositionAndOrientation(self.robot)[0]
        return np.array([x, y])

# === Action Space ===
actions = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]

# === Define Neural Network for DQN ===
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === Experience Replay Memory ===
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# === DQN Agent ===
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.epsilon = EPSILON

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (GAMMA * next_q * (1 - dones))
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

# === Train the Agent ===
env = MazeEnv()
state_size = 2
action_size = len(actions)
agent = DQNAgent(state_size, action_size)

try:
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        print(f"\n🎬 Episode {episode+1}")

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.replay()
        agent.decay_epsilon()
except KeyboardInterrupt:
    print("\n🛑 Training manually stopped!")

p.disconnect()