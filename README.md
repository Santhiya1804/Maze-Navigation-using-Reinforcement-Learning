# Maze Navigation Using Reinforcement Learning (DQN)




##  Methodology

### 1 Maze Environment Setup
- Maze defined on a **10×10 grid**, each cell representing **0.5m × 0.5m** space.
- **Robot:** R2D2 loaded in PyBullet (physics-based interactions).
- **Walls:** Strategically placed to form a challenging path.
- **Goal Position:** (8, 8).  
- **Start Position:** (0.5, 0.5).

### 2 Reinforcement Learning Approach

#### State Representation
- Robot’s current position `(x, y)`.
- Relative distance to the goal.

#### Action Space
The robot can move in four discrete directions:
- Right `( +0.5,  0 )`
- Left  `( -0.5,  0 )`
- Up    `(  0, +0.5 )`
- Down  `(  0, -0.5 )`

#### Reward Function
| Event            | Reward |
|------------------|--------|
| Reaching the goal| +10    |
| Hitting a wall   | -5     |
| Each step taken  | -1     |

### 3 Deep Q-Network (DQN) Implementation
- **Neural Network:** Three-layer fully connected network with ReLU activation.
- **Experience Replay:** Stores past experiences to improve learning stability.
- **Epsilon-Greedy Strategy:** Starts with random actions and gradually shifts to policy-based actions.
- **Target Network Update:** Stabilizes the learning process.

### 4 Training Process
- Agent trained for **20 episodes**, each with a **maximum of 200 steps**.
- **Exploration–Exploitation trade-off** handled using epsilon decay.
- The robot learns to avoid walls and reach the goal through trial and error.

---

##  Results and Analysis

### Observations During Training
- Initially, the robot explores randomly and frequently hits walls.
- Over episodes, Q-values update and the robot starts choosing better actions.
- By the final episodes, the robot efficiently reaches the goal with fewer steps and minimal penalties.

### Performance Metrics
- **Success Rate:** Percentage of episodes where the robot reaches the goal.
- **Average Steps to Goal:** Measures navigation efficiency.
- **Reward Accumulation:** Tracks learning progress.

---
