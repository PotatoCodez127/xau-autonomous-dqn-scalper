
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from env_trading import XAUUSDEnvironment
from model_dqn import TradingBrain

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)

def train_agent():
    # Environment and Model initialization
    env = XAUUSDEnvironment(data_length=500)
    
    # Decoupled Networks
    online_model = TradingBrain()
    target_model = TradingBrain()
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()  # Target network strictly functions in evaluation mode
    
    optimizer = optim.Adam(online_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(capacity=5000)
    
    # Hyperparameters
    epochs = 100
    batch_size = 32
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.95
    epsilon_min = 0.05
    target_update_frequency = 5  # Synchronize target network every N episodes
    
    print("Training Stabilized DQN Agent in Sandbox...")
    
    for epoch in range(epochs):
        state = env.reset()
        total_reward = 0
        
        while True:
            # 1. Exploration vs Exploitation
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = online_model(state_tensor)
                    action = torch.argmax(q_values).item()
            
            # 2. Step Environment
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. Memory Preservation
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            # 4. Batch Optimization Optimization Step
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                
                # Compute current Q values
                current_q_values = online_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute stable target Q values using target network
                with torch.no_grad():
                    max_next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
                
                # Optimize loss
                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
                
        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Target Network Synchronization Layer
        if (epoch + 1) % target_update_frequency == 0:
            target_model.load_state_dict(online_model.state_dict())
            
        if (epoch + 1) % 20 == 0:
            print(f"Episode {epoch+1}/{epochs} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")

    torch.save(online_model.state_dict(), "rl_scalper.pth")
    return online_model

if __name__ == "__main__":
    train_agent()