# train_dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from env_trading import XAUUSDEnvironment
from model_dqn import TradingBrain

def train_agent():
    env = XAUUSDEnvironment(data_length=500)
    model = TradingBrain()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 100
    gamma = 0.95 # Discount factor for future rewards
    epsilon = 1.0 # Start 100% random
    epsilon_decay = 0.95
    epsilon_min = 0.05
    
    print("Training RL Agent in Sandbox...")
    
    for epoch in range(epochs):
        state = env.reset()
        state_tensor = torch.FloatTensor(state)
        total_reward = 0
        
        while True:
            # 1. EXPLORATION vs EXPLOITATION
            if random.random() < epsilon:
                action = random.randint(0, 2) # Random guess
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item() # Best calculated guess
                    
            # 2. TAKE ACTION IN ENVIRONMENT
            next_state, reward, done = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state)
            total_reward += reward
            
            # 3. BELLMAN EQUATION MATH
            # Q_target = Reward + Gamma * Max(Next State Q-Values)
            with torch.no_grad():
                next_q_values = model(next_state_tensor)
                max_next_q = torch.max(next_q_values)
                target_q = reward + (gamma * max_next_q * (1 - int(done)))
            
            # 4. NETWORK UPDATE
            current_q_values = model(state_tensor)
            # We only update the Q-value for the action we actually took
            target_q_values = current_q_values.clone()
            target_q_values[action] = target_q
            
            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state_tensor = next_state_tensor
            
            if done:
                break
                
        # Decay Epsilon so the bot slowly stops acting randomly
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (epoch + 1) % 20 == 0:
            print(f"Episode {epoch+1}/{epochs} | Total Reward (Profit): {total_reward:.2f} | Epsilon: {epsilon:.2f}")

    torch.save(model.state_dict(), "rl_scalper.pth")
    return model

if __name__ == "__main__":
    train_agent()