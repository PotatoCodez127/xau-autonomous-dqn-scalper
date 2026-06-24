# test_dqn.py
import torch

from env_trading import XAUUSDEnvironment
from model_dqn import TradingBrain
from train_dqn import train_agent


def evaluate_rl_bot():
    print("Initializing RL Sandbox Training...")
    train_agent()

    # Load trained brain
    model = TradingBrain()
    model.load_state_dict(torch.load("rl_scalper.pth"))
    model.eval()  # Turn off training features

    # Create a BRAND NEW, unseen market sequence
    test_env = XAUUSDEnvironment(data_length=100)
    state = test_env.reset()

    total_profit = 0
    trade_log = []

    print("\n--- Live Execution Diagnostics (Unseen Data) ---")

    while True:
        state_tensor = torch.FloatTensor(state)

        # 100% Exploitation (No random guesses)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        current_price = state[0]
        action_name = ["Close/Hold", "Buy", "Sell"][action]

        next_state, reward, done = test_env.step(action)

        if action != 0:  # Log only active entries
            trade_log.append(f"Price: ${current_price:.2f} | Action: {action_name}")

        total_profit += reward
        state = next_state

        if done:
            break

    print(f"Total Trades Executed: {len(trade_log)}")
    print("Sample of last 5 decisions:")
    for log in trade_log[-5:]:
        print(f"  • {log}")
    print(f"\nFinal Net Profit/Reward: {total_profit:.2f}")

    if total_profit > 0:
        print("✅ RL Agent Successfully Extracted Yield from Unseen Market.")
    else:
        print("❌ RL Agent Failed to Outperform Choppy Market Conditions.")


if __name__ == "__main__":
    evaluate_rl_bot()
