# env_trading.py

import numpy as np


class XAUUSDEnvironment:
    def __init__(self, data_length=1000):
        # Simulate a trending gold market with noise
        time = np.linspace(0, 50, data_length)
        self.prices = np.sin(time) * 10 + time + 1900 + np.random.normal(0, 2, data_length)
        self.current_step = 0
        self.max_steps = data_length - 1
        self.position = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        return self._get_state()

    def _get_state(self):
        # The bot sees the current price and its current position
        return np.array([self.prices[self.current_step], self.position], dtype=np.float32)

    def step(self, action):
        # Actions: 0 = Hold/Close, 1 = Buy, 2 = Sell
        reward = 0
        current_price = self.prices[self.current_step]

        # Calculate reward based on previous positions closing
        if action == 0 and self.position != 0:  # Closing a position
            if self.position == 1:
                reward = current_price - self.entry_price
            elif self.position == -1:
                reward = self.entry_price - current_price
            self.position = 0

        elif action == 1 and self.position <= 0:  # Going Long
            if self.position == -1:  # Close short first
                reward = self.entry_price - current_price
            self.position = 1
            self.entry_price = current_price

        elif action == 2 and self.position >= 0:  # Going Short
            if self.position == 1:  # Close long first
                reward = current_price - self.entry_price
            self.position = -1
            self.entry_price = current_price

        # Time penalty to discourage holding forever
        if self.position != 0:
            reward -= 0.1

        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_state = self._get_state()

        return next_state, reward, done
