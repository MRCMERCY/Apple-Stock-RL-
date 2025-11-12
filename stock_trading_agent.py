import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

class StockTradingEnv(gym.Env):
    """
    Custom Gym environment for stock trading.
    Agent can buy, sell, or hold shares of a stock (e.g., AAPL).
    Goal: Maximize profit.
    """
    def __init__(self):
        super(StockTradingEnv, self).__init__()
        
        # Download historical stock data (AAPL as example)
        data = yf.download('AAPL', start='2020-01-01', end='2023-01-01', auto_adjust=True)
        self.stock_data = data['Close'].to_numpy()
        self.data_length = len(self.stock_data)
        
        # Initial state variables
        self.initial_balance = 10000.0
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        
        # Action space: 0 - hold, 1 - buy (all available balance), 2 - sell (all shares)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [current_price, balance, shares_held, net_worth]
        # Use low=0 for simplicity, as values are non-negative
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        # Reset environment to initial state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        return self._get_observation(), {}
    
    def step(self, action):
        current_price = self.stock_data[self.current_step].item()
        
        if action == 1:  # Buy
            if current_price > 0:
                shares_to_buy = self.balance / current_price
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0.0
        
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Reward: change in net worth (simple profit maximization)
        reward = self.net_worth - self.initial_balance
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= self.data_length - 1
        truncated = False  # No truncation in this env
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _get_observation(self):
        current_price = self.stock_data[self.current_step].item()
        return np.array([current_price, self.balance, self.shares_held, self.net_worth], dtype=np.float32)
    
    def render(self, mode='human'):
        # Optional: Print current state for debugging
        print(f"Step: {self.current_step}, Price: {self.stock_data[self.current_step].item():.2f}, "
              f"Balance: {self.balance:.2f}, Shares: {self.shares_held:.2f}, Net Worth: {self.net_worth:.2f}")

# Main function to train and test the agent
if __name__ == "__main__":
    # Create the environment
    env = make_vec_env(StockTradingEnv, n_envs=1, vec_env_cls=DummyVecEnv)
    
    # Initialize PPO model (you can tune hyperparameters)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    
    # Train the model
    print("Training the agent...")
    model.learn(total_timesteps=10000)  # Increase for better performance
    
    # Save the model (optional)
    model.save("ppo_stock_trading")
    
    # Test the trained agent
    print("\nTesting the agent...")
    obs = env.reset()
    total_reward = 0
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01', auto_adjust=True)
    for _ in range(len(data) - 1):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        # env.render()  # Uncomment to see step-by-step output
        if done:
            break
    print(f"Total reward (profit): {total_reward[0]:.2f}")