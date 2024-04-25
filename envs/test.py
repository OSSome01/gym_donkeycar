import numpy as np
import torch as th
from stable_baselines3 import PPO, SAC, DDPG
from sb3_contrib import TQC

from gym_donkeycar.envs.donkey_env import DonkeyEnv

# Load the saved model
model = SAC.load("logs/SAC_Cnn_nolaplimit_1e6steps_aggressive_withbrakes/best_model")

# Initialize the Donkey environment
env = DonkeyEnv("donkey-mountain-track-v0")  # Use the appropriate environment name

# Reset the environment
obs = env.reset()
num_laps = 0
i = 0
# Run the agent in the simulator for a specified number of steps
for _ in range(100000000):  # Adjust the number of steps as needed
    i += 1
    action, _ = model.predict(obs, deterministic=True)  # Use the model to predict actions
    obs, reward, done, info = env.step(action)

    # Optional: You can print or log the reward, done flag, or any other information
    # print(f"Reward: {reward}, Done: {done}, Info: {info}")
    # if info['lap_count'] == 10:
    #     print('i', i)
    #     break
    if done:
        print('i', i)
        obs = env.reset()  # Reset the environment if an episode is done

# Close the environment
env.close()

