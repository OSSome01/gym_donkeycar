import os
import numpy as np
import torch as th
import gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.logger import configure

from gym_donkeycar.envs.donkey_env import DonkeyEnv 

# Define a function to create the Donkey environment
def make_env():
    env = DonkeyEnv("donkey-mountain-track-v0")  # You can choose any Donkey environment here
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=10.0)
    return env

# Create the environment
env = make_env()

tmp_path = "logs/TD3_Cnn_nolaplimit"
# set up logger
new_logger = configure(tmp_path, ["csv"])

# Define the PPO model
model = TD3("CnnPolicy", env, verbose=1)
model.set_logger(new_logger)

# Define the callback to evaluate the model and save checkpoints
eval_callback = EvalCallback(env, eval_freq=1000, n_eval_episodes=5, deterministic=True, verbose=1, best_model_save_path="logs/TD3_Cnn_nolaplimit/")
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/TD3_Cnn_nolaplimit", name_prefix="donkey_td3_cnn_nolaplimit")

# Train the model with callbacks
model.learn(total_timesteps=int(1e6), progress_bar=True, callback=[eval_callback, checkpoint_callback])

# Save the trained model
model.save("logs/TD3_Cnn_nolaplimit/td3_donkey_model_cnn_nolaplimit")

# Close the environment
env.close()
