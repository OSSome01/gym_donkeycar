import os
import numpy as np
import torch as th
import gym
from stable_baselines3 import PPO, SAC, DDPG, TD3
from sb3_contrib import TQC
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

# tmp_path = "logs/PPO_Cnn_nolaplimit_1e6steps_removecte_cont"
tmp_path = "logs/TQC_Cnn_nolaplimit_1e6steps_aggressive_withbrakes_cont"
# set up logger
logger = configure(tmp_path, ["stdout","csv"])

# Define the PPO model
# model = TQC("CnnPolicy", env, verbose=1, tensorboard_log=tmp_path)
model = TQC.load("logs/TQC_Cnn_nolaplimit_1e6steps_aggressive_withbrakes_cont/donkey_tqc_cnn_nolaplimit_218000_steps", env, verbose=1)

model.set_logger(logger)

# Define the callback to evaluate the model and save checkpoints
eval_callback = EvalCallback(env, eval_freq=1000, n_eval_episodes=5, deterministic=True, verbose=1, best_model_save_path="logs/TQC_Cnn_nolaplimit_1e6steps_aggressive_withbrakes_cont/")
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/TQC_Cnn_nolaplimit_1e6steps_aggressive_withbrakes_cont", name_prefix="donkey_tqc_cnn_nolaplimit")

# Train the model with callbacks
model.learn(total_timesteps=int(1e6), progress_bar=True, callback=[eval_callback, checkpoint_callback], reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e6), reset_num_timesteps=False)

# Save the trained model
model.save("TQC_Cnn_nolaplimit_1e6steps_aggressive_withbrakes")

# Close the environment
env.close()
