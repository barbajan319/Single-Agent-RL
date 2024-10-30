import gymnasium as gym
from stable_baselines3 import SAC
import os
import numpy as np
import time
import pandas as pd
from Charge_ENV import ChargingEnv
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import configure

tmp_path = "/tmp/sb3_log"
models_dir = f"models/SAC-{int(time.time())}"
log_dir = f"logs/SAC-{int(time.time())}"
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = ChargingEnv()
env.reset()
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions), sigma=float(0.8) * np.ones(n_actions), theta=0.15
)
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# model = DDPG(
#     MlpPolicy,
#     env,
#     action_noise=action_noise,
#     tensorboard_log=log_dir,
#     learning_rate=0.005,
#     buffer_size=int(1e5),
#     batch_size=100,
#     gamma=0.99,
#     learning_starts=2400,
#     tau=0.005,
#     verbose=1,
# )
policy_kwargs = dict(net_arch=[256, 256, 256])  # Three layers with 256 units each

model = SAC(
    "MlpPolicy",
    env,
    tensorboard_log=log_dir,
    learning_rate=0.005,
    buffer_size=int(1e5),
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    verbose=1,
    policy_kwargs=policy_kwargs,
)
TIMESTEPS = 24000
rewards = []
avg_rewards = []
charging_cost = []
avg_cost = []
energy_usage = []
avg_cons = []
degradation_penalty = []
avg_deg = []
avg_soc_dep = []


class EpisodeLoggingCallback(BaseCallback):
    def __init__(self, log_interval=24, verbose=0):
        super(EpisodeLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.ep_counter = 0

    def _on_step(self) -> bool:
        if env.timestep % self.log_interval == 0 and env.timestep != 0:
            ep_rewards = sum(env.Reward)
            ep_rewards_mean = ep_rewards / len(env.Reward)
            ep_cost = sum(env.Cost_History)
            ep_cost_mean = ep_cost / len(env.Cost_History)
            ep_energy = sum(env.Grid_Evol)
            ep_energy_mean = ep_energy / len(env.Grid_Evol)
            deg_penalty = sum(env.Battery_Degradation)
            deg_pen_mean = deg_penalty / len(env.Battery_Degradation)
            avg_soc_dep.append(sum(env.SOC_FINISH) / len(env.SOC_FINISH))

            rewards.append(-ep_rewards)
            avg_rewards.append(-ep_rewards_mean)
            charging_cost.append(-ep_cost)
            avg_cost.append(-ep_cost_mean)
            energy_usage.append(-ep_energy)
            avg_cons.append(-ep_energy_mean)
            degradation_penalty.append(deg_penalty)
            avg_deg.append(deg_pen_mean)
        return True


logging_callback = EpisodeLoggingCallback(log_interval=23)
model.learn(
    total_timesteps=TIMESTEPS,
    log_interval=10,
    tb_log_name="DDPG",
    callback=logging_callback,
)


def moving_average(data, window_size=250):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# # Plot rewards and avg_rewards with smoothing
plt.figure(figsize=(10, 6))
plt.plot(moving_average(rewards), label="Rewards (smoothed)", linestyle="-")
plt.plot(
    moving_average(avg_rewards), label="Average Rewards (smoothed)", linestyle="--"
)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards and Average Rewards Over Episodes (Smoothed)")
plt.legend()
plt.show()

# Plot charging_cost and avg_cost with smoothing
plt.figure(figsize=(10, 6))
plt.plot(moving_average(charging_cost), label="Charging Cost (smoothed)", linestyle="-")
plt.plot(moving_average(avg_cost), label="Average Cost (smoothed)", linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Charging Cost")
plt.title("Charging Cost and Average Cost Over Episodes (Smoothed)")
plt.legend()
plt.show()

# Plot energy_usage and avg_cons with smoothing
plt.figure(figsize=(10, 6))
plt.plot(moving_average(energy_usage), label="Energy Usage (smoothed)", linestyle="-")
plt.plot(
    moving_average(avg_cons),
    label="Average Energy Consumption (smoothed)",
    linestyle="--",
)
plt.xlabel("Episode")
plt.ylabel("Energy Usage")
plt.title("Energy Usage and Average Consumption Over Episodes (Smoothed)")
plt.legend()
plt.show()

# Plot degradation_penalty and avg_deg with smoothing
plt.figure(figsize=(10, 6))
plt.plot(
    moving_average(degradation_penalty),
    label="Degradation Penalty (smoothed)",
    linestyle="-",
)
plt.plot(
    moving_average(avg_deg), label="Average Degradation (smoothed)", linestyle="--"
)
plt.xlabel("Episode")
plt.ylabel("Degradation Penalty")
plt.title("Degradation Penalty and Average Degradation Over Episodes (Smoothed)")
plt.legend()
plt.show()

# Plot avg_soc_dep as an area graph with smoothing
smoothed_avg_soc_dep = moving_average(avg_soc_dep)
plt.figure(figsize=(10, 6))
plt.fill_between(
    range(len(smoothed_avg_soc_dep)), smoothed_avg_soc_dep, color="skyblue", alpha=0.4
)
plt.plot(smoothed_avg_soc_dep, color="Slateblue", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Average SOC at Departure")
plt.title("Average SOC at Departure Over Episodes (Smoothed)")
plt.show()
