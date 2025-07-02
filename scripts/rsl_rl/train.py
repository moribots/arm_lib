# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script is the main entry point for training the Franka trajectory tracking
agent using the RSL-RL library.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train a Franka agent for trajectory tracking.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to train on.")
parser.add_argument("--task", type=str, default="Franka-Reach-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import arm_lib  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg


def main():
    """Main function to train the agent."""
    # Create the environment
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)
    # Wrap the environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create the runner configuration
    from arm_lib.tasks.manager_based.arm_lib.agents import rsl_rl_ppo_cfg
    runner_cfg: RslRlOnPolicyRunnerCfg = rsl_rl_ppo_cfg.RslRlPpoAgentCfg
    runner_cfg.seed = args_cli.seed

    # Create the runner
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir="logs/rsl_rl", device=args_cli.device)

    # Train the agent
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    # Close the environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        simulation_app.close()
