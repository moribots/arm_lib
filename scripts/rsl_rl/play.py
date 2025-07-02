# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script is the main entry point for evaluating a trained Franka agent
using the RSL-RL library.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained Franka agent.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to evaluate on.")
parser.add_argument("--task", type=str, default="Franka-Reach-v0", help="Name of the task.")
parser.add_argument("--load_run", type=str, required=True, help="Run to load.")
parser.add_argument("--checkpoint", type=str, default="latest", help="Checkpoint to load.")
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
    """Main function to evaluate the agent."""
    # Create the environment
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)
    # Wrap the environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create the runner configuration
    from arm_lib.tasks.manager_based.arm_lib.agents import rsl_rl_ppo_cfg
    runner_cfg: RslRlOnPolicyRunnerCfg = rsl_rl_ppo_cfg.RslRlPpoAgentCfg
    runner_cfg.resume = True
    runner_cfg.load_run = args_cli.load_run
    runner_cfg.load_checkpoint = args_cli.checkpoint

    # Create the runner
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir="logs/rsl_rl/play", device=args_cli.device)

    # Obtain the trained policy
    policy = runner.get_inference_policy(device=env.device)

    # Evaluate the policy
    for _ in range(1000):
        obs, _ = env.get_observations()
        actions = policy(obs)
        env.step(actions)

    # Close the environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        simulation_app.close()
