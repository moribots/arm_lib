# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from ..franka_reach_env import FrankaReachEnv


def distance_to_target(env: FrankaReachEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for minimizing the distance to the target."""
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee_body_idx = robot.find_bodies(asset_cfg.body_names)[0]

    ee_pos = robot.data.body_pos_w[:, ee_body_idx]
    target_pos = target.data.root_pos_w

    return torch.linalg.norm(ee_pos - target_pos, dim=1)


def joint_limit_penalty(env: FrankaReachEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize being close to joint limits."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]

    # Using a simple quadratic penalty, can be replaced with more sophisticated functions
    upper_limit_penalty = torch.square(joint_pos - robot.data.soft_joint_pos_limits_upper[:, asset_cfg.joint_ids])
    lower_limit_penalty = torch.square(joint_pos - robot.data.soft_joint_pos_limits_lower[:, asset_cfg.joint_ids])

    return torch.sum(upper_limit_penalty + lower_limit_penalty, dim=1)


def velocity_penalty(env: FrankaReachEnv, asset_cfg: SceneEntityCfg, is_ee: bool) -> torch.Tensor:
    """Penalize high velocities."""
    robot: Articulation = env.scene[asset_cfg.name]
    if is_ee:
        ee_body_idx = robot.find_bodies(asset_cfg.body_names)[0]
        vel = robot.data.body_vel_w[:, ee_body_idx]
    else:
        vel = robot.data.joint_vel[:, asset_cfg.joint_ids]

    return torch.linalg.norm(vel, dim=1)


def action_smoothness_penalty(env: FrankaReachEnv, action_name: str) -> torch.Tensor:
    """Penalize large changes in actions."""
    action_tensor = env.action_manager.get_action(action_name)
    return torch.sum(torch.square(action_tensor), dim=1)


def acceleration_penalty(env: FrankaReachEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high accelerations."""
    robot: Articulation = env.scene[asset_cfg.name]
    current_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    accel = (current_vel - env.prev_joint_vel) / env.sim.dt
    return torch.sum(torch.square(accel), dim=1)


def jerk_penalty(env: FrankaReachEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high jerk."""
    robot: Articulation = env.scene[asset_cfg.name]
    current_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    current_accel = (current_vel - env.prev_joint_vel) / env.sim.dt
    jerk = (current_accel - env.prev_joint_accel) / env.sim.dt
    return torch.sum(torch.square(jerk), dim=1)


def upright_bonus(env: FrankaReachEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the end-effector upright."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_body_idx = robot.find_bodies(asset_cfg.body_names)[0]
    ee_pos = robot.data.body_pos_w[:, ee_body_idx]
    return (ee_pos[:, 2] > 0.1).float()


def collision_penalty(env: FrankaReachEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize collisions."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.any(contact_sensor.data.in_contact, dim=1).float()
