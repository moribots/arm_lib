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


def terminate_on_success(env: FrankaReachEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Terminate when the end-effector is close to the target."""
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee_body_idx = robot.find_bodies(asset_cfg.body_names)[0]

    ee_pos = robot.data.body_pos_w[:, ee_body_idx]
    target_pos = target.data.root_pos_w

    dist_to_target = torch.linalg.norm(ee_pos - target_pos, dim=1)
    return dist_to_target < threshold


def terminate_on_collision(env: FrankaReachEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate on collision."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.any(contact_sensor.data.in_contact, dim=1)