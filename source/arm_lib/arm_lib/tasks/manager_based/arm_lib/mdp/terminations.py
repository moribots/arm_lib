# moribots/arm_lib/arm_lib-5bc5497a62f1367fdbda2ad18ae748a674e2cff5/source/arm_lib/arm_lib/tasks/manager_based/arm_lib/mdp/terminations.py

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
    print(f"Executing termination: terminate_on_success, asset: {asset_cfg.name}, target: {target_cfg.name}")
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee_body_idx = robot.find_bodies(asset_cfg.body_names)[0]

    ee_pos = robot.data.body_pos_w[:, ee_body_idx]
    target_pos = target.data.root_pos_w

    dist_to_target = torch.linalg.norm(ee_pos - target_pos, dim=1)
    return dist_to_target < threshold


def terminate_on_collision(env: FrankaReachEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate on collision."""
    print(f"Executing termination: terminate_on_collision, sensor: {sensor_cfg.name}")
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.any(contact_sensor.data.in_contact, dim=1)
