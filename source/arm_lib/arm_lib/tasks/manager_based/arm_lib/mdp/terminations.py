# moribots/arm_lib/arm_lib-5bc5497a62f1367fdbda2ad18ae748a674e2cff5/source/arm_lib/arm_lib/tasks/manager_based/arm_lib/mdp/terminations.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from ..franka_reach_env import FrankaReachEnv


def terminate_on_success(env: FrankaReachEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Terminate when the end-effector is close to the target."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_body_idx = robot.find_bodies(asset_cfg.body_names)[0]

    ee_pos = robot.data.body_pos_w[:, ee_body_idx]
    # The target's pose is now correctly retrieved from the command manager.
    target_pos = env.command_manager.get_command("target_pose")[:, :3]

    dist_to_target = torch.linalg.norm(ee_pos - target_pos, dim=1)
    return dist_to_target < threshold


def terminate_on_collision(env: FrankaReachEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate on collision."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.any(contact_sensor.data.in_contact, dim=1)
