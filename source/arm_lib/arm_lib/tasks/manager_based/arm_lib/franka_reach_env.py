# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections import deque
import numpy as np
import random
import math
import importlib
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import sample_uniform
from scipy.spatial.transform import Rotation as ScipyRotation

from .franka_reach_env_cfg import FrankaReachEnvCfg
from .curriculum_core import LinearCurriculum


class TaskLogic:
    """
    Manages the procedural generation logic for the Franka shelf task.
    This class is responsible for sampling random shelf and target poses.
    """
    DEFAULT_PLATE_WIDTH: float = 0.5
    DEFAULT_PLATE_DEPTH: float = 0.4
    DEFAULT_PLATE_THICKNESS: float = 0.02
    DEFAULT_WALL_THICKNESS: float = 0.02
    DEFAULT_ACTUAL_OPENING_HEIGHT: float = 0.25

    def __init__(self, num_envs: int, randomize_shelf_config: bool, device: str):
        self.num_envs = num_envs
        self.randomize_shelf_config = randomize_shelf_config
        self.device = device
        self._define_shelf_configurations()
        self.current_shelf_instance_params_per_env: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def _define_shelf_configurations(self):
        self.shelf_configurations: dict[str, dict[str, Any]] = {
            "default_center_reach": {"name": "default_center_reach", "base_pos_range_x": [(-0.6, -0.35), (0.35, 0.6)], "base_pos_range_y": (0.0, 0.0), "base_pos_range_z": (0.3, 0.3)},
            "high_center_reach": {"name": "high_center_reach", "base_pos_range_x": [(-0.6, -0.35), (0.35, 0.6)], "base_pos_range_y": (-0.4, 0.4), "base_pos_range_z": (0.4, 0.65)},
            "low_forward_reach": {"name": "low_forward_reach", "base_pos_range_x": [(-0.60, -0.35), (0.35, 0.60)], "base_pos_range_y": (-0.4, 0.4), "base_pos_range_z": (0.1, 0.30)},
            "mid_side_reach_right": {"name": "mid_side_reach_right", "base_pos_range_x": [(0.3, 0.5)], "base_pos_range_y": (-0.6, 0.6), "base_pos_range_z": (0.2, 0.5)},
            "mid_side_reach_left": {"name": "mid_side_reach_left", "base_pos_range_x": [(-0.5, -0.3)], "base_pos_range_y": (-0.6, -0.6), "base_pos_range_z": (0.2, 0.5)}
        }
        self.shelf_config_keys_list: list[str] = list(self.shelf_configurations.keys())
        self.default_shelf_config_key: str = "default_center_reach"

    def _sample_shelf_assembly_poses(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_selected_envs = len(env_ids)
        origins = torch.zeros((num_selected_envs, 3), device=self.device)
        yaws = torch.zeros(num_selected_envs, device=self.device)

        for i, env_idx in enumerate(env_ids):
            config_key = random.choice(self.shelf_config_keys_list) if self.randomize_shelf_config else self.default_shelf_config_key
            config = self.shelf_configurations[config_key]

            chosen_x_sub_range = random.choice(config["base_pos_range_x"])
            origins[i, 0] = sample_uniform(chosen_x_sub_range[0], chosen_x_sub_range[1], (1,), device=self.device)
            origins[i, 1] = sample_uniform(config["base_pos_range_y"][0], config["base_pos_range_y"][1], (1,), device=self.device)
            origins[i, 2] = sample_uniform(config["base_pos_range_z"][0], config["base_pos_range_z"][1], (1,), device=self.device)

            if "center" in config["name"] or "forward" in config["name"]:
                yaws[i] = torch.atan2(-origins[i, 1], -origins[i, 0]) - math.pi / 2.0
            else:
                yaws[i] = 0.0

            self.current_shelf_instance_params_per_env[env_idx] = {
                "shelf_assembly_origin_world": origins[i].clone(), "shelf_assembly_yaw": yaws[i].item(),
                "internal_width": self.DEFAULT_PLATE_WIDTH - 2 * self.DEFAULT_WALL_THICKNESS,
                "internal_depth": self.DEFAULT_PLATE_DEPTH - self.DEFAULT_WALL_THICKNESS,
                "actual_opening_height": self.DEFAULT_ACTUAL_OPENING_HEIGHT
            }
        return origins, yaws

    def compute_shelf_pose(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assembly_origins_world, assembly_yaws = self._sample_shelf_assembly_poses(env_ids)
        cos_yaw_half = torch.cos(assembly_yaws / 2.0)
        sin_yaw_half = torch.sin(assembly_yaws / 2.0)
        assembly_quats_wxyz = torch.zeros((len(env_ids), 4), device=self.device)
        assembly_quats_wxyz[:, 0] = cos_yaw_half
        assembly_quats_wxyz[:, 3] = sin_yaw_half
        return assembly_origins_world, assembly_quats_wxyz

    def compute_target_poses(self, env_ids: torch.Tensor) -> torch.Tensor:
        num_selected_envs = len(env_ids)
        target_positions = torch.zeros((num_selected_envs, 3), device=self.device)
        target_quats_wxyz = torch.zeros((num_selected_envs, 4), device=self.device)
        target_quats_wxyz[:, 0] = 1.0

        for i, env_idx in enumerate(env_ids):
            params = self.current_shelf_instance_params_per_env[env_idx]
            if not params:
                continue
            w, d, h = params["internal_width"], params["internal_depth"], params["actual_opening_height"]
            target_in_opening_frame = torch.tensor([
                sample_uniform(-w / 2 * 0.7, w / 2 * 0.7, (1,), device=self.device),
                sample_uniform(-d / 2 * 0.7, d / 2 * 0.3, (1,), device=self.device),
                sample_uniform(-h / 2 * 0.8, h / 2 * 0.8, (1,), device=self.device)
            ], device=self.device)
            assembly_origin = params["shelf_assembly_origin_world"]
            assembly_yaw = params["shelf_assembly_yaw"]
            offset_to_opening_center = torch.tensor([0, 0, self.DEFAULT_PLATE_THICKNESS + h / 2.0], device=self.device)
            rotation = ScipyRotation.from_euler('z', assembly_yaw)
            rot_matrix = torch.from_numpy(rotation.as_matrix()).float().to(self.device)
            opening_center_world = assembly_origin + torch.mv(rot_matrix, offset_to_opening_center)
            target_offset_world = torch.mv(rot_matrix, target_in_opening_frame)
            target_positions[i] = opening_center_world + target_offset_world

        return torch.cat([target_positions, target_quats_wxyz], dim=1)


class FrankaReachEnv(ManagerBasedRLEnv):
    """Custom environment for Franka Reach task that handles curriculum and state history."""
    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)

        randomize_shelf_config = getattr(self.cfg, 'randomize_shelf_config', False)
        self.task_logic = TaskLogic(self.num_envs, randomize_shelf_config, self.device)
        self._init_curriculum()
        self._init_buffers()

    def _setup_scene(self):
        super()._setup_scene()

    def _init_curriculum(self):
        self.curricula = {}
        curriculum_configs = {
            "threshold": "threshold_curriculum",
            "joint_velocity_penalty": "joint_velocity_penalty_curriculum",
            "ee_velocity_penalty": "ee_velocity_penalty_curriculum",
            "action_penalty": "action_penalty_curriculum",
            "accel_penalty": "accel_penalty_curriculum",
            "jerk_penalty": "jerk_penalty_curriculum",
            "upright_bonus": "upright_bonus_curriculum",
        }
        for name, cfg_attr in curriculum_configs.items():
            if hasattr(self.cfg, cfg_attr):
                self.curricula[name] = LinearCurriculum(getattr(self.cfg, cfg_attr))

        self.success_buffer = deque(maxlen=100 * self.num_envs)
        self.current_success_rate = 0.0

    def _init_buffers(self):
        num_joints = 7
        self.prev_joint_vel = torch.zeros((self.num_envs, num_joints), device=self.device)
        self.prev_joint_accel = torch.zeros((self.num_envs, num_joints), device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        self._update_curriculum()

    def _update_curriculum(self):
        if len(self.success_buffer) > 0:
            self.current_success_rate = np.mean(list(self.success_buffer))
        else:
            self.current_success_rate = 0.0

        for name, curriculum in self.curricula.items():
            if name == 'threshold':
                continue
            curriculum.update(self.current_success_rate)
            if hasattr(self, 'reward_manager') and name in self.reward_manager.terms:
                self.reward_manager.terms[name].weight = curriculum.current_value
            self.extras[f"curriculum/{name}"] = curriculum.current_value

        if 'threshold' in self.curricula:
            threshold_curriculum = self.curricula['threshold']
            threshold_curriculum.update(self.current_success_rate)
            if 'successful_reach' in self.termination_manager.terms:
                self.termination_manager.terms['successful_reach'].params['threshold'] = threshold_curriculum.current_value
            self.extras["curriculum/threshold"] = threshold_curriculum.current_value

        self.extras["curriculum/success_rate"] = self.current_success_rate

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self._reset_task(env_ids)
            self.prev_joint_vel[env_ids] = 0.
            self.prev_joint_accel[env_ids] = 0.

    def _reset_task(self, env_ids: torch.Tensor):
        # Sample new poses for the shelf
        shelf_pos, shelf_rot = self.task_logic.compute_shelf_pose(env_ids)
        # Set the root state of the shelf asset
        shelf = self.scene["shelf"]
        shelf.set_root_state(torch.cat([shelf_pos, shelf_rot], dim=1), env_ids=env_ids)

        # Sample new poses for the target
        target_pose = self.task_logic.compute_target_poses(env_ids)
        # Update the command manager with the new target pose
        self.command_manager.update_command("target_pose", target_pose, env_ids=env_ids)
