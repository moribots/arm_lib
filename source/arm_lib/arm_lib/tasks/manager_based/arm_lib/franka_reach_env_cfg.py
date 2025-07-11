# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

# Direct imports for standard MDP terms from Isaac Lab
from isaaclab.envs.mdp import actions as mdp_actions
from isaaclab.envs.mdp import commands as mdp_commands
from isaaclab.envs.mdp import observations as mdp_observations
from isaaclab.envs.mdp import terminations as mdp_terminations
from isaaclab.envs.mdp import events as mdp_events

# Local import for custom-defined rewards and terminations
from . import mdp

from .franka_reach_scene_cfg import FrankaSceneCfg
from .curriculum_core import CurriculumConfig

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Commands for the MDP."""
    target_pose = mdp_commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(99.0, 100.0),
        goal_pose_visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="{ENV_REGEX_NS}/goal_marker"),
        ranges=mdp_commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action = mdp_actions.JointEffortActionCfg(asset_name="robot", joint_names=["panda_joint[1-7]"], scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp_observations.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])})
        joint_vel = ObsTerm(func=mdp_observations.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])})
        ee_pose = ObsTerm(func=mdp_observations.body_pose_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])})
        relative_target_pos = ObsTerm(func=mdp_observations.generated_commands, params={"command_name": "target_pose"})
        actions = ObsTerm(func=mdp_observations.last_action, params={"action_name": "arm_action"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_scene = EventTerm(func=mdp_events.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    distance_to_target = RewTerm(
        func=mdp.rewards.distance_to_target,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), "target_cfg": SceneEntityCfg("target")},
    )
    joint_limit_penalty = RewTerm(
        func=mdp.rewards.joint_limit_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    collision_penalty = RewTerm(
        func=mdp.rewards.collision_penalty,
        weight=-20.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
    )
    action_penalty = RewTerm(
        func=mdp.rewards.action_smoothness_penalty,
        weight=0.0,
        params={"action_name": "arm_action"}
    )
    accel_penalty = RewTerm(func=mdp.rewards.acceleration_penalty, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])})
    jerk_penalty = RewTerm(func=mdp.rewards.jerk_penalty, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])})
    joint_velocity_penalty = RewTerm(func=mdp.rewards.velocity_penalty, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]), "is_ee": False})
    ee_velocity_penalty = RewTerm(func=mdp.rewards.velocity_penalty, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), "is_ee": True})
    upright_bonus = RewTerm(func=mdp.rewards.upright_bonus, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp_terminations.time_out, time_out=True)
    successful_reach = DoneTerm(
        func=mdp.terminations.terminate_on_success,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), "target_cfg": SceneEntityCfg("target"), "threshold": 0.05},
    )
    collision = DoneTerm(
        func=mdp.terminations.terminate_on_collision,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
    )

##
# Environment configuration
##


@configclass
class FrankaReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka arm environment."""
    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=1024, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Curriculum settings
    randomize_shelf_config: bool = True
    threshold_curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(start_value=0.05, end_value=0.005, start_metric_val=0.5, end_metric_val=0.84))
    joint_velocity_penalty_curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(start_value=0.0, end_value=0.5, start_metric_val=0.85, end_metric_val=0.95))
    ee_velocity_penalty_curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(start_value=0.0, end_value=0.5, start_metric_val=0.85, end_metric_val=0.95))
    action_penalty_curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(start_value=1.0e-4, end_value=1.0e-3, start_metric_val=0.0, end_metric_val=0.2))
    accel_penalty_curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(start_value=0.0, end_value=1.0e-6, start_metric_val=0.4, end_metric_val=0.6))
    jerk_penalty_curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(start_value=0.0, end_value=1.0e-12, start_metric_val=0.7, end_metric_val=0.8))
    upright_bonus_curriculum: CurriculumConfig = field(default_factory=lambda: CurriculumConfig(start_value=1.0, end_value=0.0, start_metric_val=0.0, end_metric_val=0.4))

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
