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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

# Direct imports for standard MDP terms from Isaac Lab
from isaaclab.envs.mdp import actions as mdp_actions
from isaaclab.envs.mdp import commands as mdp_commands
from isaaclab.envs.mdp import observations as mdp_observations
from isaaclab.envs.mdp import terminations as mdp_terminations

# Local import for custom-defined rewards and terminations
from . import mdp

from .arm_lib_reach_env import FrankaSceneCfg
from .curriculum_core import CurriculumConfig

##
# MDP settings
##


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
        actions = ObsTerm(func=mdp_observations.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # This is now handled by the custom environment class
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    distance_to_target = RewTerm(
        func=mdp.rewards.distance_to_target,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), "target_cfg": SceneEntityCfg("target")},
    )

    # MODIFIED: Replaced 'joint_names_expr' with an explicit list for 'joint_names'
    joint_limit_penalty = RewTerm(
        func=mdp.rewards.joint_limit_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"])},
    )

    collision_penalty = RewTerm(
        func=mdp.rewards.collision_penalty,
        weight=-20.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
    )

    action_penalty = RewTerm(func=mdp.rewards.action_smoothness_penalty, weight=0.0)

    # MODIFIED: Replaced 'joint_names_expr' with an explicit list for 'joint_names'
    accel_penalty = RewTerm(func=mdp.rewards.acceleration_penalty, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"])})
    jerk_penalty = RewTerm(func=mdp.rewards.jerk_penalty, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"])})
    joint_velocity_penalty = RewTerm(func=mdp.rewards.velocity_penalty, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]), "is_ee": False})

    ee_velocity_penalty = RewTerm(func=mdp.rewards.velocity_penalty, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), "is_ee": True})
    upright_bonus = RewTerm(func=mdp.rewards.upright_bonus, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp_terminations.time_out, time_out=True)

    successful_reach = DoneTerm(
        func=mdp.terminations.terminate_on_success,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
                "target_cfg": SceneEntityCfg("target"),
                "threshold": 0.05},
    )

    collision = DoneTerm(
        func=mdp.terminations.terminate_on_collision,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
    )


##
# Environment configuration
##


@configclass
class ArmLibEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka arm environment."""

    # Scene settings
    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # Play settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post-initialization checks."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # simulation settings
        self.sim.dt = 1 / 120.0
        # self.sim.render_interval = self.decimation
        # update viewer
        self.viewer.eye = (3.0, 3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
