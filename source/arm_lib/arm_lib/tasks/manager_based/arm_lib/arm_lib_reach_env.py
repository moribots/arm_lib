# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script defines the scene configuration for the Franka trajectory tracking task.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import ContactSensorCfg

from .arm_lib_robot_cfg import FRANKA_PANDA_CFG

##
# Scene definition
##


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """Configuration for the Franka trajectory tracking scene."""

    # Ground plane
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    # Robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG

    # Shelf
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 0.25),
            collision=True,  # This is the fix
            visual_material=materials.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            collision_props=schemas.CollisionPropertiesCfg(),
            rigid_props=schemas.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
    )

    # Target (as a non-physical visual object)
    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=materials.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        )
    )

    # Contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link(5|6|7|hand)",
        force_threshold=1.0
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )
