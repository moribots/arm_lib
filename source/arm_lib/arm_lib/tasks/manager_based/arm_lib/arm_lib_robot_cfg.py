# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script defines the configuration for the Franka robot asset.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Define the Franka robot configuration
FRANKA_PANDA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="path/to/your/franka/panda.usd",  # TODO: Update this path
        activate_contact_sensors=True,  # Enable contact reporting
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=0.0,
            damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=1e5,
            damping=1e3,
        ),
    },
)
