from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


ISAAC_JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

ISAAC_FOOT_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

ISAAC_DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _isaac_go2_usd_path() -> str:
    return os.environ.get("GO2_BACKFLIP_USD_PATH") or f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"


ISAAC_GO2_ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_isaac_go2_usd_path(),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=ISAAC_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "isaac_torque": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=23.7,
            velocity_limit_sim=30.1,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)


@configclass
class Go2BackflipIsaacEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 2.0
    decimation = 4
    is_finite_horizon = False
    action_space = 12
    observation_space = 60
    state_space = 64

    # Backflip control setup.
    action_scale = 0.5
    clip_actions = 100.0
    action_latency = True
    kp = 70.0
    kd = 3.0
    base_init_height = 0.32
    target_stand_height = 0.3
    rotation_reward_start_s = 0.5
    rotation_reward_end_s = 1.25
    lift_reward_start_s = 0.5
    lift_reward_end_s = 0.75
    orientation_start_s = 0.5
    orientation_ramp_duration_s = 0.75

    # Train with domain randomization and disable it only for evaluation.
    randomize_controls = _env_bool("GO2_BACKFLIP_RANDOMIZE_CONTROLS", True)
    randomize_motor_strength = _env_bool("GO2_BACKFLIP_RANDOMIZE_MOTOR_STRENGTH", False)
    motor_offset_range = (-0.02, 0.02)
    kp_scale_range = (0.8, 1.2)
    kd_scale_range = (0.8, 1.2)
    motor_strength_range = (0.9, 1.1)

    randomize_rigids = _env_bool("GO2_BACKFLIP_RANDOMIZE_RIGIDS", True)
    randomize_friction = _env_bool("GO2_BACKFLIP_RANDOMIZE_FRICTION", True)
    randomize_base_mass = _env_bool("GO2_BACKFLIP_RANDOMIZE_BASE_MASS", True)
    randomize_com_displacement = _env_bool("GO2_BACKFLIP_RANDOMIZE_COM", True)
    friction_range = (0.2, 1.5)
    added_mass_range = (-1.0, 3.0)
    com_displacement_range = (-0.01, 0.01)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_env_int("NUM_ENVS", 4096),
        env_spacing=2.5,
        replicate_physics=True,
    )

    # robot
    robot: ArticulationCfg = ISAAC_GO2_ROBOT_CFG
    isaac_joint_order = ISAAC_JOINT_ORDER
    isaac_foot_order = ISAAC_FOOT_ORDER
    # Runtime reward multiplies these scales by step_dt.
    rew_ang_vel_y_scale = 5.0
    rew_ang_vel_z_scale = -1.0
    rew_lin_vel_z_scale = 20.0
    rew_orientation_control_scale = -1.0
    rew_feet_height_before_backflip_scale = -30.0
    rew_height_control_scale = -10.0
    rew_actions_symmetry_scale = -0.1
    rew_gravity_y_scale = -10.0
    rew_feet_distance_scale = -1.0
    rew_action_rate_scale = -0.001

    def __post_init__(self):
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class Go2BackflipIsaacPlayEnvCfg(Go2BackflipIsaacEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.randomize_controls = False
        self.randomize_rigids = False
