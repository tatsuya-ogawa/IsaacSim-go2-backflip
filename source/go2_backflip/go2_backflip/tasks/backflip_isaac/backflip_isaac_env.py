from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .backflip_isaac_env_cfg import Go2BackflipIsaacEnvCfg


def _normalize(x: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    out = vec + quat[:, :1] * t + torch.cross(xyz, t, dim=-1)
    return out.view(shape)


def _quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half_angle = (angle * 0.5).unsqueeze(-1)
    xyz = _normalize(axis).reshape(1, 3) * torch.sin(half_angle)
    w = torch.cos(half_angle)
    return _normalize(torch.cat((w, xyz), dim=-1))


def _transform_by_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return _quat_apply(quat, pos)


class Go2BackflipIsaacEnv(DirectRLEnv):
    cfg: Go2BackflipIsaacEnvCfg

    def __init__(self, cfg: Go2BackflipIsaacEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        joint_ids, joint_names = self._robot.find_joints(self.cfg.isaac_joint_order, preserve_order=True)
        if joint_names != list(self.cfg.isaac_joint_order):
            raise RuntimeError(
                "Could not resolve Isaac joint order in Isaac articulation. "
                f"Expected {self.cfg.isaac_joint_order}, got {joint_names}."
            )
        foot_ids, foot_names = self._robot.find_bodies(self.cfg.isaac_foot_order, preserve_order=True)
        if foot_names != list(self.cfg.isaac_foot_order):
            raise RuntimeError(
                "Could not resolve Isaac foot body order in Isaac articulation. "
                f"Expected {self.cfg.isaac_foot_order}, got {foot_names}."
            )
        base_body_ids, base_body_names = self._robot.find_bodies(["base"], preserve_order=True)
        if base_body_names != ["base"]:
            raise RuntimeError(f"Could not resolve Isaac base body. Got {base_body_names}.")
        self._joint_ids = joint_ids
        self._foot_body_ids = foot_ids
        self._base_body_id = int(base_body_ids[0])
        self._actions = torch.zeros(self.num_envs, int(self.cfg.action_space), device=self.device)
        self._last_actions = torch.zeros_like(self._actions)
        self._exec_actions = torch.zeros_like(self._actions)
        self._torques = torch.zeros_like(self._actions)
        self._motor_offsets = torch.zeros_like(self._actions)
        self._motor_strengths = torch.ones_like(self._actions)
        self._p_gains = torch.full_like(self._actions, float(self.cfg.kp))
        self._d_gains = torch.full_like(self._actions, float(self.cfg.kd))
        self._default_dof_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()
        self._global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=self.device)
        self._default_coms = self._robot.root_physx_view.get_coms().clone()
        self._default_materials = self._robot.root_physx_view.get_material_properties().clone()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "ang_vel_y",
                "ang_vel_z",
                "lin_vel_z",
                "orientation_control",
                "feet_height_before_backflip",
                "height_control",
                "actions_symmetry",
                "gravity_y",
                "feet_distance",
                "action_rate",
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._last_actions = self._actions.clone()
        self._actions = actions.clamp(-float(self.cfg.clip_actions), float(self.cfg.clip_actions))
        self._exec_actions = self._last_actions if self.cfg.action_latency else self._actions

    def _apply_action(self):
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        actions_scaled = self._exec_actions * float(self.cfg.action_scale)
        torques = self._p_gains * (actions_scaled + self._default_dof_pos - joint_pos + self._motor_offsets)
        torques = torques - self._d_gains * joint_vel
        torques = torques * self._motor_strengths
        effort_limits = self._robot.data.joint_effort_limits[:, self._joint_ids]
        self._torques = torch.clamp(torques, -effort_limits, effort_limits)
        self._robot.set_joint_effort_target(self._torques, joint_ids=self._joint_ids)

    def _phase_features(self) -> torch.Tensor:
        phase = math.pi * self.episode_length_buf.float().unsqueeze(-1) * self.step_dt / 2.0
        return torch.cat(
            (
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2.0),
                torch.cos(phase / 2.0),
                torch.sin(phase / 4.0),
                torch.cos(phase / 4.0),
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        phase_features = self._phase_features()

        policy_obs = torch.cat(
            (
                self._robot.data.root_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                joint_pos - self._default_dof_pos,
                joint_vel * 0.05,
                self._actions,
                self._last_actions,
                phase_features,
            ),
            dim=-1,
        )
        critic_obs = torch.cat(
            (
                self._robot.data.root_pos_w[:, 2:3],
                self._robot.data.root_lin_vel_w * 2.0,
                self._robot.data.root_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                joint_pos - self._default_dof_pos,
                joint_vel * 0.05,
                self._actions,
                self._last_actions,
                phase_features,
            ),
            dim=-1,
        )
        return {"policy": policy_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        current_time = self.episode_length_buf.float() * self.step_dt
        base_pos = self._robot.data.root_pos_w
        base_quat = self._robot.data.root_quat_w
        base_lin_vel_w = self._robot.data.root_lin_vel_w
        base_ang_vel_b = self._robot.data.root_ang_vel_b
        projected_gravity_b = self._robot.data.projected_gravity_b
        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_body_ids]

        active_rotation = torch.logical_and(
            current_time > float(self.cfg.rotation_reward_start_s),
            current_time < float(self.cfg.rotation_reward_end_s),
        )
        active_lift = torch.logical_and(
            current_time > float(self.cfg.lift_reward_start_s),
            current_time < float(self.cfg.lift_reward_end_s),
        )
        active_height = torch.logical_or(current_time < 0.4, current_time > 1.4)

        rew_ang_vel_y = -base_ang_vel_b[:, 1].clamp(min=-7.2, max=7.2) * active_rotation.float()
        rew_ang_vel_z = torch.abs(base_ang_vel_b[:, 2])
        rew_lin_vel_z = base_lin_vel_w[:, 2].clamp(max=3.0) * active_lift.float()

        orientation_duration = max(float(self.cfg.orientation_ramp_duration_s), 1.0e-6)
        phase = (current_time - float(self.cfg.orientation_start_s)).clamp(min=0.0, max=orientation_duration)
        pitch_quat = _quat_from_angle_axis(
            2.0 * math.pi * phase / orientation_duration,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float, device=self.device),
        )
        desired_projected_gravity = _transform_by_quat(self._global_gravity.expand(self.num_envs, 3), _quat_conjugate(pitch_quat))
        rew_orientation_control = torch.sum(torch.square(projected_gravity_b - desired_projected_gravity), dim=1)

        rew_height_control = torch.square(float(self.cfg.target_stand_height) - base_pos[:, 2]) * active_height.float()
        rew_feet_height = (foot_pos_w[:, :, 2] - 0.02).clamp(min=0.0).sum(dim=1) * (current_time < 0.5).float()

        rew_actions_symmetry = torch.square(self._actions[:, 0] + self._actions[:, 3])
        rew_actions_symmetry += torch.square(self._actions[:, 1:3] - self._actions[:, 4:6]).sum(dim=-1)
        rew_actions_symmetry += torch.square(self._actions[:, 6] + self._actions[:, 9])
        rew_actions_symmetry += torch.square(self._actions[:, 7:9] - self._actions[:, 10:12]).sum(dim=-1)

        rew_gravity_y = torch.square(projected_gravity_b[:, 1])

        foot_pos_translated = foot_pos_w - base_pos.unsqueeze(1)
        foot_pos_body = _quat_apply(_quat_conjugate(base_quat).unsqueeze(1).expand(-1, 4, -1), foot_pos_translated)
        desired_ys = torch.tensor([0.15, -0.15, 0.15, -0.15], dtype=torch.float, device=self.device).unsqueeze(0)
        rew_feet_distance = torch.square(desired_ys - foot_pos_body[:, :, 1]).sum(dim=1)

        rew_action_rate = torch.sum(torch.square(self._last_actions - self._actions), dim=1)

        rewards = {
            "ang_vel_y": self.cfg.rew_ang_vel_y_scale * rew_ang_vel_y * self.step_dt,
            "ang_vel_z": self.cfg.rew_ang_vel_z_scale * rew_ang_vel_z * self.step_dt,
            "lin_vel_z": self.cfg.rew_lin_vel_z_scale * rew_lin_vel_z * self.step_dt,
            "orientation_control": self.cfg.rew_orientation_control_scale * rew_orientation_control * self.step_dt,
            "feet_height_before_backflip": self.cfg.rew_feet_height_before_backflip_scale * rew_feet_height * self.step_dt,
            "height_control": self.cfg.rew_height_control_scale * rew_height_control * self.step_dt,
            "actions_symmetry": self.cfg.rew_actions_symmetry_scale * rew_actions_symmetry * self.step_dt,
            "gravity_y": self.cfg.rew_gravity_y_scale * rew_gravity_y * self.step_dt,
            "feet_distance": self.cfg.rew_feet_distance_scale * rew_feet_distance * self.step_dt,
            "action_rate": self.cfg.rew_action_rate_scale * rew_action_rate * self.step_dt,
        }

        total = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf > self.max_episode_length
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _randomize_controls(self, env_ids: Sequence[int] | torch.Tensor):
        self._p_gains[env_ids] = float(self.cfg.kp)
        self._d_gains[env_ids] = float(self.cfg.kd)
        self._motor_offsets[env_ids] = 0.0
        self._motor_strengths[env_ids] = 1.0
        if not self.cfg.randomize_controls:
            return

        kp_min, kp_max = self.cfg.kp_scale_range
        kd_min, kd_max = self.cfg.kd_scale_range
        offset_min, offset_max = self.cfg.motor_offset_range
        self._p_gains[env_ids] *= torch.empty_like(self._p_gains[env_ids]).uniform_(kp_min, kp_max)
        self._d_gains[env_ids] *= torch.empty_like(self._d_gains[env_ids]).uniform_(kd_min, kd_max)
        self._motor_offsets[env_ids] = torch.empty_like(self._motor_offsets[env_ids]).uniform_(offset_min, offset_max)
        if self.cfg.randomize_motor_strength:
            strength_min, strength_max = self.cfg.motor_strength_range
            self._motor_strengths[env_ids] = torch.empty_like(self._motor_strengths[env_ids]).uniform_(
                strength_min, strength_max
            )

    def _randomize_rigids(self, env_ids: Sequence[int] | torch.Tensor):
        if len(env_ids) == 0:
            return

        env_ids_cpu = torch.as_tensor(env_ids, device="cpu", dtype=torch.long)
        if not self.cfg.randomize_rigids:
            return

        if self.cfg.randomize_friction:
            friction_min, friction_max = self.cfg.friction_range
            materials = self._default_materials.clone()
            friction = torch.empty((len(env_ids_cpu), 1), device="cpu").uniform_(friction_min, friction_max)
            materials[env_ids_cpu, :, 0] = friction
            materials[env_ids_cpu, :, 1] = friction
            materials[env_ids_cpu, :, 2] = 0.0
            self._robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

        if self.cfg.randomize_base_mass:
            mass_min, mass_max = self.cfg.added_mass_range
            masses = self._robot.root_physx_view.get_masses()
            inertias = self._robot.root_physx_view.get_inertias()
            added_mass = torch.empty((len(env_ids_cpu),), device="cpu").uniform_(mass_min, mass_max)
            default_mass = self._robot.data.default_mass[env_ids_cpu, self._base_body_id].cpu()
            masses[env_ids_cpu, self._base_body_id] = (default_mass + added_mass).clamp(min=1.0e-6)
            ratios = masses[env_ids_cpu, self._base_body_id] / default_mass
            inertias[env_ids_cpu, self._base_body_id] = (
                self._robot.data.default_inertia[env_ids_cpu, self._base_body_id].cpu() * ratios.unsqueeze(-1)
            )
            self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
            self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

        if self.cfg.randomize_com_displacement:
            com_min, com_max = self.cfg.com_displacement_range
            coms = self._default_coms.clone()
            displacement = torch.empty((len(env_ids_cpu), 3), device="cpu").uniform_(com_min, com_max)
            coms[env_ids_cpu, self._base_body_id, :3] += displacement
            self._robot.root_physx_view.set_coms(coms, env_ids_cpu)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._exec_actions[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self._randomize_controls(env_ids)
        self._randomize_rigids(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + float(self.cfg.base_init_height)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device)
        root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_effort_target(torch.zeros_like(self._torques[env_ids]), joint_ids=self._joint_ids, env_ids=env_ids)

        extras = {}
        if hasattr(self, "_episode_sums"):
            for key, value in self._episode_sums.items():
                extras[f"Episode_Reward/{key}"] = torch.mean(value[env_ids]) / self.max_episode_length_s
                value[env_ids] = 0.0
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras
