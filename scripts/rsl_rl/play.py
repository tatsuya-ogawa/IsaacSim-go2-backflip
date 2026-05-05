#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from importlib.metadata import version

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a Go2 backflip RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--camera_tracking", type=int, default=1)
parser.add_argument("--camera_eye_offset", type=str, default="1.2,-1.7,0.75")
parser.add_argument("--camera_target_offset", type=str, default="0.0,0.0,0.25")
parser.add_argument("--camera_smoothing", type=float, default=0.35)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import go2_backflip.tasks  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
from go2_backflip.utils.cfg import parse_env_cfg
from go2_backflip.utils.rsl_rl_compat import sanitize_rsl_rl_cfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path


def _parse_vec3(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected a comma-separated vec3, got: {raw}")
    return tuple(float(part) for part in parts)


class TrackingCameraWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        eye_offset: tuple[float, float, float],
        target_offset: tuple[float, float, float],
        smoothing: float,
    ):
        super().__init__(env)
        self._eye_offset = eye_offset
        self._target_offset = target_offset
        self._alpha = min(max(float(smoothing), 0.01), 1.0)
        self._eye: list[float] | None = None
        self._target: list[float] | None = None

    def _root_position(self) -> list[float] | None:
        robot = getattr(self.unwrapped, "_robot", None)
        if robot is None:
            return None
        root_pos = getattr(robot.data, "root_pos_w", None)
        if root_pos is None:
            return None
        return [float(value) for value in root_pos[0].detach().cpu().tolist()]

    def _smooth(self, current: list[float] | None, desired: list[float]) -> list[float]:
        if current is None:
            return desired
        return [old + self._alpha * (new - old) for old, new in zip(current, desired)]

    def update_camera(self) -> None:
        root = self._root_position()
        if root is None:
            return
        desired_target = [root[i] + self._target_offset[i] for i in range(3)]
        desired_eye = [root[i] + self._eye_offset[i] for i in range(3)]
        self._target = self._smooth(self._target, desired_target)
        self._eye = self._smooth(self._eye, desired_eye)
        self.unwrapped.sim.set_camera_view(eye=self._eye, target=self._target)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.update_camera()
        return result

    def step(self, action):
        result = self.env.step(action)
        self.update_camera()
        return result

    def render(self):
        self.update_camera()
        return self.env.render()


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.camera_tracking:
        env = TrackingCameraWrapper(
            env,
            eye_offset=_parse_vec3(args_cli.camera_eye_offset),
            target_offset=_parse_vec3(args_cli.camera_target_offset),
            smoothing=args_cli.camera_smoothing,
        )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner_cfg = sanitize_rsl_rl_cfg(agent_cfg.to_dict())
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
