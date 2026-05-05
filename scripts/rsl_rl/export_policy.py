#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export a deterministic Go2 backflip actor as TorchScript.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path


def _resolve_checkpoint(agent_cfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = _resolve_checkpoint(agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner_cfg = sanitize_rsl_rl_cfg(agent_cfg.to_dict())
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    actor = runner.get_inference_policy(device=env.unwrapped.device)

    if not hasattr(actor, "as_jit"):
        raise TypeError("Expected an RSL-RL actor with an as_jit() export method.")

    policy = actor.as_jit().to("cpu").eval()
    input_dim = int(getattr(actor, "obs_dim", 60))
    action_dim = int(getattr(env.unwrapped, "num_actions", 12))
    example = torch.zeros(1, input_dim, dtype=torch.float32)
    with torch.inference_mode():
        scripted = torch.jit.script(policy)
        output = scripted(example)

    if tuple(output.shape) != (1, action_dim):
        raise RuntimeError(f"Expected exported policy output shape (1, {action_dim}), got {tuple(output.shape)}.")

    output_path = args_cli.output or os.path.join(os.path.dirname(resume_path), "policy.pt")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scripted.save(output_path)
    print(f"[INFO] Exported deterministic actor mean to: {output_path}")
    print(f"[INFO] TorchScript input shape:  (1, {input_dim})")
    print(f"[INFO] TorchScript output shape: (1, {action_dim})")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
