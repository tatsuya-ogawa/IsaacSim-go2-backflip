from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg


def parse_env_cfg(
    task_name: str,
    device: str = "cuda:0",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
    entry_point_key: str = "env_cfg_entry_point",
) -> "ManagerBasedRLEnvCfg | DirectRLEnvCfg":
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    cfg = load_cfg_from_registry(task_name, entry_point_key)
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for task '{task_name}' is not a class config.")

    cfg.sim.device = device
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    group.add_argument("--experiment_name", type=str, default=None)
    group.add_argument("--run_name", type=str, default=None)
    group.add_argument("--resume", action="store_true", default=False)
    group.add_argument("--load_run", type=str, default=None)
    group.add_argument("--checkpoint", type=str, default=None)
    group.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    group.add_argument("--log_project_name", type=str, default=None)


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> Any:
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if cfg.experiment_name == "":
        cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")
    return update_rsl_rl_cfg(cfg, args_cli)


def update_rsl_rl_cfg(agent_cfg: Any, args_cli: argparse.Namespace) -> Any:
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    return agent_cfg
