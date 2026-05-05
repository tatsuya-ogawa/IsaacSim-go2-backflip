from __future__ import annotations

import argparse
from typing import Any


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    from go2_backflip.utils.cfg import add_rsl_rl_args as _add_rsl_rl_args

    _add_rsl_rl_args(parser)


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> Any:
    from go2_backflip.utils.cfg import parse_rsl_rl_cfg as _parse_rsl_rl_cfg

    return _parse_rsl_rl_cfg(task_name, args_cli)


def update_rsl_rl_cfg(agent_cfg: Any, args_cli: argparse.Namespace) -> Any:
    from go2_backflip.utils.cfg import update_rsl_rl_cfg as _update_rsl_rl_cfg

    return _update_rsl_rl_cfg(agent_cfg, args_cli)

__all__ = ["add_rsl_rl_args", "parse_rsl_rl_cfg", "update_rsl_rl_cfg"]
