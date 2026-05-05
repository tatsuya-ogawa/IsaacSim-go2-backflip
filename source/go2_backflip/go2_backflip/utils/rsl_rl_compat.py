from __future__ import annotations

import math
import os
from typing import Any

import torch


def sanitize_rsl_rl_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate Isaac Lab 2.x runner configs into the rsl-rl 5.x shape."""
    policy_cfg = cfg.pop("policy", None)
    obs_groups = cfg.get("obs_groups")
    if isinstance(obs_groups, dict):
        if "actor" not in obs_groups and "policy" in obs_groups:
            obs_groups["actor"] = obs_groups["policy"]
        if "critic" not in obs_groups:
            obs_groups["critic"] = obs_groups.get("actor", ["policy"])
    else:
        cfg["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}

    if isinstance(policy_cfg, dict) and "actor" not in cfg:
        activation = policy_cfg.get("activation", "elu")
        init_std = policy_cfg.get("init_noise_std", 1.0)
        std_type = policy_cfg.get("noise_std_type", "scalar")
        distribution_class = (
            "HeteroscedasticGaussianDistribution"
            if policy_cfg.get("state_dependent_std", False)
            else "GaussianDistribution"
        )
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
            "activation": activation,
            "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
            "distribution_cfg": {
                "class_name": distribution_class,
                "init_std": init_std,
                "std_type": std_type,
            },
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
            "activation": activation,
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }

    for model_key in ("actor", "critic"):
        model_cfg = cfg.get(model_key)
        if isinstance(model_cfg, dict):
            for key in ("stochastic", "distribution_cfg"):
                if model_key == "critic" or key == "stochastic":
                    model_cfg.pop(key, None)
    return cfg


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _clamp_action_std(actor: Any, min_std: float, max_std: float) -> None:
    distribution = getattr(actor, "distribution", None)
    if distribution is None:
        return
    with torch.no_grad():
        if hasattr(distribution, "std_param"):
            distribution.std_param.clamp_(min=min_std, max=max_std)
        if hasattr(distribution, "log_std_param"):
            distribution.log_std_param.clamp_(min=math.log(min_std), max=math.log(max_std))
        current_distribution = getattr(distribution, "_distribution", None)
        if current_distribution is not None:
            mean = current_distribution.mean
            if hasattr(distribution, "std_param"):
                std = distribution.std_param.expand_as(mean)
            else:
                std = torch.exp(distribution.log_std_param).expand_as(mean)
            distribution._distribution = torch.distributions.Normal(mean, std)


def install_action_std_clamp(runner: Any) -> None:
    min_std = _env_float("BACKFLIP_ACTION_STD_MIN", 0.05)
    max_std = _env_float("BACKFLIP_ACTION_STD_MAX", 1.0)
    if max_std <= 0.0 or min_std <= 0.0 or min_std > max_std:
        raise ValueError("Expected 0 < BACKFLIP_ACTION_STD_MIN <= BACKFLIP_ACTION_STD_MAX.")

    actor = getattr(runner.alg, "actor", None)
    if actor is None:
        print("[WARN] Skipping action std clamp: runner.alg.actor is unavailable.")
        return

    original_update = runner.alg.update

    def update_with_action_std_clamp(*args, **kwargs):
        result = original_update(*args, **kwargs)
        _clamp_action_std(actor, min_std, max_std)
        return result

    runner.alg.update = update_with_action_std_clamp
    _clamp_action_std(actor, min_std, max_std)
    print(f"[INFO] Clamping actor action std to [{min_std}, {max_std}].")
