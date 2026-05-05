import gymnasium as gym

gym.register(
    id="Unitree-Go2-Backflip-Isaac-v0",
    entry_point="go2_backflip.tasks.backflip_isaac.backflip_isaac_env:Go2BackflipIsaacEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_backflip.tasks.backflip_isaac.backflip_isaac_env_cfg:Go2BackflipIsaacEnvCfg",
        "play_env_cfg_entry_point": "go2_backflip.tasks.backflip_isaac.backflip_isaac_env_cfg:Go2BackflipIsaacPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "go2_backflip.tasks.backflip_isaac.rsl_rl_ppo_cfg:BackflipIsaacPPORunnerCfg",
    },
)
