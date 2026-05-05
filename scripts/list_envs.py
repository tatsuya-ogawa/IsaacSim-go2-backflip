#!/usr/bin/env python3
from __future__ import annotations

import gymnasium as gym

import go2_backflip.tasks  # noqa: F401


def main() -> None:
    for spec in sorted(gym.registry.values(), key=lambda item: item.id):
        if "Backflip" in spec.id or "Go2" in spec.id:
            print(spec.id)


if __name__ == "__main__":
    main()
