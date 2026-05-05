COMPOSE ?= docker compose
COMPOSE_RUN ?= $(COMPOSE) run --rm --build

.PHONY: bootstrap build list-envs train resume export-policy play-video

bootstrap: build

build:
	$(COMPOSE) build backflip

list-envs:
	$(COMPOSE_RUN) backflip scripts/list_envs.sh

train:
	$(COMPOSE_RUN) -e TASK_NAME=Unitree-Go2-Backflip-Isaac-v0 -e RUN_NAME=$${RUN_NAME:-backflip} backflip scripts/train.sh

resume:
	$(COMPOSE_RUN) -e TASK_NAME=Unitree-Go2-Backflip-Isaac-v0 -e RUN_NAME=$${RUN_NAME:-backflip-resume} -e RESUME=$${RESUME:-1} backflip scripts/train.sh

export-policy:
	$(COMPOSE_RUN) -e TASK_NAME=Unitree-Go2-Backflip-Isaac-v0 backflip scripts/export_policy.sh

play-video:
	$(COMPOSE_RUN) -e TASK_NAME=Unitree-Go2-Backflip-Isaac-v0 backflip scripts/play_video.sh
