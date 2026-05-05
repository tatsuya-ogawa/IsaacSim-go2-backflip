ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
FROM ${ISAAC_LAB_IMAGE}

ARG RSL_RL_VERSION=5.0.1

ENV OMNI_KIT_ALLOW_ROOT=1
ENV PYTHONPATH=/workspace/backflip/source/go2_backflip
ENV PIP_NO_CACHE_DIR=1

RUN set -eux; \
    if [ -x /workspace/isaaclab/isaaclab.sh ]; then \
      isaac_py="/workspace/isaaclab/isaaclab.sh -p"; \
    elif [ -x /workspace/IsaacLab/isaaclab.sh ]; then \
      isaac_py="/workspace/IsaacLab/isaaclab.sh -p"; \
    else \
      isaac_py="python"; \
    fi; \
    ${isaac_py} -m pip install "rsl-rl-lib==${RSL_RL_VERSION}"

WORKDIR /workspace/backflip
