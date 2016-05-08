# Leaf ML Samples

This is a simple collection of samples for the [leaf](https://github.com/autumnai/leaf) deep learning framework written in [rust](https://www.rust-lang.org/).

These samples use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for a standard cuda runtime. A base image is provided in `0_docker_base`, which samples assume is available as `kodraus/rust-leaf:latest`.

## Usage

1. Install the latest `nvidia` and `cuda` drivers.
1. Install [docker-engine](https://docs.docker.com/engine/installation/).
1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#plugin-install-recommended).
1. Run `docker build -t kodraus/rust-leaf:latest 0_docker_base/Dockerfile`.
1. Build and run a sample in one of the other subdirs.