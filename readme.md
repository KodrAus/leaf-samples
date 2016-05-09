# Leaf ML Samples

_Work in Progress_

This is a simple collection of samples for the [leaf](https://github.com/autumnai/leaf) deep learning framework written in [rust](https://www.rust-lang.org/).

These samples use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for a standard cuda runtime. A base image is provided in `0_docker_base`, which samples assume is available as `kodraus/rust-leaf:latest`.

## Usage

1. Install the latest `nvidia` and `cuda` drivers.
1. Install [docker-engine](https://docs.docker.com/engine/installation/).
1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#plugin-install-recommended).
1. Run `docker build -t kodraus/rust-leaf:latest 0_docker_base/Dockerfile`.
1. Build and run a sample in one of the other subdirs.

## Building

To make building inside the container faster, you can first try to build a sample locally (doesn't matter if it fails), 
and then copy the contents of your `cargo` registry (either `~/.cargo` or `~/.multirust/toolchains/nightly/cargo`)
into the local `cargo` folder (for example `2_simple_linear/cargo`).
The registry info will be synced on the container and used for building, so you don't have to wait for it every time.

## Goals

To provide some easy to follow, useful samples for `leaf` along with a container that can be used for either dev or
production hosting.

Right now, I'm just getting stuff to work so the quality is fairly low. Any contributions are most welcome :)
