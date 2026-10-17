#!/usr/bin/env bash
# Prepares a ros:lyrical-ros-base container to build and test trajex and its MoveIt 2 integration on
# ROS 2 Lyrical Luth (Ubuntu 26.04), by installing what the build needs from apt. CI runs it in that
# image. For development, make a long-lived container from the same image, with the repository
# mounted at its host path so that paths in compiler output and compile_commands.json stay valid on
# the host, and run this in it once, from the repository root:
#
#   docker run -d --name trajex-lyrical-moveit-dev -v "$PWD:$PWD" -w "$PWD" ros:lyrical-ros-base sleep infinity
#   docker exec trajex-lyrical-moveit-dev .github/scripts/lyrical-moveit-setup.sh --dev
#
# The image's entrypoint sources the ROS environment for `docker run`. Commands run with `docker exec`
# bypass it, so they must source it themselves:
#
#   docker exec trajex-lyrical-moveit-dev bash -c 'source /opt/ros/lyrical/setup.bash && <command>'
#
# The container does not restart on its own; `docker start trajex-lyrical-moveit-dev` revives it. A
# container made anew, for instance after `docker rm -f`, needs this script run again.
#
# Usage: lyrical-moveit-setup.sh [--dev]
#
#   --dev  Also upgrade the image's packages and install interactive development tools. CI takes the
#          pinned base image as it is.
#
# Runs as root, as both the container and CI do.

set -euxo pipefail

dev=false
case "${1-}" in
    "") ;;
    --dev) dev=true ;;
    *)
        echo "usage: $0 [--dev]" >&2
        exit 2
        ;;
esac

export DEBIAN_FRONTEND=noninteractive

packages=(
    build-essential
    git
    libboost-test-dev
    libeigen3-dev
    libjsoncpp-dev
    libxtensor-dev
    ninja-build
    ros-dev-tools
    ros-lyrical-moveit-core
    ros-lyrical-moveit-planners-ompl
    ros-lyrical-moveit-resources-panda-moveit-config
    ros-lyrical-moveit-ros-planning
    xtl-dev
)

apt-get update
if $dev; then
    apt-get -y dist-upgrade
    packages+=(ccache clang clang-tidy clangd gdb less)
fi
apt-get -y --no-install-recommends install "${packages[@]}"
rm -rf /var/lib/apt/lists/*
