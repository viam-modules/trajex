#!/usr/bin/env bash
# Installs what's needed from apt to build and test trajex and its MoveIt 2 integration in a
# ros:lyrical-ros-base container (ROS 2 Lyrical Luth, Ubuntu 26.04). CI runs it in that image. For
# development, make a long-lived container from the same image and run this in it once, from the
# repository root. Mounting the repository at its host path keeps the paths in compiler output and
# compile_commands.json valid on the host:
#
#   docker run -d --name trajex-lyrical-moveit-dev -v "$PWD:$PWD" -w "$PWD" ros:lyrical-ros-base sleep infinity
#   docker exec trajex-lyrical-moveit-dev .github/scripts/lyrical-moveit-setup.sh --dev
#
# The image's entrypoint sources the ROS environment for `docker run`. Commands run with `docker exec`
# bypass it, so they must source it themselves:
#
#   docker exec trajex-lyrical-moveit-dev bash -c 'source /opt/ros/lyrical/setup.bash && <command>'
#
# The container doesn't restart on its own, but `docker start trajex-lyrical-moveit-dev` revives it.
# A new container, for instance after `docker rm -f`, needs this script run again.
#
# Usage: lyrical-moveit-setup.sh [--dev]
#
#   --dev  Also upgrade the image's packages and install interactive development tools. CI uses the
#          pinned image as it is.
#
# Expects to run as root, which is the default in the container and in CI.

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
