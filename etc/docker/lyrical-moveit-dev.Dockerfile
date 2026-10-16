# Development environment for building trajex against MoveIt 2 on ROS 2 Lyrical Luth (Ubuntu
# 26.04). The source tree is bind-mounted, not copied in.
#
#   docker build -t trajex-lyrical-moveit-dev -f etc/docker/lyrical-moveit-dev.Dockerfile etc/docker
#   docker run --rm -it -v "$PWD:$PWD" -w "$PWD" trajex-lyrical-moveit-dev
#
# Mounting the repository at its host path keeps paths in compiler output and in
# compile_commands.json valid on the host as well.
#
# The image's entrypoint sources the ROS environment for `docker run`. Commands run with
# `docker exec` bypass it, so they must source /opt/ros/lyrical/setup.bash themselves.

FROM ros:lyrical-ros-base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get -y --no-install-recommends install \
    build-essential \
    ccache \
    clang \
    clang-tidy \
    clangd \
    gdb \
    git \
    less \
    libboost-test-dev \
    libeigen3-dev \
    libjsoncpp-dev \
    libxtensor-dev \
    ninja-build \
    ros-dev-tools \
    ros-lyrical-moveit-core \
    ros-lyrical-moveit-planners-ompl \
    ros-lyrical-moveit-resources-panda-moveit-config \
    ros-lyrical-moveit-ros-planning \
    xtl-dev && \
    rm -rf /var/lib/apt/lists/*
