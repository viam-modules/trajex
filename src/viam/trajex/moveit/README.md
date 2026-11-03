# viam_trajex_moveit

A ROS 2 package that lets MoveIt 2 time parameterize planned trajectories with trajex, Viam's implementation of the
Kunz & Stilman time-optimal trajectory generation algorithm (TOTG). MoveIt's own TOTG descends from the same
reference implementation, but trajex is a separate implementation, and its results won't match MoveIt's.

The package is new, and the adapters and their parameters may still change.

## Requirements

We build and test against ROS 2 Lyrical Luth on Ubuntu 26.04, with MoveIt 2.15 from the ROS apt repository, on amd64
and arm64. trajex is C++20, which Ubuntu 26.04's compiler handles. Older ROS distributions aren't supported.

## Building it in a workspace

Clone the trajex repository into your workspace and build as usual:

    cd ~/ws/src
    git clone https://github.com/viam-modules/trajex
    cd ~/ws
    rosdep install --from-paths src --ignore-src -y
    colcon build
    source install/setup.bash

The repository's submodule isn't needed, since this package doesn't build the reference implementation it holds.

Every colcon command in the workspace will then warn that `src/trajex` "has type or name but is incomplete". The
warning is harmless, and comes from a `colcon.pkg` file at the repository root. The root is a plain CMake project
that builds all of trajex, and without that file colcon would take it for a package and never find this one inside
it.

trajex's own CMake package is installed into the workspace too, so `find_package(trajex)` works there.

## Choosing an adapter

A MoveIt planning pipeline runs its response adapters on the planner's trajectory, and one of them is usually
MoveIt's `default_planning_response_adapters/AddTimeOptimalParameterization`. To use trajex, replace it in the
pipeline's `response_adapters` with one of this package's adapters:

- `viam_trajex_moveit/AddTotgTimeParameterizationCompat` reads the same parameters as MoveIt's adapter, so an
  existing configuration carries over unchanged.
- `viam_trajex_moveit/AddTotgTimeParameterization` reads trajex's own parameters under `trajex.totg`, which
  expose more of trajex's options.

The parameters go under the pipeline's name, since that's where MoveIt hands them to its adapters. For an OMPL
pipeline named `ompl`, using the trajex adapter with blending turned on, that looks like this:

    ompl:
      planning_plugins:
        - ompl_interface/OMPLPlanner
      response_adapters:
        - viam_trajex_moveit/AddTotgTimeParameterization
        - default_planning_response_adapters/ValidateSolution
      trajex:
        totg:
          path:
            max_blend_deviation: 0.05
          sampling:
            frequency: 50.0

The adapters read their parameters on every plan, so changes made while the node runs take effect on the next one.
When an adapter starts, it declares any of its parameters that aren't already declared, with their defaults, so they
all show up in `ros2 param list`.

### The compat adapter

| Parameter | Default | Meaning |
|---|---|---|
| `totg.path_tolerance` | 0.1 | How far a blend may deviate from the corner it rounds, in rad for revolute joints and m for prismatic ones. |
| `totg.resample_dt` | 0.1 | The longest interval between waypoints in the result, in s. |
| `totg.min_angle_change` | 0.001 | Waypoints that move no joint further than this from the previous one are dropped, in rad or m. |

These are MoveIt's names and defaults, with MoveIt's meanings. trajex can also merge nearly collinear waypoints,
which MoveIt doesn't, so this adapter turns that off.

### The trajex adapter

| Parameter | Default | Meaning |
|---|---|---|
| `trajex.totg.path.max_blend_deviation` | 0 | How far a blend may deviate from the corner it rounds, in rad or m. 0 turns blending off. |
| `trajex.totg.path.max_linear_deviation` | 0 | The maximum deviation allowed when merging nearly collinear waypoints, in rad or m. 0 turns merging off. |
| `trajex.totg.path.min_blend_curvature` | 1e-5 | The minimum curvature a blend may have, in 1/rad or 1/m. |
| `trajex.totg.path.max_blend_curvature` | 1e5 | The maximum curvature a blend may have, in 1/rad or 1/m. |
| `trajex.totg.path.deduplication_tolerance` | 0.001 | Waypoints that move no joint further than this from the previous one are dropped, in rad or m. |
| `trajex.totg.trajectory.delta` | 0.001 | The integration time step, in s. |
| `trajex.totg.trajectory.epsilon` | 1e-6 | The tolerance trajex uses for numerical comparisons. |
| `trajex.totg.sampling.frequency` | 10 | The minimum rate at which waypoints are written into the result, in Hz. |

Unlike MoveIt's adapter, this one only blends if you ask it to. A blend rounds a corner of the planned path so the
arm doesn't have to stop there, but it cuts inside the corner, so it can leave the collision-free path the planner
found.
Without blending the trajectory follows the planner's path exactly, stopping at every corner. On a simulated Panda
working around an obstacle, `ValidateSolution` occasionally rejected blended trajectories for collisions, and never
an unblended one. If you turn blending on, keep `ValidateSolution` after the adapter.

## What the adapters do

Velocity and acceleration limits come from the robot model, usually from `joint_limits.yaml`, scaled by the
request's scaling factors. Planning fails if a joint lacks either limit, so check that your robot has acceleration
limits, since many robot descriptions don't. Scaling factors outside (0, 1] are replaced by 1.0, with a warning.

Only the active joints of the trajectory's group are timed. Continuous joints are unwound, waypoints that barely
move are dropped (always keeping the last), and a trajectory that never moves comes back as its starting state at
rest, as with MoveIt's adapter. The result's waypoints are evenly spaced, no more than one period of the sampling
frequency apart, with the last at the very end.

If trajex can't time the trajectory, the adapter logs why and the plan fails. The adapters never throw.

## Using it from C++

To time trajectories yourself rather than through a planning pipeline, link `viam::trajex::moveit::totg`:

    find_package(viam_trajex_moveit REQUIRED)
    target_link_libraries(my_node PRIVATE viam::trajex::moveit::totg)

Its `viam::trajex::moveit::totg::time_parameterization` is a `trajectory_processing::TimeParameterization`, usable
wherever MoveIt's is:

    #include <viam/trajex/moveit/totg/time_parameterization.hpp>

    viam::trajex::moveit::totg::time_parameterization timing;
    if (!timing.computeTimeStamps(trajectory, 0.5, 0.5)) {
        // The reason has been logged.
    }

As with MoveIt's, other overloads of `computeTimeStamps` take limits by joint variable name or as
`moveit_msgs::msg::JointLimits`, falling back to the robot model for joints they don't cover. The constructor takes
trajex's path and trajectory options, the deduplication tolerance and the sampling frequency, all defaulted and
documented in `time_parameterization.hpp`. Code including that header must build as C++20, which isn't ROS's
default, so set `CMAKE_CXX_STANDARD` to 20 in your package.

## Working on it

The package builds trajex from the repository root rather than using an installed copy. The time parameterization
class is part of trajex itself, built when `VIAM_TRAJEX_ENABLE_MOVEIT` is on, and this package wraps it in pluginlib
adapters.

We develop in a long-lived container made from the stock `ros:lyrical-ros-base` image, with the repository mounted at
its host path. `.github/scripts/lyrical-moveit-setup.sh` sets it up, and its header has the commands for creating the
container and running things in it. The `MoveIt Integration` workflow, `.github/workflows/moveit.yml`, runs the same
script in the same image, then builds and tests trajex with MoveIt enabled, and this package. It runs daily, on pull
requests that change this directory, and on demand.

To run this package's tests in a workspace:

    colcon test --packages-select viam_trajex_moveit --event-handlers console_direct+
    colcon test-result --verbose
