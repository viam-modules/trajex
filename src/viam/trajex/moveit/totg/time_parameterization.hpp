#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_parameterization.hpp>
#include <moveit_msgs/msg/joint_limits.hpp>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/hertz.hpp>

namespace viam::trajex::moveit::totg {

///
/// MoveIt time parameterization backed by trajex's TOTG.
///
/// A replacement for MoveIt's own `trajectory_processing::TimeOptimalTrajectoryGeneration`. The results are
/// trajex's, not MoveIt's. Around the algorithm it does what MoveIt's does: limits come from the robot model or
/// from the call, scaling factors outside (0, 1] fall back to 1.0 with a warning, continuous joints are unwound,
/// near-duplicate waypoints are removed while keeping the final one, a trajectory that does not move becomes a single
/// stationary waypoint, and the result is written back as waypoints with positions, velocities, accelerations and
/// durations. Failures are logged and reported by returning false, never by throwing.
///
/// Waypoints are written at uniform spacing with the last exactly at the end of the trajectory, so the spacing is
/// the largest interval no greater than one period of the sampling frequency that divides the duration evenly.
///
/// The trajectory options are a prototype, copied for each call with the velocity and acceleration limits replaced
/// by those of the call. Anything else set in them, such as an observer or TCP limits, applies to every call. An
/// observer is shared by every request this object serves.
///
/// Only the active joints of the trajectory's group are timed. Their variables are the degrees of freedom handed
/// to trajex, in the order of the group's active joints.
///
class time_parameterization final : public trajectory_processing::TimeParameterization {
   public:
    ///
    /// Default sampling frequency, MoveIt's default of one waypoint every 0.1 s.
    ///
    static constexpr double k_default_sampling_frequency_hz = 10.0;

    ///
    /// Default waypoint deduplication tolerance, MoveIt's default `min_angle_change`.
    ///
    static constexpr double k_default_deduplication_tolerance = 1e-3;

    ///
    /// Creates a time parameterization.
    ///
    /// @param path_options Options for building the path through the waypoints
    /// @param deduplication_tolerance Waypoints within this distance of the previous kept waypoint in every joint
    ///        are dropped, in configuration space units
    /// @param trajectory_options Prototype trajectory options; see the class documentation
    /// @param sampling_frequency Minimum rate at which waypoints are written back
    /// @throws std::invalid_argument if the deduplication tolerance is negative or not finite, or the sampling
    ///         frequency is not positive and finite
    ///
    explicit time_parameterization(trajex::totg::path::options path_options = {},
                                   double deduplication_tolerance = k_default_deduplication_tolerance,
                                   trajex::totg::trajectory::options trajectory_options = {},
                                   types::hertz sampling_frequency = types::hertz{k_default_sampling_frequency_hz});

    ///
    /// Time parameterizes a trajectory using the robot model's limits.
    ///
    /// @param trajectory Trajectory to time parameterize, replaced by the result on success
    /// @param max_velocity_scaling_factor Factor in (0, 1] applied to every velocity limit
    /// @param max_acceleration_scaling_factor Factor in (0, 1] applied to every acceleration limit
    /// @return true on success, false on failure, which is logged
    ///
    bool computeTimeStamps(robot_trajectory::RobotTrajectory& trajectory,
                           double max_velocity_scaling_factor = 1.0,
                           double max_acceleration_scaling_factor = 1.0) const override;

    ///
    /// Time parameterizes a trajectory using limits given by joint variable name, falling back to the robot model.
    ///
    /// @param trajectory Trajectory to time parameterize, replaced by the result on success
    /// @param velocity_limits Velocity limits by joint variable name
    /// @param acceleration_limits Acceleration limits by joint variable name
    /// @param max_velocity_scaling_factor Factor in (0, 1] applied to every velocity limit
    /// @param max_acceleration_scaling_factor Factor in (0, 1] applied to every acceleration limit
    /// @return true on success, false on failure, which is logged
    ///
    bool computeTimeStamps(robot_trajectory::RobotTrajectory& trajectory,
                           const std::unordered_map<std::string, double>& velocity_limits,
                           const std::unordered_map<std::string, double>& acceleration_limits,
                           double max_velocity_scaling_factor = 1.0,
                           double max_acceleration_scaling_factor = 1.0) const override;

    ///
    /// Time parameterizes a trajectory using limits from joint limit messages, falling back to the robot model.
    ///
    /// @param trajectory Trajectory to time parameterize, replaced by the result on success
    /// @param joint_limits Joint limits; only those flagged as present are used
    /// @param max_velocity_scaling_factor Factor in (0, 1] applied to every velocity limit
    /// @param max_acceleration_scaling_factor Factor in (0, 1] applied to every acceleration limit
    /// @return true on success, false on failure, which is logged
    ///
    bool computeTimeStamps(robot_trajectory::RobotTrajectory& trajectory,
                           const std::vector<moveit_msgs::msg::JointLimits>& joint_limits,
                           double max_velocity_scaling_factor = 1.0,
                           double max_acceleration_scaling_factor = 1.0) const override;

   private:
    bool compute_(robot_trajectory::RobotTrajectory& trajectory,
                  const std::unordered_map<std::string, double>& velocity_limits,
                  const std::unordered_map<std::string, double>& acceleration_limits,
                  double max_velocity_scaling_factor,
                  double max_acceleration_scaling_factor) const;

    trajex::totg::path::options path_options_;
    double deduplication_tolerance_;
    trajex::totg::trajectory::options trajectory_options_;
    types::hertz sampling_frequency_;
};

}  // namespace viam::trajex::moveit::totg
