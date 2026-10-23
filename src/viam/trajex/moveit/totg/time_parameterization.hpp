#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_parameterization.hpp>
#include <moveit_msgs/msg/joint_limits.hpp>

#include <viam/trajex/moveit/totg/export.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/hertz.hpp>

namespace viam::trajex::moveit::totg {

///
/// A MoveIt time parameterization that uses trajex's TOTG.
///
/// It can stand in for MoveIt's own `trajectory_processing::TimeOptimalTrajectoryGeneration`. The algorithm is
/// trajex's, so the results won't match MoveIt's, but everything around the algorithm behaves the same way. Limits
/// come from the robot model or from the call, and scaling factors outside (0, 1] fall back to 1.0 with a warning.
/// Continuous joints are unwound, and near-duplicate waypoints are removed, always keeping the final one. A
/// trajectory that doesn't move becomes a single waypoint at rest. The result is written back as waypoints with
/// positions, velocities, accelerations and durations. Failures are logged and reported by returning false, never by
/// throwing.
///
/// Waypoints are written at a uniform spacing that divides the duration evenly, so that the last one lands at the
/// very end of the trajectory. The spacing is the largest such interval that is no longer than one period of the
/// sampling frequency.
///
/// The trajectory options are copied for each call, with the velocity and acceleration limits replaced by those of
/// the call. Anything else set in them, such as an observer or TCP limits, applies to every call, so an observer will
/// see every request this object serves.
///
/// Only the active joints of the trajectory's group are timed. Their variables become trajex's degrees of freedom,
/// in the order of the group's active joints.
///
class VIAM_TRAJEX_MOVEIT_TOTG_API time_parameterization final : public trajectory_processing::TimeParameterization {
   public:
    ///
    /// Default sampling frequency, matching MoveIt's default of one waypoint every 0.1 s.
    ///
    static constexpr double k_default_sampling_frequency_hz = 10.0;

    ///
    /// Default waypoint deduplication tolerance, matching MoveIt's default `min_angle_change`.
    ///
    static constexpr double k_default_deduplication_tolerance = 1e-3;

    ///
    /// Creates a time parameterization.
    ///
    /// @param path_options Options for building the path through the waypoints
    /// @param deduplication_tolerance Distance in configuration space units. A waypoint within this distance of the
    ///        last waypoint kept, in every joint, is dropped.
    /// @param trajectory_options Trajectory options to copy for each call, as described in the class documentation
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
    /// @return true on success. On failure, logs the reason and returns false.
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
    /// @return true on success. On failure, logs the reason and returns false.
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
    /// @param joint_limits Joint limits, of which only those flagged as present are used
    /// @param max_velocity_scaling_factor Factor in (0, 1] applied to every velocity limit
    /// @param max_acceleration_scaling_factor Factor in (0, 1] applied to every acceleration limit
    /// @return true on success. On failure, logs the reason and returns false.
    ///
    bool computeTimeStamps(robot_trajectory::RobotTrajectory& trajectory,
                           const std::vector<moveit_msgs::msg::JointLimits>& joint_limits,
                           double max_velocity_scaling_factor = 1.0,
                           double max_acceleration_scaling_factor = 1.0) const override;

   private:
    VIAM_TRAJEX_MOVEIT_TOTG_HIDDEN bool compute_(robot_trajectory::RobotTrajectory& trajectory,
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
