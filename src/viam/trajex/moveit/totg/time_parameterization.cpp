#include <viam/trajex/moveit/totg/time_parameterization.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>

#include <moveit/robot_model/joint_model.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/utils/logger.hpp>

#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/totg/waypoint_utils.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::moveit::totg {

namespace {

// Since we're inside `viam::trajex::moveit`, MoveIt's own `moveit` namespace has to be named from the root.
namespace mc = ::moveit::core;

rclcpp::Logger logger() {
    return ::moveit::getLogger("viam.trajex.moveit.totg");
}

// A variable of one of the group's active joints, which is one degree of freedom for trajex.
struct active_variable {
    std::size_t state_index;
    const std::string* name;
    const mc::VariableBounds* bounds;
};

std::vector<active_variable> active_variables(const mc::JointModelGroup& group) {
    std::vector<active_variable> result;
    for (const auto* joint : group.getActiveJointModels()) {
        for (std::size_t i = 0; i < joint->getVariableCount(); ++i) {
            result.push_back({joint->getFirstVariableIndex() + i, &joint->getVariableNames()[i], &joint->getVariableBounds()[i]});
        }
    }
    return result;
}

double verified_scaling_factor(double requested, const char* kind) {
    if (requested > 0.0 && requested <= 1.0) {
        return requested;
    }
    RCLCPP_WARN(logger(), "Invalid max_%s_scaling_factor %f specified, defaulting to 1.0 instead.", kind, requested);
    return 1.0;
}

// Uses the limit given for the variable by name if there is one, and otherwise the robot model's bound. If there's
// no usable limit, logs why and returns nothing.
std::optional<double> resolve_limit(const active_variable& variable,
                                    const std::unordered_map<std::string, double>& given,
                                    bool model_bounded,
                                    double model_min,
                                    double model_max,
                                    double scaling_factor,
                                    const char* kind) {
    double limit;
    if (const auto it = given.find(*variable.name); it != given.end()) {
        limit = it->second;
    } else if (model_bounded) {
        limit = std::min(std::abs(model_min), std::abs(model_max));
    } else {
        RCLCPP_ERROR(logger(),
                     "No %s limit was defined for joint variable '%s'. Define one in the URDF or joint_limits.yaml.",
                     kind,
                     variable.name->c_str());
        return std::nullopt;
    }
    if (!std::isfinite(limit) || limit < 0.0) {
        RCLCPP_ERROR(logger(), "Invalid %s limit %f for joint variable '%s'.", kind, limit, variable.name->c_str());
        return std::nullopt;
    }
    return limit * scaling_factor;
}

}  // namespace

time_parameterization::time_parameterization(trajex::totg::path::options path_options,
                                             double deduplication_tolerance,
                                             trajex::totg::trajectory::options trajectory_options,
                                             types::hertz sampling_frequency)
    : path_options_(std::move(path_options)),
      deduplication_tolerance_(deduplication_tolerance),
      trajectory_options_(std::move(trajectory_options)),
      sampling_frequency_(sampling_frequency) {
    if (!std::isfinite(deduplication_tolerance_) || deduplication_tolerance_ < 0.0) {
        throw std::invalid_argument{"time_parameterization: deduplication tolerance must be finite and non-negative"};
    }
    if (!std::isfinite(sampling_frequency_.value) || sampling_frequency_.value <= 0.0) {
        throw std::invalid_argument{"time_parameterization: sampling frequency must be finite and positive"};
    }
}

bool time_parameterization::computeTimeStamps(robot_trajectory::RobotTrajectory& trajectory,
                                              double max_velocity_scaling_factor,
                                              double max_acceleration_scaling_factor) const {
    return compute_(trajectory, {}, {}, max_velocity_scaling_factor, max_acceleration_scaling_factor);
}

bool time_parameterization::computeTimeStamps(robot_trajectory::RobotTrajectory& trajectory,
                                              const std::unordered_map<std::string, double>& velocity_limits,
                                              const std::unordered_map<std::string, double>& acceleration_limits,
                                              double max_velocity_scaling_factor,
                                              double max_acceleration_scaling_factor) const {
    return compute_(trajectory, velocity_limits, acceleration_limits, max_velocity_scaling_factor, max_acceleration_scaling_factor);
}

bool time_parameterization::computeTimeStamps(robot_trajectory::RobotTrajectory& trajectory,
                                              const std::vector<moveit_msgs::msg::JointLimits>& joint_limits,
                                              double max_velocity_scaling_factor,
                                              double max_acceleration_scaling_factor) const {
    std::unordered_map<std::string, double> velocity_limits;
    std::unordered_map<std::string, double> acceleration_limits;
    for (const auto& limit : joint_limits) {
        if (limit.has_velocity_limits) {
            velocity_limits[limit.joint_name] = limit.max_velocity;
        }
        if (limit.has_acceleration_limits) {
            acceleration_limits[limit.joint_name] = limit.max_acceleration;
        }
    }
    return compute_(trajectory, velocity_limits, acceleration_limits, max_velocity_scaling_factor, max_acceleration_scaling_factor);
}

bool time_parameterization::compute_(robot_trajectory::RobotTrajectory& trajectory,
                                     const std::unordered_map<std::string, double>& velocity_limits,
                                     const std::unordered_map<std::string, double>& acceleration_limits,
                                     double max_velocity_scaling_factor,
                                     double max_acceleration_scaling_factor) const try {
    if (trajectory.empty()) {
        return true;
    }

    const mc::JointModelGroup* const group = trajectory.getGroup();
    if (!group) {
        RCLCPP_ERROR(logger(), "The trajectory has no group; the planner did not set the group it planned for.");
        return false;
    }

    // We resolve all the limits before touching the trajectory, so that a failure here leaves it as it was handed to
    // us. A later failure leaves it unwound, but an unwound trajectory still describes the same motion.
    const double velocity_scaling = verified_scaling_factor(max_velocity_scaling_factor, "velocity");
    const double acceleration_scaling = verified_scaling_factor(max_acceleration_scaling_factor, "acceleration");

    const auto variables = active_variables(*group);
    const std::size_t dof = variables.size();
    xvector<> max_velocity(xvector<>::shape_type{dof});
    xvector<> max_acceleration(xvector<>::shape_type{dof});
    for (std::size_t i = 0; i < dof; ++i) {
        const auto& bounds = *variables[i].bounds;
        const auto velocity = resolve_limit(variables[i],
                                            velocity_limits,
                                            bounds.velocity_bounded_,
                                            bounds.min_velocity_,
                                            bounds.max_velocity_,
                                            velocity_scaling,
                                            "velocity");
        const auto acceleration = resolve_limit(variables[i],
                                                acceleration_limits,
                                                bounds.acceleration_bounded_,
                                                bounds.min_acceleration_,
                                                bounds.max_acceleration_,
                                                acceleration_scaling,
                                                "acceleration");
        if (!velocity || !acceleration) {
            return false;
        }
        max_velocity(i) = *velocity;
        max_acceleration(i) = *acceleration;
    }

    if (std::ranges::any_of(group->getActiveJointModels(),
                            [](const auto* joint) { return joint->getType() == mc::JointModel::PRISMATIC; }) &&
        std::ranges::any_of(group->getActiveJointModels(),
                            [](const auto* joint) { return joint->getType() == mc::JointModel::REVOLUTE; })) {
        RCLCPP_WARN(logger(),
                    "The group mixes revolute and prismatic joints. Path deviations and tolerances are in "
                    "configuration space and will not mean the same thing for both.");
    }

    // We build the path in configuration space, where a continuous joint that wraps around would appear to jump by a
    // full turn.
    trajectory.unwind();

    const std::size_t waypoint_count = trajectory.getWayPointCount();
    xmatrix<> waypoints(xmatrix<>::shape_type{waypoint_count, dof});
    for (std::size_t w = 0; w < waypoint_count; ++w) {
        const auto& state = trajectory.getWayPoint(w);
        for (std::size_t i = 0; i < dof; ++i) {
            waypoints(w, i) = state.getVariablePosition(static_cast<int>(variables[i].state_index));
        }
    }

    // We write every waypoint from this state, so that variables outside the group carry through unchanged.
    mc::RobotState state(trajectory.getWayPoint(0));
    state.zeroVelocities();
    state.zeroAccelerations();

    const auto unique = trajex::totg::deduplicate_waypoints(trajex::totg::waypoint_accumulator{waypoints}, deduplication_tolerance_);

    // A trajectory that never moves has no path to follow. MoveIt returns the starting state, at rest, so we do too.
    if (unique.size() < 2) {
        trajectory.clear();
        trajectory.addSuffixWayPoint(state, 0.0);
        return true;
    }

    auto options = trajectory_options_;
    options.max_velocity = std::move(max_velocity);
    options.max_acceleration = std::move(max_acceleration);
    const auto timed = trajex::totg::trajectory::create(trajex::totg::path::create(unique, path_options_), std::move(options));

    // Assemble the result before replacing the input, so that nothing is lost if sampling throws partway.
    std::vector<std::pair<mc::RobotState, double>> result;
    Eigen::VectorXd positions(static_cast<Eigen::Index>(dof));
    double previous_time = 0.0;
    for (const auto& sample : timed.samples(trajex::totg::uniform_sampler::quantized_for_trajectory(timed, sampling_frequency_))) {
        std::copy_n(sample.configuration.data(), dof, positions.data());
        state.setJointGroupActivePositions(group, positions);
        for (std::size_t i = 0; i < dof; ++i) {
            const auto index = static_cast<int>(variables[i].state_index);
            state.setVariableVelocity(index, sample.velocity(i));
            state.setVariableAcceleration(index, sample.acceleration(i));
        }
        const double time = sample.time.count();
        result.emplace_back(state, time - previous_time);
        previous_time = time;
    }

    trajectory.clear();
    for (const auto& [waypoint, duration] : result) {
        trajectory.addSuffixWayPoint(waypoint, duration);
    }
    return true;
} catch (const std::exception& e) {
    RCLCPP_ERROR(logger(), "trajex time parameterization failed: %s", e.what());
    return false;
} catch (...) {
    RCLCPP_ERROR(logger(), "trajex time parameterization failed with an unknown exception");
    return false;
}

}  // namespace viam::trajex::moveit::totg
