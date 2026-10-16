#include <viam/trajex/moveit/totg/time_parameterization.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/utils/robot_model_test_utils.hpp>

namespace {

namespace mc = ::moveit::core;
using viam::trajex::moveit::totg::time_parameterization;

// Allowance for the integration step's overshoot of the velocity limit.
constexpr double k_limit_tolerance = 1e-3;

// MoveIt's testing model has no acceleration limits in its URDF, so its own TOTG tests add them. So do we.
mc::RobotModelPtr panda(bool with_acceleration_limits = true) {
    auto model = mc::loadTestingRobotModel("panda");
    BOOST_REQUIRE(model);
    if (with_acceleration_limits) {
        for (auto* joint : model->getActiveJointModels()) {
            auto bounds = joint->getVariableBoundsMsg();
            for (auto& bound : bounds) {
                bound.has_acceleration_limits = true;
                bound.max_acceleration = 1.0;
            }
            joint->setVariableBounds(bounds);
        }
    }
    return model;
}

struct fixture {
    mc::RobotModelPtr model = panda();
    const mc::JointModelGroup* group = model->getJointModelGroup("panda_arm");

    robot_trajectory::RobotTrajectory make(const std::vector<std::vector<double>>& waypoints) const {
        robot_trajectory::RobotTrajectory trajectory(model, group);
        mc::RobotState state(model);
        state.setToDefaultValues();
        for (const auto& waypoint : waypoints) {
            state.setJointGroupPositions(group, waypoint);
            trajectory.addSuffixWayPoint(state, 0.1);
        }
        return trajectory;
    }

    Eigen::VectorXd positions(const robot_trajectory::RobotTrajectory& trajectory, std::size_t i) const {
        Eigen::VectorXd values;
        trajectory.getWayPoint(i).copyJointGroupPositions(group, values);
        return values;
    }

    // Checks what holds for any successful result: starts at rest at time zero where the input started, ends at
    // rest where it ended, time never runs backwards, and no joint exceeds its velocity limit times the scaling.
    void check_valid(const robot_trajectory::RobotTrajectory& result,
                     const Eigen::VectorXd& start,
                     const Eigen::VectorXd& end,
                     double velocity_scaling = 1.0) const {
        BOOST_REQUIRE_GE(result.getWayPointCount(), 2U);
        BOOST_CHECK_EQUAL(result.getWayPointDurationFromPrevious(0), 0.0);
        BOOST_CHECK_LE((positions(result, 0) - start).cwiseAbs().maxCoeff(), 1e-9);
        BOOST_CHECK_LE((positions(result, result.getWayPointCount() - 1) - end).cwiseAbs().maxCoeff(), 1e-9);

        const auto& names = group->getVariableNames();
        for (std::size_t w = 0; w < result.getWayPointCount(); ++w) {
            BOOST_CHECK_GE(result.getWayPointDurationFromPrevious(w), 0.0);
            const auto& state = result.getWayPoint(w);
            for (const auto& name : names) {
                const double limit = model->getVariableBounds(name).max_velocity_ * velocity_scaling;
                BOOST_CHECK_LE(std::abs(state.getVariableVelocity(name)), limit * (1.0 + k_limit_tolerance));
            }
        }
        for (const auto& name : names) {
            BOOST_CHECK_SMALL(result.getFirstWayPoint().getVariableVelocity(name), 1e-9);
            BOOST_CHECK_SMALL(result.getLastWayPoint().getVariableVelocity(name), 1e-9);
        }
    }
};

const std::vector<double> k_start{-0.5, -1.5, 1.35, -2.51, -0.88, 0.63, 0.0};
const std::vector<double> k_end{-0.45, -1.2, 1.2, -2.4, -0.8, 0.6, 0.0};

// Several corners, from MoveIt's own TOTG tests, kept within the testing model's joint bounds.
const std::vector<std::vector<double>> k_cornered{
    {-0.5, -1.5, 1.35, -2.51, -0.88, 0.63, 0.0},
    {0.0, -1.5, 1.4, -1.2, -1.0, -0.2, 0.0},
    {-0.5, -1.0, 1.35, -2.51, -0.88, 0.63, 0.0},
    {0.0, -1.5, 1.4, -1.2, -1.5, -0.2, 0.0},
    {-0.5, -1.5, 1.35, -2.51, -0.88, 0.63, 0.0},
};

Eigen::VectorXd vec(const std::vector<double>& values) {
    return Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
}

}  // namespace

BOOST_FIXTURE_TEST_SUITE(moveit_totg_time_parameterization, fixture)

BOOST_AUTO_TEST_CASE(construction_validates_its_own_arguments) {
    BOOST_CHECK_THROW(time_parameterization({}, -1.0), std::invalid_argument);
    BOOST_CHECK_THROW(time_parameterization({}, std::nan("")), std::invalid_argument);
    BOOST_CHECK_THROW(time_parameterization({}, 1e-3, {}, viam::trajex::types::hertz{0.0}), std::invalid_argument);
    BOOST_CHECK_NO_THROW(time_parameterization{});
}

BOOST_AUTO_TEST_CASE(empty_trajectory_succeeds_unchanged) {
    robot_trajectory::RobotTrajectory trajectory(model, group);
    BOOST_CHECK(time_parameterization{}.computeTimeStamps(trajectory));
    BOOST_CHECK(trajectory.empty());
}

BOOST_AUTO_TEST_CASE(trajectory_without_group_fails) {
    robot_trajectory::RobotTrajectory trajectory(model);
    mc::RobotState state(model);
    state.setToDefaultValues();
    trajectory.addSuffixWayPoint(state, 0.0);
    trajectory.addSuffixWayPoint(state, 0.1);
    BOOST_CHECK(!time_parameterization{}.computeTimeStamps(trajectory));
}

// A straight line has no blends, so the motion is pure accelerate, cruise or not, decelerate, and acceleration
// stays within its limits too.
BOOST_AUTO_TEST_CASE(straight_line) {
    auto trajectory = make({k_start, k_end});
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(trajectory));
    check_valid(trajectory, vec(k_start), vec(k_end));

    for (std::size_t w = 0; w < trajectory.getWayPointCount(); ++w) {
        for (const auto& name : group->getVariableNames()) {
            BOOST_CHECK_LE(std::abs(trajectory.getWayPoint(w).getVariableAcceleration(name)), 1.0 * (1.0 + k_limit_tolerance));
        }
    }
}

BOOST_AUTO_TEST_CASE(waypoints_are_uniformly_spaced_at_no_less_than_the_sampling_frequency) {
    auto trajectory = make({k_start, k_end});
    BOOST_REQUIRE(time_parameterization({}, 1e-3, {}, viam::trajex::types::hertz{100.0}).computeTimeStamps(trajectory));
    const double spacing = trajectory.getWayPointDurationFromPrevious(1);
    BOOST_CHECK_LE(spacing, 0.01 + 1e-12);
    for (std::size_t w = 1; w < trajectory.getWayPointCount(); ++w) {
        BOOST_CHECK_CLOSE(trajectory.getWayPointDurationFromPrevious(w), spacing, 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(cornered_path) {
    auto trajectory = make(k_cornered);
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(trajectory));
    check_valid(trajectory, vec(k_cornered.front()), vec(k_cornered.back()));
}

// With unit acceleration limits this path never gets near the Panda's velocity limits, so velocity scaling only
// binds once it is severe. Scale each limit enough that it must be the binding one.
BOOST_AUTO_TEST_CASE(scaling_slows_the_trajectory) {
    auto unscaled = make(k_cornered);
    auto slow_velocity = make(k_cornered);
    auto slow_acceleration = make(k_cornered);
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(unscaled, 1.0, 1.0));
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(slow_velocity, 0.1, 1.0));
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(slow_acceleration, 1.0, 0.5));
    BOOST_CHECK_GT(slow_velocity.getDuration(), unscaled.getDuration());
    BOOST_CHECK_GT(slow_acceleration.getDuration(), unscaled.getDuration());
    check_valid(slow_velocity, vec(k_cornered.front()), vec(k_cornered.back()), 0.1);
}

BOOST_AUTO_TEST_CASE(invalid_scaling_falls_back_to_one) {
    auto unscaled = make(k_cornered);
    auto invalid = make(k_cornered);
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(unscaled, 1.0, 1.0));
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(invalid, 1.5, 0.0));
    BOOST_CHECK_EQUAL(invalid.getDuration(), unscaled.getDuration());
}

BOOST_AUTO_TEST_CASE(given_limits_override_the_model) {
    const std::string joint = "panda_joint1";
    const double limit = model->getVariableBounds(joint).max_velocity_ / 4.0;

    auto by_map = make(k_cornered);
    const std::unordered_map<std::string, double> velocity_limits{{joint, limit}};
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(by_map, velocity_limits, {}));
    for (std::size_t w = 0; w < by_map.getWayPointCount(); ++w) {
        BOOST_CHECK_LE(std::abs(by_map.getWayPoint(w).getVariableVelocity(joint)), limit * (1.0 + k_limit_tolerance));
    }

    moveit_msgs::msg::JointLimits message;
    message.joint_name = joint;
    message.has_velocity_limits = true;
    message.max_velocity = limit;
    auto by_message = make(k_cornered);
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(by_message, std::vector{message}));
    BOOST_CHECK_EQUAL(by_message.getDuration(), by_map.getDuration());
}

BOOST_AUTO_TEST_CASE(missing_limits_fail_and_leave_the_trajectory_alone) {
    model = panda(false);
    group = model->getJointModelGroup("panda_arm");
    auto trajectory = make(k_cornered);
    BOOST_CHECK(!time_parameterization{}.computeTimeStamps(trajectory));
    BOOST_CHECK_EQUAL(trajectory.getWayPointCount(), k_cornered.size());
}

BOOST_AUTO_TEST_CASE(no_motion_is_a_single_waypoint_at_rest) {
    auto nudged = k_start;
    nudged[0] += 1e-4;
    auto trajectory = make({k_start, nudged, k_start});
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(trajectory));
    BOOST_REQUIRE_EQUAL(trajectory.getWayPointCount(), 1U);
    BOOST_CHECK_LE((positions(trajectory, 0) - vec(k_start)).cwiseAbs().maxCoeff(), 1e-12);
    for (const auto& name : group->getVariableNames()) {
        BOOST_CHECK_EQUAL(trajectory.getFirstWayPoint().getVariableVelocity(name), 0.0);
    }
}

// The last waypoint is within the deduplication tolerance of the one before it. It is the goal, so it is the one
// kept.
BOOST_AUTO_TEST_CASE(deduplication_keeps_the_final_waypoint) {
    auto near_end = k_end;
    near_end[0] -= 5e-4;
    auto trajectory = make({k_start, near_end, k_end});
    BOOST_REQUIRE(time_parameterization{}.computeTimeStamps(trajectory));
    check_valid(trajectory, vec(k_start), vec(k_end));
}

// The prototype's limits are replaced on every call, whatever they were.
BOOST_AUTO_TEST_CASE(prototype_limits_are_replaced) {
    viam::trajex::totg::trajectory::options prototype;
    prototype.max_velocity = viam::trajex::xvector<>{1.0};
    prototype.max_acceleration = viam::trajex::xvector<>{1.0};
    auto trajectory = make(k_cornered);
    BOOST_REQUIRE(time_parameterization({}, 1e-3, prototype).computeTimeStamps(trajectory));
    check_valid(trajectory, vec(k_cornered.front()), vec(k_cornered.back()));
}

// Options trajex rejects surface as a failed call, not an exception.
BOOST_AUTO_TEST_CASE(trajex_failures_are_reported_not_thrown) {
    auto trajectory = make(k_cornered);
    const time_parameterization parameterization(viam::trajex::totg::path::options{}.set_max_blend_deviation(-1.0));
    bool succeeded = true;
    BOOST_CHECK_NO_THROW(succeeded = parameterization.computeTimeStamps(trajectory));
    BOOST_CHECK(!succeeded);
}

BOOST_AUTO_TEST_SUITE_END()
