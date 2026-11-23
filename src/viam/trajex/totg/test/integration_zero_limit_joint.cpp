#include <stdexcept>

#include <boost/test/unit_test.hpp>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/test/test_utils.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/angles.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/xt.hpp>

namespace {

using namespace viam::trajex::totg;
using namespace viam::trajex::totg::test;
using viam::trajex::xmatrix;
using viam::trajex::xvector;

}  // namespace

// ---------------------------------------------------------------------------
// Zero-limit joint tests.
//
// Three scenarios: one joint with zero velocity limit, one joint with zero
// acceleration limit, one joint with both zero. Each scenario is exercised
// with two paths: one that does not move the constrained joint (must succeed),
// and one that does move it (must throw, not hang or crash).
//
// 3-DOF, L-C-L path (three waypoints, one 90-degree corner with blend).
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE(zero_limit_joint)

namespace {

constexpr double k_blend_deviation = 0.1;

// Waypoints where joint 2 is stationary: {0,0,0} -> {1,0,0} -> {1,1,0}.
// Forms an L-C-L path with a 90-degree corner at the middle waypoint.
xmatrix<> waypoints_joint2_stationary() {
    return {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
}

// Waypoints where joint 2 also moves: {0,0,0} -> {1,0,0.5} -> {1,1,1}.
xmatrix<> waypoints_joint2_moving() {
    return {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.5}, {1.0, 1.0, 1.0}};
}

trajectory run_zero_limit_success(const xmatrix<>& waypoints, const xvector<>& max_vel, const xvector<>& max_acc) {
    trajectory_test_fixture fix{3};
    // TODO(RSDK-12981): Tolerances, etc.
    fix.validation_tolerance_percent *= 9;
    fix.set_waypoints_rad(waypoints)
        .set_max_velocity(max_vel)
        .set_max_acceleration(max_acc)
        .set_max_blend_deviation(k_blend_deviation)
        .allow_any_events();
    return fix.create_and_validate();
}

void run_zero_limit_failure(const xmatrix<>& waypoints, const xvector<>& max_vel, const xvector<>& max_acc) {
    path::options popt;
    popt.set_max_blend_deviation(k_blend_deviation);

    trajectory::options topt;
    topt.max_velocity = max_vel;
    topt.max_acceleration = max_acc;

    BOOST_CHECK_THROW((void)trajectory::create(path::create(waypoints, popt), topt), std::runtime_error);
}

}  // namespace

// --- Zero velocity limit ---

BOOST_AUTO_TEST_CASE(zero_vel_limit_stationary_joint_succeeds) {
    // Joint 2 has zero velocity limit but the path never moves it.
    run_zero_limit_success(waypoints_joint2_stationary(), xvector<>{1.0, 1.0, 0.0}, xvector<>{10.0, 10.0, 10.0});
}

BOOST_AUTO_TEST_CASE(zero_vel_limit_moving_joint_throws) {
    // Joint 2 has zero velocity limit but the path requires it to move.
    run_zero_limit_failure(waypoints_joint2_moving(), xvector<>{1.0, 1.0, 0.0}, xvector<>{10.0, 10.0, 10.0});
}

// --- Zero acceleration limit ---

BOOST_AUTO_TEST_CASE(zero_accel_limit_stationary_joint_succeeds) {
    // Joint 2 has zero acceleration limit but the path never moves it.
    run_zero_limit_success(waypoints_joint2_stationary(), xvector<>{10.0, 10.0, 10.0}, xvector<>{10.0, 10.0, 0.0});
}

BOOST_AUTO_TEST_CASE(zero_accel_limit_moving_joint_throws) {
    // Joint 2 has zero acceleration limit but the path requires it to move.
    run_zero_limit_failure(waypoints_joint2_moving(), xvector<>{10.0, 10.0, 10.0}, xvector<>{10.0, 10.0, 0.0});
}

// --- Zero velocity and acceleration limit ---

BOOST_AUTO_TEST_CASE(zero_vel_and_accel_limit_stationary_joint_succeeds) {
    // Joint 2 has zero velocity and acceleration limits but the path never moves it.
    run_zero_limit_success(waypoints_joint2_stationary(), xvector<>{1.0, 1.0, 0.0}, xvector<>{10.0, 10.0, 0.0});
}

BOOST_AUTO_TEST_CASE(zero_vel_and_accel_limit_moving_joint_throws) {
    // Joint 2 has zero velocity and acceleration limits but the path requires it to move.
    run_zero_limit_failure(waypoints_joint2_moving(), xvector<>{1.0, 1.0, 0.0}, xvector<>{10.0, 10.0, 0.0});
}

BOOST_AUTO_TEST_SUITE_END()  // zero_limit_joint
