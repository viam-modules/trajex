#include <cstddef>
#include <vector>

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
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
using viam::trajex::arc_length;
using viam::trajex::arc_velocity;
using viam::trajex::degrees_to_radians;
using viam::trajex::xmatrix;
using viam::trajex::xvector;

// Same velocity/acceleration values as the UR arm profiles, sized to the given DOF.
// "all_" prefix indicates limits are uniform across all joints.
std::vector<constraint_profile> get_extremal_constraint_profiles(size_t dof) {
    return {
        {"all_accel_constrained",
         degrees_to_radians(xt::ones<double>({dof}) * 150.0),  // 150 deg/s (non-binding)
         degrees_to_radians(xt::ones<double>({dof}) * 5.0)},   // 5 deg/s^2 (binding)
        {"all_vel_constrained",
         degrees_to_radians(xt::ones<double>({dof}) * 5.0),     // 5 deg/s (binding)
         degrees_to_radians(xt::ones<double>({dof}) * 150.0)},  // 150 deg/s^2 (non-binding)
    };
}

}  // namespace

BOOST_AUTO_TEST_SUITE(end_to_end_tests)

// Extremal path tests: exercise TOTG directly (no segmentation layer) on degenerate geometries.
//
// Segmentation is a workaround for the legacy generator; the TOTG algorithm should handle these
// cases natively. At a cusp (direct reversal), s_dot must reach zero but s_ddot need not — the
// arm should decelerate through the reversal and accelerate out, with continuous acceleration.
//
// These tests are purely exploratory: no event expectations, arc-length kinematic validation
// runs (and passes). Joint-space kinematic validation is opt-in per test.

BOOST_AUTO_TEST_SUITE(extremal_path_tests)

#define EXTREMAL_DATA(dof)                                     \
    boost::unit_test::data::make([]() {                        \
        static auto d = get_extremal_constraint_profiles(dof); \
        return boost::unit_test::data::make(d);                \
    }())

// 1 DoF: direct reversal. Path goes A -> B -> A, with a sharp cusp at B.
// s_dot must reach zero at the cusp; s_ddot should remain non-zero (decelerating in, accelerating out).
BOOST_DATA_TEST_CASE(one_dof_direct_reversal, EXTREMAL_DATA(1), profile) {
    // The shape is given explicitly because at 1 DoF a brace literal cannot distinguish a
    // three-row, one-column matrix from a three-element vector.
    auto wp = xmatrix<>::from_shape({3, 1});
    wp(0, 0) = 0.0;
    wp(1, 0) = 1.0;
    wp(2, 0) = 0.0;

    trajectory_test_fixture fixture(1);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.0)
        .set_waypoints_rad(wp);
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LL");
}

// 1 DoF: exact colinear. Middle point lies exactly on the line between start and end.
BOOST_DATA_TEST_CASE(one_dof_colinear, EXTREMAL_DATA(1), profile) {
    auto wp = xmatrix<>::from_shape({3, 1});
    wp(0, 0) = 0.0;
    wp(1, 0) = 0.5;
    wp(2, 0) = 1.0;

    trajectory_test_fixture fixture(1);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad(wp);
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LL");
}

// 2 DoF: direct reversal, only one joint moves. (0,0) -> (1,0) -> (0,0).
// Joint 1 is stationary throughout; tangent has a zero component on joint 1.
BOOST_DATA_TEST_CASE(two_dof_reversal_single_joint, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad({{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}});
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LL");
}

// 2 DoF: direct reversal, both joints move equally. (0,0) -> (1,1) -> (0,0).
BOOST_DATA_TEST_CASE(two_dof_reversal_both_joints, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad({{0.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}});
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LL");
}

// 2 DoF: exact colinear, one joint stationary. (0,0) -> (0.5,0) -> (1,0).
BOOST_DATA_TEST_CASE(two_dof_colinear_single_joint, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad({{0.0, 0.0}, {0.5, 0.0}, {1.0, 0.0}});
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LL");
}

// 2 DoF: exact colinear, both joints move equally. (0,0) -> (1,1) -> (2,2).
BOOST_DATA_TEST_CASE(two_dof_colinear_both_joints, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad({{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}});
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LL");
}

// 2 DoF: axis-aligned right angle, zero deviation. (0,0) -> (0,1) -> (1,1).
// Segment 1 direction (0,1), segment 2 direction (1,0): exactly 90 degrees, no blending.
BOOST_DATA_TEST_CASE(two_dof_right_angle_axis_aligned, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.0)
        .set_waypoints_rad({{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}});
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LL");
}

// 2 DoF: rotated right angle, zero deviation. (-1,0) -> (0,1) -> (1,0).
// Segment 1 direction (1,1)/sqrt(2), segment 2 direction (1,-1)/sqrt(2): dot product = 0.
// All coordinates are integers (exactly representable).
BOOST_DATA_TEST_CASE(two_dof_right_angle_rotated, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.0)
        .set_waypoints_rad({{-1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}});
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LL");
}

// 2 DoF: Z-staircase, adjacent blends consume the connecting segment entirely (RSDK-12711).
//
// Path: (0,0) -> (0,45) -> (45,45) -> (45,90) deg. Three equal 45-deg segments with two 90-deg
// corners. With max_deviation >> needed, each blend reaches maximum radius (half-segment = 22.5
// deg), exactly consuming the middle segment. The two blends are adjacent (C-C), producing LCCL.
// The C-C boundary is tangent-continuous by blend construction, so TOTG does not stop there.
BOOST_DATA_TEST_CASE(two_dof_zero_length_segment, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);

    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(100.0)  // Much larger than needed; blend radius capped by geometry
        .set_waypoints_deg({{0.0, 0.0}, {0.0, 45.0}, {45.0, 45.0}, {45.0, 90.0}});
    const trajectory traj = fixture.create_and_validate();

    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LCCL");
}

// 2 DoF: asymmetric Z-staircase, first blend loose and second tight (RSDK-12711).
//
// Three unit segments. Corner B: 90-degree turn, radius = 0.5. Corner C: 120-degree turn,
// radius = 0.5/sqrt(3) ≈ 0.289. Both trims = 0.5 (geometry-limited), consuming the middle
// segment entirely: LCCL with R1 > R2. The C-C curvature tightens going forward, so the Eq. 38
// boundary check fires and finds the switching point at the C-C boundary.
BOOST_DATA_TEST_CASE(two_dof_asymmetric_blend_loose_tight, EXTREMAL_DATA(2), profile) {
    // Waypoints (radians): (0,0) -> (0,1) -> (1,1) -> (0.5, 1+sqrt(3)/2)
    // Turn at (0,1): 90 deg. Turn at (1,1): 120 deg (incoming (1,0), outgoing (-0.5, sqrt(3)/2)).
    trajectory_test_fixture fixture(2);

    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(100.0)
        .set_waypoints_rad({{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {0.5, 1.0 + (std::numbers::sqrt3 / 2.0)}});
    const trajectory traj = fixture.create_and_validate();

    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LCCL");

    std::vector<double> radii;
    for (const auto& seg : traj.path()) {
        seg.visit([&radii](const auto& s) {
            if constexpr (std::is_same_v<std::decay_t<decltype(s)>, path::segment::circular>) {
                radii.push_back(s.radius);
            }
        });
    }
    BOOST_REQUIRE_EQUAL(radii.size(), 2U);
    BOOST_CHECK_GT(radii[0], radii[1]);
    BOOST_CHECK_CLOSE(radii[0], 0.5, 1e-9);
    BOOST_CHECK_CLOSE(radii[1], 0.5 / std::numbers::sqrt3, 1e-9);
}

// 2 DoF: asymmetric Z-staircase, first blend tight and second loose (RSDK-12711).
//
// Three unit segments. Corner B: 120-degree turn, radius = 0.5/sqrt(3) ≈ 0.289. Corner C:
// 90-degree turn, radius = 0.5. LCCL with R1 < R2. The C-C curvature relaxes going forward, so
// Eq. 38 does not fire. A joint reaches its zero-tangent extremum at the start of the second arc
// (C-C boundary, extremum_angle=0); Case 2 must find and validate it.
BOOST_DATA_TEST_CASE(two_dof_asymmetric_blend_tight_loose, EXTREMAL_DATA(2), profile) {
    // Waypoints (radians): (0,0) -> (1,0) -> (0.5, sqrt(3)/2) -> (0.5+sqrt(3)/2, 0.5+sqrt(3)/2)
    // Turn at (1,0): 120 deg (incoming (1,0), outgoing (-0.5, sqrt(3)/2)).
    // Turn at (0.5, sqrt(3)/2): 90 deg (incoming (-0.5, sqrt(3)/2), outgoing (sqrt(3)/2, 0.5)).
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(100.0)
        .set_waypoints_rad({{0.0, 0.0},
                            {1.0, 0.0},
                            {0.5, std::numbers::sqrt3 / 2.0},
                            {0.5 + (std::numbers::sqrt3 / 2.0), 0.5 + (std::numbers::sqrt3 / 2.0)}});
    const trajectory traj = fixture.create_and_validate();

    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LCCL");

    std::vector<double> radii;
    for (const auto& seg : traj.path()) {
        seg.visit([&radii](const auto& s) {
            if constexpr (std::is_same_v<std::decay_t<decltype(s)>, path::segment::circular>) {
                radii.push_back(s.radius);
            }
        });
    }
    BOOST_REQUIRE_EQUAL(radii.size(), 2U);
    BOOST_CHECK_LT(radii[0], radii[1]);
    BOOST_CHECK_CLOSE(radii[0], 0.5 / std::numbers::sqrt3, 1e-9);
    BOOST_CHECK_CLOSE(radii[1], 0.5, 1e-9);
}

// 1 DoF: two successive reversals. Path goes A -> B -> A -> B, with cusps at B and then A.
// Tests that the algorithm handles multiple chained cusps and still ends with s_dot=0.
BOOST_DATA_TEST_CASE(one_dof_double_reversal, EXTREMAL_DATA(1), profile) {
    auto wp = xmatrix<>::from_shape({4, 1});
    wp(0, 0) = 0.0;
    wp(1, 0) = 1.0;
    wp(2, 0) = 0.0;
    wp(3, 0) = 1.0;

    trajectory_test_fixture fixture(1);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad(wp);
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LLL");
}

// 2 DoF: two successive reversals, only one joint moves. (0,0) -> (1,0) -> (0,0) -> (1,0).
BOOST_DATA_TEST_CASE(two_dof_double_reversal, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad({{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}});
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LLL");
}

// 2 DoF: two successive reversals, both joints move. (0,0) -> (1,1) -> (0,0) -> (1,1).
BOOST_DATA_TEST_CASE(two_dof_double_reversal_two_joints_in_the_morning, EXTREMAL_DATA(2), profile) {
    trajectory_test_fixture fixture(2);
    fixture.allow_any_events()
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad({{0.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}});
    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LLL");
}

#undef EXTREMAL_DATA

// A tricky test which engineers a situation where the first cusp at a blend is not a local minimum and we need to keep
// looking. Before the fix for RSDK-13450, this test would fail. The particular waypoints and accelerations here were
// found empirically by iterating until we engineered a graph shape that met the needs.
BOOST_AUTO_TEST_CASE(RSDK_13450_nonfirst_extremum_is_switching_point) {
    trajectory_test_fixture fixture(5);

    // TODO(RSDK-12981): Use a very relaxed tolerance in this test since empirically we require it;
    // the expectation is that fixing RSDK-12981 will remove the need for this.
    fixture.validation_tolerance_percent *= 13;

    const xvector<> max_velocity = {100.0, 100.0, 100.0, 100.0, 100.0};
    const xvector<> max_acceleration = {6, 0.75, 2, 0.5, 4};
    fixture.set_max_velocity(max_velocity)
        .set_max_acceleration(max_acceleration)
        .set_max_blend_deviation(0.1)
        .set_waypoints_rad({{0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 1.0, 1.0}, {0.0, 0.5, 0.6, 0.7, 0.8}});

    // The key assertion for RSDK-13450: after hitting the limit at the L-C junction, the first
    // nondiff extremum candidate is not a local minimum, so the algorithm keeps looking and finds
    // the second candidate (k_nondifferentiable_extremum) as the actual switching point.
    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_hit_limit(arc_length{2.10559}, arc_velocity{2.16985}, arc_velocity{0.36975}, arc_velocity{223.607})
        .expect_backward_start(arc_length{2.10559}, arc_velocity{0.36975}, trajectory::switching_point_kind::k_discontinuous_curvature)
        .expect_splice(std::nullopt, size_t{549})
        .expect_forward_start(arc_length{2.10559}, arc_velocity{0.36975})
        .expect_hit_limit(arc_length{2.12877}, arc_velocity{0.37161}, arc_velocity{0.37161}, arc_velocity{159.008})
        .expect_backward_start(arc_length{2.18090}, arc_velocity{0.18160}, trajectory::switching_point_kind::k_nondifferentiable_extremum)
        .expect_splice(std::nullopt, size_t{1132})
        .expect_forward_start(arc_length{2.18090}, arc_velocity{0.18160})
        .expect_backward_start(arc_length{3.30795}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(std::nullopt, size_t{322});

    const trajectory traj = fixture.create_and_validate();
    BOOST_CHECK_EQUAL(path_type_sequence(traj.path()), "LCL");
}

BOOST_AUTO_TEST_SUITE_END()  // extremal_path_tests

BOOST_AUTO_TEST_SUITE_END()  // end_to_end_tests
