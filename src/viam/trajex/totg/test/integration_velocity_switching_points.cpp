#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <viam/trajex/totg/observers.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/test/test_utils.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/angles.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/hertz.hpp>
#include <viam/trajex/types/xt.hpp>

namespace {

using namespace viam::trajex::totg;
using namespace viam::trajex::totg::test;
using viam::trajex::arc_length;
using viam::trajex::arc_velocity;
using viam::trajex::xmatrix;
using viam::trajex::xvector;

trajectory create_velocity_switching_test_trajectory(const xmatrix<>& waypoints_rad,
                                                     const xvector<>& max_velocity,
                                                     const xvector<>& max_acceleration,
                                                     double max_deviation,
                                                     trajectory::seconds delta,
                                                     trajectory_integration_event_collector& collector) {
    path::options popt;
    popt.set_max_blend_deviation(max_deviation);
    path p = path::create(waypoints_rad, popt);
    validate_path_quality(p, waypoint_accumulator{waypoints_rad}, popt);

    trajectory::options topt;
    topt.max_velocity = max_velocity;
    topt.max_acceleration = max_acceleration;
    topt.delta = delta;
    topt.observer = &collector;

    return trajectory::create(std::move(p), topt);
}

std::vector<trajectory::integration_observer::started_backward_event> get_backward_events_by_kind(
    const trajectory_integration_event_collector& collector, trajectory::switching_point_kind kind) {
    std::vector<trajectory::integration_observer::started_backward_event> events;
    for (const auto& ev : collector.events()) {
        if (const auto* backward = std::get_if<trajectory::integration_observer::started_backward_event>(&ev)) {
            if (backward->kind == kind) {
                events.push_back(*backward);
            }
        }
    }
    return events;
}

// Independent test oracle for Eq. 40 sign checks.
// We intentionally estimate d/ds(s_dot_max_vel) by finite differences via public APIs
// instead of calling internal production helpers directly. This keeps the test validating
// behavior (sign transitions around switching points) rather than reusing the same
// implementation path under test.
double estimate_eq40_delta(const trajectory& traj, arc_length s) {
    auto cursor = traj.path().create_cursor();

    const double path_length = static_cast<double>(traj.path().length());
    const double s_value = static_cast<double>(s);

    const double h = std::min(1e-3, std::max(1e-6, path_length * 1e-4));
    const double s_before = std::max(0.0, s_value - h);
    const double s_after = std::min(path_length, s_value + h);
    if (s_after <= s_before) {
        throw std::runtime_error{"estimate_eq40_delta: invalid finite-difference interval"};
    }

    cursor.seek(arc_length{s_before});
    const auto limits_before = traj.get_velocity_limits(cursor);

    cursor.seek(arc_length{s_after});
    const auto limits_after = traj.get_velocity_limits(cursor);

    const double curve_slope =
        (static_cast<double>(limits_after.s_dot_max_vel) - static_cast<double>(limits_before.s_dot_max_vel)) / (s_after - s_before);

    cursor.seek(s);
    const auto limits = traj.get_velocity_limits(cursor);
    if (static_cast<double>(limits.s_dot_max_vel) <= 0.0) {
        throw std::runtime_error{"estimate_eq40_delta: velocity limit is non-positive"};
    }

    const auto bounds = traj.get_acceleration_bounds(cursor, limits.s_dot_max_vel);
    const double trajectory_slope = static_cast<double>(bounds.s_ddot_min) / static_cast<double>(limits.s_dot_max_vel);
    return trajectory_slope - curve_slope;
}

}  // namespace

// =============================================================================
// Velocity switching point search tests
//
// These tests exercise the continuous and discontinuous velocity switching point
// detection (Kunz & Stilman equations 40-42). Cases 1-11 use high acceleration
// limits so the velocity curve is below the acceleration curve everywhere,
// making the trajectory velocity-dominated. Cases 12-15 use tight curvature
// (small deviation) where the centripetal term pushes the acceleration curve
// below the velocity curve in some regions, testing mixed regimes.
// =============================================================================

BOOST_AUTO_TEST_SUITE(velocity_switching_point_search)

// ---------------------------------------------------------------------------
// Case 1: Velocity curve drops sharply -> velocity escape switching point.
//
// Path turns from a direction where one joint has a high velocity limit to a
// direction where a slower joint becomes the constraint. The velocity curve
// drops, forcing the trajectory to leave the velocity curve.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(sharp_velocity_curve_drop_produces_velocity_escape) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // 2-DOF robot: joint 1 is fast, joint 2 is slow.
    // Path goes along joint 1 then turns 90 deg toward joint 2.
    // Velocity curve drops from ~v_max_1 to ~v_max_2 across the blend.
    // Tight blend so the trajectory hits the velocity curve before the blend,
    // then the curve drops sharply.
    trajectory_test_fixture fixture(2);

    fixture.set_max_velocity(xvector<>{0.5, 0.08}).set_max_acceleration(xvector<>{10.0, 10.0}).set_max_blend_deviation(0.03);

    fixture.set_waypoints_rad({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}});

    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_hit_limit(arc_length{0.939658}, arc_velocity{0.5038}, arc_velocity{0.911088}, arc_velocity{0.503797})
        .expect_backward_start(arc_length{0.942656}, arc_velocity{0.386946}, trajectory::switching_point_kind::k_velocity_escape)
        .expect_splice(trajectory::seconds{1.91158}, size_t{4})
        .expect_forward_start(arc_length{0.942656}, arc_velocity{0.386946})
        .expect_backward_start(arc_length{1.96891}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{14.3962}, size_t{5});

    const trajectory traj = fixture.create_and_validate();
}

// ---------------------------------------------------------------------------
// Case 2: Velocity curve rises steeply -> no velocity switching point.
//
// Path goes along the slow joint first, then turns toward the fast joint.
// The velocity curve rises, so the trajectory never needs to leave it due to
// velocity constraints. Only acceleration or path-end switching points occur.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(rising_velocity_curve_no_velocity_switching_point) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // Opposite of Case 1: path goes along joint 2 (slow) then turns to joint 1 (fast).
    // Same limits as Case 1 but opposite path direction -- velocity curve rises instead of drops.
    trajectory_test_fixture fixture(2);

    // TODO(RSDK-12981): Use a slightly relaxed tolerance in this test since empirically we require it;
    // the expectation is that fixing RSDK-12981 will remove the need for this.
    fixture.validation_tolerance_percent *= 3;

    fixture.set_max_velocity(xvector<>{0.5, 0.08}).set_max_acceleration(xvector<>{10.0, 10.0}).set_max_blend_deviation(0.03);

    fixture.set_waypoints_rad({{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}});

    // No velocity switching point expected -- velocity curve rises, not drops.
    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_backward_start(arc_length{1.96891}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{14.396}, size_t{26});

    const trajectory traj = fixture.create_and_validate();
}

// ---------------------------------------------------------------------------
// Case 3: Gradual velocity curve drop -> switching point found via bisection.
//
// Similar to Case 1 but with a wider blend (larger deviation), creating a more
// gradual velocity curve transition. Tests that bisection accurately locates
// the switching point when the sign change spans a wider arc.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(gradual_velocity_curve_drop_bisection_accuracy) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // Moderate blend deviation -> wider blend -> more gradual velocity curve transition.
    // Same velocity limits as Case 1 but with a wider blend, so the velocity curve
    // drops more gradually across the blend region.
    trajectory_test_fixture fixture(2);

    fixture.set_max_velocity(xvector<>{0.5, 0.08}).set_max_acceleration(xvector<>{10.0, 10.0}).set_max_blend_deviation(0.05);

    fixture.set_waypoints_rad({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}});

    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_hit_limit(arc_length{0.89844}, arc_velocity{0.506359}, arc_velocity{1.175865}, arc_velocity{0.506356})
        .expect_backward_start(arc_length{0.900447}, arc_velocity{0.458772}, trajectory::switching_point_kind::k_velocity_escape)
        .expect_splice(trajectory::seconds{1.82592}, size_t{2})
        .expect_forward_start(arc_length{0.900447}, arc_velocity{0.458772})
        .expect_backward_start(arc_length{1.94819}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{14.3069}, size_t{5});

    fixture.create_and_validate();

    trajectory_integration_event_collector collector;
    const trajectory observed_traj = create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}},
                                                                               xvector<>{0.5, 0.08},
                                                                               xvector<>{10.0, 10.0},
                                                                               0.05,
                                                                               trajectory::seconds{0.001},
                                                                               collector);

    const auto velocity_escapes = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_velocity_escape);
    BOOST_REQUIRE_EQUAL(velocity_escapes.size(), 1U);

    const auto s_switch = velocity_escapes.front().start.s;
    const double coarse_step = 0.001;
    const double coarse_index = static_cast<double>(s_switch) / coarse_step;
    const double grid_residual = std::abs(coarse_index - std::round(coarse_index));

    // Coarse scan samples at multiples of delta. Off-grid switching point implies bisection refinement.
    BOOST_CHECK_GT(grid_residual, 1e-6);

    const auto sample_offset = arc_length{1e-4};
    const auto s_before = std::max(arc_length{0.0}, s_switch - sample_offset);
    const auto s_after = std::min(observed_traj.path().length(), s_switch + sample_offset);

    const double delta_before = estimate_eq40_delta(observed_traj, s_before);
    const double delta_after = estimate_eq40_delta(observed_traj, s_after);

    BOOST_CHECK_GT(delta_before, 0.0);
    BOOST_CHECK_LE(delta_after, 0.0);
}

// ---------------------------------------------------------------------------
// Case 4: Multiple velocity switching points -> finds the first.
//
// Path with multiple turns, each causing the velocity curve to drop. The
// search should find the earliest switching point.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(multiple_velocity_drops_finds_first_switching_point) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // Zigzag path: joint 1 direction -> joint 2 direction -> joint 1 direction -> joint 2 direction.
    // Each turn from joint 1 to joint 2 creates a velocity drop.
    // Same limits as Case 1 which produces velocity_escape.
    trajectory_test_fixture fixture(2);

    // TODO(RSDK-12981): Use a slightly relaxed tolerance in this test since empirically we require it;
    // the expectation is that fixing RSDK-12981 will remove the need for this.
    fixture.validation_tolerance_percent *= 2;

    fixture.set_max_velocity(xvector<>{0.5, 0.08}).set_max_acceleration(xvector<>{10.0, 10.0}).set_max_blend_deviation(0.03);

    fixture.set_waypoints_rad({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});

    // Two velocity escape switching points (one at each joint-1-to-joint-2 turn).
    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_hit_limit(arc_length{0.939658}, arc_velocity{0.503821}, arc_velocity{0.911089}, arc_velocity{0.503789})
        .expect_backward_start(arc_length{0.942656}, arc_velocity{0.386946}, trajectory::switching_point_kind::k_velocity_escape)
        .expect_splice(trajectory::seconds{1.91158}, size_t{4})
        .expect_forward_start(arc_length{0.942656}, arc_velocity{0.386946})
        .expect_hit_limit(arc_length{2.87749}, arc_velocity{0.5038}, arc_velocity{0.911089}, arc_velocity{0.503798})
        .expect_backward_start(arc_length{2.88048}, arc_velocity{0.386946}, trajectory::switching_point_kind::k_velocity_escape)
        .expect_splice(trajectory::seconds{16.1457}, size_t{4})
        .expect_forward_start(arc_length{2.88048}, arc_velocity{0.386946})
        .expect_backward_start(arc_length{3.90674}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{28.6304}, size_t{5});

    const trajectory traj = fixture.create_and_validate();
}

// ---------------------------------------------------------------------------
// Case 5: Multiple velocity escapes on a multi-turn path with wider blends.
//
// Zigzag path with three 90 deg turns at moderate deviation. Each turn from the
// fast joint toward the slow joint creates a velocity drop. Tests that the
// search correctly finds velocity escapes across multiple blend regions with
// wider arcs than Cases 1/3.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(multiple_velocity_escapes_wider_blends) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_test_fixture fixture(2);

    // TODO(RSDK-12981): Use a slightly relaxed tolerance in this test since empirically we require it;
    // the expectation is that fixing RSDK-12981 will remove the need for this.
    fixture.validation_tolerance_percent *= 2;

    fixture.set_max_velocity(xvector<>{0.5, 0.08}).set_max_acceleration(xvector<>{10.0, 10.0}).set_max_blend_deviation(0.05);

    // Three turns with moderate blend radius.
    fixture.set_waypoints_rad({{0.0, 0.0}, {0.5, 0.0}, {0.5, 0.5}, {1.0, 0.5}, {1.0, 1.0}});

    // Two velocity escape switching points (one at each joint-1-to-joint-2 turn).
    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_hit_limit(arc_length{0.39844}, arc_velocity{0.50636}, arc_velocity{1.175865}, arc_velocity{0.506356})
        .expect_backward_start(arc_length{0.400447}, arc_velocity{0.458772}, trajectory::switching_point_kind::k_velocity_escape)
        .expect_splice(trajectory::seconds{0.825918}, size_t{2})
        .expect_forward_start(arc_length{0.400447}, arc_velocity{0.458772})
        .expect_hit_limit(arc_length{1.29579}, arc_velocity{0.506359}, arc_velocity{1.175865}, arc_velocity{0.506356})
        .expect_backward_start(arc_length{1.29683}, arc_velocity{0.458772}, trajectory::switching_point_kind::k_velocity_escape)
        .expect_splice(trajectory::seconds{7.63155}, size_t{2})
        .expect_forward_start(arc_length{1.29683}, arc_velocity{0.458772})
        .expect_backward_start(arc_length{1.84457}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{13.8625}, size_t{5});

    const trajectory traj = fixture.create_and_validate();
}

// ---------------------------------------------------------------------------
// Case 6: Constant velocity curve -> no velocity switching point.
//
// Straight-line path with uniform velocity limits. The velocity curve is flat,
// so there is no sign change in Delta(s). Only path-end switching point occurs.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(constant_velocity_curve_no_switching_point) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // Straight line -- no blends, constant q', constant velocity limit.
    trajectory_test_fixture fixture(2);

    fixture.set_max_velocity(xvector<>{0.5, 0.5}).set_max_acceleration(xvector<>{10.0, 10.0}).set_max_blend_deviation(0.1);

    // Diagonal straight line: both joints move equally, q' = (1/sqrt(2), 1/sqrt(2)).
    fixture.set_waypoints_rad({{0.0, 0.0}, {1.0, 1.0}});

    // No velocity switching point -- constant velocity curve on a straight line.
    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_backward_start(arc_length{1.41421}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{2.05002}, size_t{26});

    const trajectory traj = fixture.create_and_validate();
}

// ---------------------------------------------------------------------------
// Case 7: Velocity switching point early in the path.
//
// The trajectory hits the velocity curve quickly and the curve drops almost
// immediately, testing that the sign change detection works near the start
// of the coarse search.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(velocity_switching_point_near_path_start) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // Short first segment so the trajectory hits the velocity curve quickly,
    // then an immediate turn to a slower-constrained direction. High acceleration
    // ensures the trajectory reaches the velocity curve on the short first segment.
    trajectory_test_fixture fixture(2);

    fixture.set_max_velocity(xvector<>{0.5, 0.08}).set_max_acceleration(xvector<>{10.0, 10.0}).set_max_blend_deviation(0.03);

    // Moderate first segment (0.5 rad), then 90 deg turn.
    fixture.set_waypoints_rad({{0.0, 0.0}, {0.5, 0.0}, {0.5, 1.0}});

    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_hit_limit(arc_length{0.439123}, arc_velocity{0.50382}, arc_velocity{0.911089}, arc_velocity{0.503798})
        .expect_backward_start(arc_length{0.442656}, arc_velocity{0.386946}, trajectory::switching_point_kind::k_velocity_escape)
        .expect_splice(trajectory::seconds{0.911583}, size_t{4})
        .expect_forward_start(arc_length{0.442656}, arc_velocity{0.386946})
        .expect_backward_start(arc_length{1.46891}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{13.3962}, size_t{5});

    const trajectory traj = fixture.create_and_validate();
}

// ---------------------------------------------------------------------------
// Case 8: Eq. 40 sign bracket at a velocity escape switching point.
//
// For a velocity escape switching point, Delta(s) = s_ddot_min/s_dot_max_vel - d/ds(s_dot_max_vel)
// should transition from positive (trapped) to non-positive (escape possible).
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(eq40_sign_change_brackets_velocity_escape_switching_point) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_integration_event_collector collector;
    const trajectory traj = create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}},
                                                                      xvector<>{0.5, 0.08},
                                                                      xvector<>{10.0, 10.0},
                                                                      0.03,
                                                                      trajectory::seconds{0.001},
                                                                      collector);

    validate_trajectory_invariants(traj);

    const auto velocity_escapes = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_velocity_escape);
    BOOST_REQUIRE_EQUAL(velocity_escapes.size(), 1U);

    const auto s_switch = velocity_escapes.front().start.s;
    const auto sample_offset = arc_length{1e-4};
    const auto s_before = std::max(arc_length{0.0}, s_switch - sample_offset);
    const auto s_after = std::min(traj.path().length(), s_switch + sample_offset);

    const double delta_before = estimate_eq40_delta(traj, s_before);
    const double delta_after = estimate_eq40_delta(traj, s_after);

    BOOST_CHECK_GT(delta_before, 0.0);
    BOOST_CHECK_LE(delta_after, 0.0);
}

// ---------------------------------------------------------------------------
// Case 9: Rising velocity curve has no Eq. 40 escape transition.
//
// In this geometry the velocity curve rises, so there should be no Delta > 0 to Delta <= 0
// transition and no velocity-escape switching point.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(rising_velocity_curve_has_no_eq40_escape_transition) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_integration_event_collector collector;
    const trajectory traj = create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}},
                                                                      xvector<>{0.5, 0.08},
                                                                      xvector<>{10.0, 10.0},
                                                                      0.03,
                                                                      trajectory::seconds{0.001},
                                                                      collector);

    validate_trajectory_invariants(traj);

    const auto velocity_escapes = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_velocity_escape);
    BOOST_CHECK(velocity_escapes.empty());

    size_t eq40_crossings = 0;
    std::optional<double> previous_delta;
    auto cursor = traj.path().create_cursor();
    const auto path_length = static_cast<double>(traj.path().length());

    for (size_t i = 1; i < 1000; ++i) {
        const auto s = arc_length{path_length * (static_cast<double>(i) / 1000.0)};
        cursor.seek(s);

        const auto limits = traj.get_velocity_limits(cursor);
        if (static_cast<double>(limits.s_dot_max_vel) <= 0.0 ||
            static_cast<double>(limits.s_dot_max_acc) < static_cast<double>(limits.s_dot_max_vel)) {
            previous_delta.reset();
            continue;
        }

        const double delta = estimate_eq40_delta(traj, s);
        if (previous_delta.has_value() && (*previous_delta > 0.0) && (delta <= 0.0)) {
            ++eq40_crossings;
        }
        previous_delta = delta;
    }

    BOOST_CHECK_EQUAL(eq40_crossings, 0U);
}

// ---------------------------------------------------------------------------
// Case 10: Multiple velocity escapes are ordered and feasible.
//
// Verifies that the velocity-switching search finds multiple escape points in
// increasing arc-length order and each accepted switching point is feasible
// against the acceleration-limited velocity curve.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(multiple_velocity_escapes_are_ordered_and_feasible) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_integration_event_collector collector;
    const trajectory traj = create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}},
                                                                      xvector<>{0.5, 0.08},
                                                                      xvector<>{10.0, 10.0},
                                                                      0.03,
                                                                      trajectory::seconds{0.001},
                                                                      collector);

    validate_trajectory_invariants(traj);

    const auto velocity_escapes = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_velocity_escape);
    BOOST_REQUIRE_EQUAL(velocity_escapes.size(), 2U);
    BOOST_CHECK_LT(velocity_escapes[0].start.s, velocity_escapes[1].start.s);

    auto cursor = traj.path().create_cursor();
    for (const auto& escape : velocity_escapes) {
        cursor.seek(escape.start.s);
        const auto limits = traj.get_velocity_limits(cursor);

        BOOST_CHECK_LE(static_cast<double>(limits.s_dot_max_vel), static_cast<double>(limits.s_dot_max_acc) + 1e-9);
        BOOST_CHECK_CLOSE(static_cast<double>(escape.start.s_dot), static_cast<double>(limits.s_dot_max_vel), 0.5);
    }
}

// ---------------------------------------------------------------------------
// Case 11: Near-start velocity escape still exhibits Eq. 40 sign bracket.
//
// Same Eq. 40 sign-bracket assertion as Case 8, but for a switching point that
// occurs early in the path.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(near_start_velocity_escape_has_eq40_sign_bracket) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_integration_event_collector collector;
    const trajectory traj = create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {0.5, 0.0}, {0.5, 1.0}},
                                                                      xvector<>{0.5, 0.08},
                                                                      xvector<>{10.0, 10.0},
                                                                      0.03,
                                                                      trajectory::seconds{0.001},
                                                                      collector);

    validate_trajectory_invariants(traj);

    const auto velocity_escapes = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_velocity_escape);
    BOOST_REQUIRE_GE(velocity_escapes.size(), 1U);

    const auto s_switch = velocity_escapes.front().start.s;
    BOOST_CHECK_LT(s_switch, arc_length{1.0});

    const auto sample_offset = arc_length{1e-4};
    const auto s_before = std::max(arc_length{0.0}, s_switch - sample_offset);
    const auto s_after = std::min(traj.path().length(), s_switch + sample_offset);

    const double delta_before = estimate_eq40_delta(traj, s_before);
    const double delta_after = estimate_eq40_delta(traj, s_after);

    BOOST_CHECK_GT(delta_before, 0.0);
    BOOST_CHECK_LE(delta_after, 0.0);
}

// ---------------------------------------------------------------------------
// Case 12: Boundary produces k_discontinuous_velocity_limit.
//
// Extreme velocity ratio with a tiny blend so the velocity curve drops
// discontinuously at the segment boundary. High acceleration limits ensure
// feasibility (s_dot_max_acc >= s_dot_max_vel at the boundary).
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(boundary_produces_discontinuous_velocity_limit) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // 30 deg turn: the velocity curve has a steep slope at the blend-end boundary.
    // With moderate curvature (kappa ~= 50), centripetal acceleration makes trajectory_slope
    // less negative than curve_slope (condition 41 holds), while the gentler curvature
    // avoids backward-integration failure.
    trajectory_integration_event_collector collector;
    const trajectory traj =
        create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.866, 0.5}},
                                                  xvector<>{0.5, 0.1},         // 5:1 ratio
                                                  xvector<>{3.0, 3.0},         // accel tuned so c41 holds and trajectory constructs
                                                  0.0007,                      // kappa ~= 49 for 30 deg turn
                                                  trajectory::seconds{0.001},  // normal step
                                                  collector);

    validate_trajectory_invariants(traj);

    const auto disc = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_discontinuous_velocity_limit);
    BOOST_REQUIRE_GE(disc.size(), 1U);

    // Verify the switching point is at the blend-end boundary (around s ~= 1.0)
    BOOST_CHECK_CLOSE(static_cast<double>(disc.front().start.s), 1.0, 5.0);
}

// ---------------------------------------------------------------------------
// Case 13: Both discontinuous velocity limit and continuous velocity escape
// exist on the same trajectory.
//
// Extends the Case 12 geometry with a second turn that produces a velocity
// escape. Verifies Phase 4's select_switching_point correctly picks between
// both types based on arc-length ordering.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(both_discontinuous_velocity_limit_and_velocity_escape) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // First turn (30 deg) triggers discontinuous velocity limit (same as Case 12).
    // Second turn (90 deg, fast->slow) triggers continuous velocity escape.
    trajectory_integration_event_collector collector;
    const trajectory traj = create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.866, 0.5}, {1.866, 1.5}},
                                                                      xvector<>{0.5, 0.1},
                                                                      xvector<>{3.0, 3.0},
                                                                      0.0007,
                                                                      trajectory::seconds{0.001},
                                                                      collector);

    validate_trajectory_invariants(traj);

    const auto disc = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_discontinuous_velocity_limit);
    const auto vel = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_velocity_escape);

    // Both types must be present.
    BOOST_TEST_MESSAGE("Discontinuous: " << disc.size() << ", Velocity escapes: " << vel.size());
    BOOST_CHECK_GE(disc.size(), 1U);
    BOOST_CHECK_GE(vel.size(), 1U);
}

// ---------------------------------------------------------------------------
// Case 14: Multi-turn path with low acceleration -- stress test for switching
// point search (RSDK-12980 regression coverage).
//
// Uses two 90-degree turns with low acceleration, creating multiple regions where
// the switching point search must evaluate and potentially reject brackets
// before finding feasible velocity escapes.  The RSDK-12980 fix ensures that
// a rejected coarse bracket does not terminate the search prematurely.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(multi_turn_low_accel_switching_point_search) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_integration_event_collector collector;
    // Two 90-degree turns with low acceleration and asymmetric velocity limits.
    const trajectory traj = create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {0.5, 0.0}, {0.5, 0.5}, {1.0, 0.5}, {1.0, 1.0}},
                                                                      xvector<>{0.5, 0.08},  // asymmetric velocity
                                                                      xvector<>{2.0, 2.0},   // low-ish acceleration
                                                                      0.05,                  // moderate deviation
                                                                      trajectory::seconds{0.001},
                                                                      collector);

    validate_trajectory_invariants(traj);

    // The search must find velocity escapes at the turns.
    const auto vel_escapes = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_velocity_escape);
    BOOST_CHECK_GE(vel_escapes.size(), 1U);
    BOOST_CHECK_GT(traj.duration().count(), 0.0);
}

// ---------------------------------------------------------------------------
// Case 15: Condition 41 false gates a boundary that would otherwise qualify.
//
// Same 30 deg geometry as Case 12, but with a larger blend deviation (0.005
// vs 0.0007).  The larger deviation reduces curvature at the blend, which
// makes curve_slope_before much shallower.  trajectory_slope_before ends up
// MORE negative than curve_slope_before, so condition 41
// (trajectory_slope_before >= curve_slope_before) is false.  Condition 42
// remains true (linear after-segment -> curve_slope_after=0, and
// trajectory_slope_after < 0).
//
// This verifies that a single false condition gates the boundary decision,
// preventing a switching point that Case 12 (with tighter curvature) does
// produce.
//
// Note: condition 42 false is structurally unreachable at integration level
// with linear/circular segments because blend-end boundaries always have
// curve_slope_after = 0 (linear after-segment, q''=0) and
// trajectory_slope_after < 0 (minimum deceleration), making c42 trivially
// true.  Blend-start boundaries are filtered before reaching the
// c41/c42 evaluation because s_dot_max_accel < s_dot_max_vel there.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(condition_41_false_gates_boundary) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_integration_event_collector collector;
    // Same waypoints, velocities, accelerations as Case 12, but deviation=0.005
    // instead of 0.0007.  The larger blend radius reduces curvature enough that
    // condition 41 flips to false.
    const trajectory traj = create_velocity_switching_test_trajectory(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.866, 0.5}},
                                                                      xvector<>{0.5, 0.1},  // 5:1 ratio (same as Case 12)
                                                                      xvector<>{3.0, 3.0},  // same accel
                                                                      0.005,                // larger deviation -> lower curvature
                                                                      trajectory::seconds{0.001},
                                                                      collector);

    validate_trajectory_invariants(traj);

    // With condition 41 false, no discontinuous velocity switching point should be produced.
    const auto disc = get_backward_events_by_kind(collector, trajectory::switching_point_kind::k_discontinuous_velocity_limit);
    BOOST_CHECK_EQUAL(disc.size(), 0U);
}

BOOST_AUTO_TEST_SUITE_END()  // velocity_switching_point_search
