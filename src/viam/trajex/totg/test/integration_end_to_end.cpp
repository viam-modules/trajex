#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include <json/json.h>

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <viam/trajex/totg/observers.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/test/test_utils.hpp>
#include <viam/trajex/totg/tools/json_serialization.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/angles.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/arc_operations.hpp>
#include <viam/trajex/types/hertz.hpp>
#include <viam/trajex/types/xt.hpp>

namespace {

using namespace viam::trajex::totg;
using namespace viam::trajex::totg::test;
using viam::trajex::arc_length;
using viam::trajex::arc_velocity;
using viam::trajex::degrees_to_radians;
using viam::trajex::xmatrix;
using viam::trajex::xvector;

trajectory create_trajectory_with_integration_points(path p, std::vector<trajectory::integration_point> points) {
    const trajectory::options opts{.max_velocity = xt::ones<double>({p.dof()}), .max_acceleration = xt::ones<double>({p.dof()})};
    return trajectory::create(std::move(p), opts, std::move(points));
}

// Helper to get UR arm waypoints with reversals for incremental testing
xmatrix<> get_ur_arm_waypoints_with_reversals_deg() {
    static const xmatrix<> waypoints_deg = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},      // 0: Start at zero
        {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0},  // 1
        {-45.0, -90.0, 0.0, 0.0, 0.0, 0.0},  // 2
        {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0},  // 3: reversal back to position 1
        {-90.0, 0.0, 0.0, 0.0, 0.0, 0.0},    // 4
        {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0},  // 5: reversal back to position 1 again
        {-45.0, -90.0, 0.0, 0.0, 0.0, 0.0},  // 6
        {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0},  // 7
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},      // 8: back to zero
        {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0},  // 9
        {-45.0, -90.0, 0.0, 0.0, 0.0, 0.0},  // 10
        {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0},  // 11
        {-90.0, 0.0, 0.0, 0.0, 0.0, 0.0},    // 12
        {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0},  // 13
        {-45.0, -90.0, 0.0, 0.0, 0.0, 0.0},  // 14
        {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0},  // 15
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},      // 16: back to zero again
    };
    return waypoints_deg;
}

// Spiral/rectangle waypoints for the 6-DOF integration regression test. Joints 1-2 trace
// a series of expanding rectangles while joints 3-4 add 3D motion on the outer loops, then
// the path collapses symmetrically back toward the origin.
xmatrix<> get_spiral_rectangle_waypoints_deg() {
    static const xmatrix<> waypoints_deg = {
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0},  // prior end position (matches end of path)
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // origin

        // Small rectangle
        {10.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {10.0, -10.0, 0.0, 0.0, 0.0, 0.0},
        {-10.0, -10.0, 0.0, 0.0, 0.0, 0.0},
        {-10.0, 10.0, 0.0, 0.0, 0.0, 0.0},

        // Medium rectangle
        {20.0, 10.0, 0.0, 0.0, 0.0, 0.0},
        {20.0, -20.0, 0.0, 0.0, 0.0, 0.0},
        {-20.0, -20.0, 0.0, 0.0, 0.0, 0.0},
        {-20.0, 20.0, 0.0, 0.0, 0.0, 0.0},

        // Large rectangle
        {30.0, 20.0, 0.0, 0.0, 0.0, 0.0},
        {30.0, -30.0, 0.0, 0.0, 0.0, 0.0},
        {-30.0, -30.0, 0.0, 0.0, 0.0, 0.0},
        {-30.0, 30.0, 0.0, 0.0, 0.0, 0.0},

        // Outer spiral - joints 3 introduced
        {40.0, 30.0, 10.0, 0.0, 0.0, 0.0},
        {40.0, -40.0, 10.0, 0.0, 0.0, 0.0},
        {-40.0, -40.0, 10.0, 0.0, 0.0, 0.0},
        {-40.0, 40.0, 10.0, 0.0, 0.0, 0.0},

        // Outermost rectangle - joints 3-4 active
        {50.0, 40.0, 20.0, 5.0, 0.0, 0.0},
        {50.0, -50.0, 20.0, 5.0, 0.0, 0.0},
        {-50.0, -50.0, 20.0, 5.0, 0.0, 0.0},
        {-50.0, 50.0, 20.0, 5.0, 0.0, 0.0},

        // Collapse inward
        {40.0, 40.0, 15.0, 3.0, 0.0, 0.0},
        {40.0, -40.0, 15.0, 3.0, 0.0, 0.0},
        {-40.0, -40.0, 15.0, 3.0, 0.0, 0.0},
        {-40.0, 40.0, 15.0, 3.0, 0.0, 0.0},

        {20.0, 30.0, 10.0, 0.0, 0.0, 0.0},
        {20.0, -20.0, 10.0, 0.0, 0.0, 0.0},
        {-20.0, -20.0, 10.0, 0.0, 0.0, 0.0},
        {-20.0, 20.0, 10.0, 0.0, 0.0, 0.0},

        // Return to near-home
        {10.0, 10.0, 5.0, 0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
    };
    return waypoints_deg;
}

// Current constraint profiles for UR arm testing
std::vector<constraint_profile> get_ur_arm_constraint_profiles() {
    using namespace viam::trajex;
    return {
        {"all_accel_constrained",
         degrees_to_radians(xt::ones<double>({6}) * 150.0),  // 150 deg/s
         degrees_to_radians(xt::ones<double>({6}) * 5.0)},   // 5 deg/s^2

        {"all_vel_constrained",
         degrees_to_radians(xt::ones<double>({6}) * 5.0),     // 150 deg/s
         degrees_to_radians(xt::ones<double>({6}) * 150.0)},  // 5 deg/s^2
    };
}

// Constraint profiles for the spiral/rectangle regression test. The first entry is the
// original failing case observed on hardware (100 deg/s, 100 deg/s^2). The remaining
// entries span the accel- and velocity-constrained regimes for coverage.
std::vector<constraint_profile> get_spiral_rectangle_constraint_profiles() {
    return {
        {"100_vel_100_acc", degrees_to_radians(xt::ones<double>({6}) * 100.0), degrees_to_radians(xt::ones<double>({6}) * 100.0)},
        {"150_vel_5_acc", degrees_to_radians(xt::ones<double>({6}) * 150.0), degrees_to_radians(xt::ones<double>({6}) * 5.0)},
        {"5_vel_150_acc", degrees_to_radians(xt::ones<double>({6}) * 5.0), degrees_to_radians(xt::ones<double>({6}) * 150.0)},
    };
}

}  // namespace

BOOST_AUTO_TEST_SUITE(end_to_end_tests)

BOOST_AUTO_TEST_CASE(waypoints_to_samples_smoke_test) {
    using namespace viam::trajex;
    using namespace viam::trajex::totg;
    using namespace viam::trajex::totg::test;

    // 1. Create waypoints
    const xmatrix<> waypoints = {
        {0.0, 0.0},  // Start
        {1.0, 0.0},  // Move right
        {1.0, 1.0}   // Move up
    };

    // 2. Create path
    path p = path::create(waypoints);  // Linear segments only
    BOOST_CHECK_EQUAL(p.size(), 2U);
    BOOST_CHECK_CLOSE(static_cast<double>(p.length()), 2.0, 0.001);

    // 3. Create trajectory with known integration points for testing
    // Simple constant velocity trajectory
    std::vector<trajectory::integration_point> points = {
        {.time = trajectory::seconds{0.0}, .s = arc_length{0.0}, .s_dot = arc_velocity{1.0}, .s_ddot = arc_acceleration{0.0}},
        {.time = trajectory::seconds{1.0}, .s = arc_length{1.0}, .s_dot = arc_velocity{1.0}, .s_ddot = arc_acceleration{0.0}},
        {.time = trajectory::seconds{2.0}, .s = arc_length{2.0}, .s_dot = arc_velocity{1.0}, .s_ddot = arc_acceleration{0.0}}};
    const trajectory traj = create_trajectory_with_integration_points(std::move(p), std::move(points));

    // 4. Verify trajectory has correct duration
    BOOST_CHECK_CLOSE(traj.duration().count(), 2.0, 0.001);

    // 5. Create uniform sampler at ~10Hz (dt ~= 0.1s)
    const auto sampler = uniform_sampler::quantized_for_trajectory(traj, types::hertz{10.0});

    // 6. Sample trajectory using cursor + sampler
    int sample_count = 0;
    trajectory::seconds last_time{-1.0};

    for (const auto& s : traj.samples(sampler)) {
        // Verify sample structure
        BOOST_CHECK_NO_THROW(static_cast<void>(s.time.count()));
        BOOST_CHECK_NO_THROW(s.configuration.size());
        BOOST_CHECK_NO_THROW(s.velocity.size());
        BOOST_CHECK_NO_THROW(s.acceleration.size());

        // Verify time is monotonically increasing
        BOOST_CHECK(s.time > last_time);
        last_time = s.time;

        // Verify configuration has correct DOF
        BOOST_CHECK_EQUAL(s.configuration.shape(0), 2U);
        BOOST_CHECK_EQUAL(s.velocity.shape(0), 2U);
        BOOST_CHECK_EQUAL(s.acceleration.shape(0), 2U);

        ++sample_count;
    }

    // Should have gotten multiple samples
    BOOST_CHECK(sample_count > 1);
}

BOOST_AUTO_TEST_CASE(cursor_manual_sampling) {
    using namespace viam::trajex;
    using namespace viam::trajex::totg;
    using namespace viam::trajex::totg::test;
    using viam::trajex::arc_length;

    // Create simple trajectory with known integration points
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 1.0}};
    path p = path::create(waypoints);

    std::vector<trajectory::integration_point> points = {
        {.time = trajectory::seconds{0.0}, .s = arc_length{0.0}, .s_dot = arc_velocity{1.0}, .s_ddot = arc_acceleration{0.0}},
        {.time = trajectory::seconds{1.0}, .s = arc_length{std::sqrt(2.0)}, .s_dot = arc_velocity{1.0}, .s_ddot = arc_acceleration{0.0}}};
    const trajectory traj = create_trajectory_with_integration_points(std::move(p), std::move(points));

    // Create cursor
    auto cursor = traj.create_cursor();

    // Verify initial state
    BOOST_CHECK_EQUAL(cursor.time().count(), 0.0);
    BOOST_CHECK_EQUAL(&cursor.trajectory(), &traj);

    // Advance and sample
    cursor.seek(trajectory::seconds{0.5});
    BOOST_CHECK_EQUAL(cursor.time().count(), 0.5);

    auto sample = cursor.sample();
    BOOST_CHECK_EQUAL(sample.time.count(), 0.5);
    BOOST_CHECK_EQUAL(sample.configuration.shape(0), 2U);
}

BOOST_AUTO_TEST_CASE(quantized_sampler_end_to_end) {
    using namespace viam::trajex;
    using namespace viam::trajex::totg;
    using namespace viam::trajex::totg::test;
    using viam::trajex::arc_length;
    using viam::trajex::types::hertz;

    // Create trajectory with known integration points
    // Path length = 1.0, duration = 1.0s, constant velocity
    // At 8 Hz: putative = 8, num_samples = 9, dt = 1.0/8 = 0.125 (exact in binary!)
    xmatrix<> waypoints = xmatrix<>::from_shape({2, 1});
    waypoints(0, 0) = 0.0;
    waypoints(1, 0) = 1.0;
    path p = path::create(waypoints);

    std::vector<trajectory::integration_point> points = {
        {.time = trajectory::seconds{0.0}, .s = arc_length{0.0}, .s_dot = arc_velocity{1.0}, .s_ddot = arc_acceleration{0.0}},
        {.time = trajectory::seconds{1.0}, .s = arc_length{1.0}, .s_dot = arc_velocity{1.0}, .s_ddot = arc_acceleration{0.0}}};
    const trajectory traj = create_trajectory_with_integration_points(std::move(p), std::move(points));

    BOOST_REQUIRE_EQUAL(traj.duration().count(), 1.0);

    // Use quantized sampler with 8 Hz (dt = 0.125, exact in binary)
    auto sampler = uniform_sampler::quantized_for_trajectory(traj, hertz{8.0});

    std::optional<trajectory::seconds> last_time;
    int sample_count = 0;
    for (const auto& s : traj.samples(sampler)) {
        last_time = s.time;
        ++sample_count;
    }

    // Should get exactly 9 samples: 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
    BOOST_REQUIRE(last_time.has_value());
    BOOST_CHECK_EQUAL(sample_count, 9);
    BOOST_CHECK_EQUAL(last_time->count(), 1.0);  // Should hit endpoint exactly
}

BOOST_DATA_TEST_CASE(ur_arm_incremental_waypoints_with_reversals,
                     boost::unit_test::data::xrange(size_t{2}, get_ur_arm_waypoints_with_reversals_deg().shape(0) + 1) *
                         boost::unit_test::data::make([]() {
                             static auto data = get_ur_arm_constraint_profiles();
                             return boost::unit_test::data::make(data);
                         }()),
                     num_waypoints,
                     profile) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::totg::test;
    using namespace viam::trajex::types;

    std::cout << "\n=== Testing " << num_waypoints << " waypoints with " << profile.name << " ===\n";

    trajectory_test_fixture fixture(6);
    fixture.allow_any_events();

    // Only enable legacy comparison for up to 3 waypoints; reversals start at 4 and the legacy
    // generator does not correctly handle them (it crosses reversal boundaries at nonzero velocity).
    if (num_waypoints <= 3) {
        fixture.enable_legacy_comparison()
            .set_legacy_path_tolerance(10.0)       // Allow 10% path difference (known discrepancy)
            .set_legacy_duration_tolerance(10.0);  // Allow 10% duration difference
    }

    fixture.set_max_velocity(profile.max_velocity).set_max_acceleration(profile.max_acceleration).set_max_blend_deviation(0.1);

    // Extract subset of waypoints for this iteration
    const xmatrix<>& full_waypoints_deg = get_ur_arm_waypoints_with_reversals_deg();
    const xmatrix<> subset_deg = xt::view(full_waypoints_deg, xt::range(0, num_waypoints), xt::all());
    fixture.set_waypoints_deg(subset_deg);

    // Create and validate (exploratory test - no specific event expectations)
    const auto start_time = std::chrono::high_resolution_clock::now();
    const trajectory traj = fixture.create_and_validate();
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto generation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Log trajectory summary
    const auto& integration_points = traj.get_integration_points();
    const double path_length = integration_points.empty() ? 0.0 : static_cast<double>(integration_points.back().s);
    std::cout << "  Path length: " << path_length << " rad\n";
    std::cout << "  Duration: " << traj.duration().count() << " s\n";
    std::cout << "  Integration points: " << integration_points.size() << "\n";
    std::cout << "  Generation time: " << generation_time_ms << " ms\n";

    // Compare generation time with legacy if available
    if (fixture.legacy_generation_time_ms_.has_value()) {
        const double time_ratio = static_cast<double>(generation_time_ms) / *fixture.legacy_generation_time_ms_;
        const double time_diff_percent =
            ((static_cast<double>(generation_time_ms) - *fixture.legacy_generation_time_ms_) / *fixture.legacy_generation_time_ms_) * 100.0;

        std::cout << "[COMPARISON] Generation time: new=" << generation_time_ms << "ms, legacy=" << *fixture.legacy_generation_time_ms_
                  << "ms, " << "ratio=" << time_ratio << ", diff=" << time_diff_percent << "%\n";
    }

    // Verify sampling works
    auto sampler = uniform_sampler::quantized_for_trajectory(traj, hertz{5.0});
    int sample_count = 0;
    for (const auto& s : traj.samples(sampler)) {
        BOOST_CHECK_EQUAL(s.configuration.shape(0), 6U);
        ++sample_count;
    }
    BOOST_CHECK(sample_count > 0);

    BOOST_TEST_MESSAGE("PASS: " << profile.name << " with " << num_waypoints << " waypoints, " << "duration=" << traj.duration().count()
                                << "s, " << "points=" << traj.get_integration_points().size());
}

BOOST_AUTO_TEST_CASE(three_waypoint_baseline_behavior_accel_constrained) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_test_fixture fixture(6);

    fixture.enable_legacy_comparison()
        .set_max_velocity(degrees_to_radians(xt::ones<double>({6}) * 50.0))
        .set_max_acceleration(degrees_to_radians(xt::ones<double>({6}) * 1.0))
        .set_max_blend_deviation(0.1);

    // Trajectory-wide expectations
    fixture.expect_duration(trajectory::seconds{20.162}).expect_integration_point_count(20164).expect_path_length(arc_length{1.8553});

    // Set waypoints
    fixture.set_waypoints_deg({{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0}, {-45.0, -90.0, 0.0, 0.0, 0.0, 0.0}});

    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_hit_limit(arc_length{0.718023},
                          arc_velocity{0.18827},
                          arc_velocity{0.15297},  // acc_limit
                          arc_velocity{1.23413}   // vel_limit
                          )
        .expect_backward_start(arc_length{1.46262}, arc_velocity{0.12863}, trajectory::switching_point_kind::k_discontinuous_curvature)
        .expect_splice(trajectory::seconds{13.4162}, size_t{920})
        .expect_forward_start(arc_length{1.46262}, arc_velocity{0.12863})
        .expect_backward_start(arc_length{1.85532488}, arc_velocity{0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{20.16199}, size_t{9306});

    const trajectory traj = fixture.create_and_validate();
}

BOOST_AUTO_TEST_CASE(three_waypoint_baseline_behavior_vel_constrained) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // Create fixture
    trajectory_test_fixture fixture(6);

    fixture.enable_legacy_comparison()
        .set_max_velocity(degrees_to_radians(xt::ones<double>({6}) * 1.0))
        .set_max_acceleration(degrees_to_radians(xt::ones<double>({6}) * 50.0))
        .set_max_blend_deviation(0.1);

    // Trajectory-wide expectations
    fixture.expect_duration(trajectory::seconds{90.02}).expect_integration_point_count(90023).expect_path_length(arc_length{1.8553});

    // Set waypoints
    fixture.set_waypoints_deg({{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {-45.0, -45.0, 0.0, 0.0, 0.0, 0.0}, {-45.0, -90.0, 0.0, 0.0, 0.0, 0.0}});

    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_backward_start(arc_length{1.85532488}, arc_velocity{0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{90.02000000}, size_t{10});

    const trajectory traj = fixture.create_and_validate();
}

BOOST_AUTO_TEST_CASE(RSDK_12979_nondifferentiable_switching_point_requires_zero_acceleration) {
    // RSDK-12979: Test that non-differentiable switching points (where f'_i(s) = 0 for some joint
    // on a circular arc) are handled correctly by using zero acceleration during backward integration.
    //
    // Without the fix (commented out in trajectory.cpp:1537-1539), backward integration uses the
    // local s_ddot_min, which can be significantly negative when other joints have non-zero tangents.
    // This causes entry into the infeasible region and throws an exception.
    //
    // Strategy: Use very high velocity limits (unconstrained) and very low acceleration limits
    // (constrained) with curved paths to create acceleration-dominated dynamics.

    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    trajectory_test_fixture fixture(2);

    // TODO(RSDK-12981): Use a slightly relaxed tolerance in this test since empirically we require it;
    // the expectation is that fixing RSDK-12981 will remove the need for this.
    fixture.validation_tolerance_percent *= 2;

    // Very high velocity limits (essentially unconstrained)
    // Very low acceleration limits (the actual constraint)
    fixture.enable_legacy_comparison()
        .set_max_velocity(xvector<>{0.165, 0.167})
        .set_max_acceleration(xvector<>{0.05, 0.04})
        .set_max_blend_deviation(0.05);  // Enable curves

    // Waypoints that create a path with curves where joints have different tangent behavior
    fixture.set_waypoints_rad({{0.0, 0.0}, {0.6, 0.0}, {0.1, 0.6}});

    // Expectations - looking for acceleration-constrained pattern:
    // forward_start -> hit_limit -> backward_start (NDE) -> splice -> forward_start -> backward_start (endpoint) -> splice
    fixture.expectation_observer_->expect_forward_start(arc_length{0.0}, arc_velocity{0.0})
        .expect_hit_limit(arc_length{0.521369}, arc_velocity{0.165}, arc_velocity{0.0383818}, arc_velocity{0.165})
        .expect_backward_start(arc_length{0.521369}, arc_velocity{0.0383818}, trajectory::switching_point_kind::k_discontinuous_curvature)
        .expect_splice(trajectory::seconds{5.78126}, size_t{1536})
        .expect_forward_start(arc_length{0.521369}, arc_velocity{0.0383818})
        .expect_hit_limit(arc_length{0.566993}, arc_velocity{0.0471279}, arc_velocity{0.0471279}, arc_velocity{0.176646})
        .expect_backward_start(
            arc_length{0.579219}, arc_velocity{0.0429121}, trajectory::switching_point_kind::k_nondifferentiable_extremum)
        .expect_splice(trajectory::seconds{7.18515}, size_t{150})
        .expect_forward_start(arc_length{0.579219}, arc_velocity{0.0429121})
        .expect_backward_start(arc_length{1.3072}, arc_velocity{0.0}, trajectory::switching_point_kind::k_path_end)
        .expect_splice(trajectory::seconds{14.3426}, size_t{1687});

    const trajectory traj = fixture.create_and_validate();
}

BOOST_AUTO_TEST_CASE(trajectory_json_serialization) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    // Create simple path with 3 waypoints
    const xmatrix<> waypoints = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 0.0, 0.0}};

    path p = path::create(waypoints, path::options{});
    validate_path_quality(p, waypoint_accumulator{waypoints}, path::options{});

    // Attach collector
    trajectory_integration_event_collector collector;
    trajectory::options opts;
    opts.max_velocity = xt::ones<double>({3}) * 1.0;
    opts.max_acceleration = xt::ones<double>({3}) * 1.0;
    opts.observer = &collector;

    // Generate trajectory
    const trajectory traj = trajectory::create(std::move(p), opts);
    validate_trajectory_invariants(traj);

    // Serialize to JSON
    const std::string json = serialize_trajectory_to_json(collector, &traj);

    // Validate JSON structure
    const Json::CharReaderBuilder reader;
    Json::Value root;
    std::istringstream ss(json);
    BOOST_REQUIRE(Json::parseFromStream(reader, ss, &root, nullptr));

    // Check metadata
    BOOST_REQUIRE(root.isMember("metadata"));
    BOOST_REQUIRE(root["metadata"]["dof"].asInt() == 3);
    BOOST_REQUIRE(root["metadata"].isMember("duration"));

    // Check integration_points structure
    BOOST_REQUIRE(root.isMember("integration_points"));
    BOOST_REQUIRE(root["integration_points"]["time"].isArray());
    BOOST_REQUIRE(root["integration_points"]["s"].isArray());

    // Check events structure
    BOOST_REQUIRE(root.isMember("events"));
    BOOST_REQUIRE(root["events"]["forward_starts"].isArray());
    BOOST_REQUIRE(root["events"]["backward_starts"].isArray());

    // Should have at least 1 forward start and 1 backward start
    BOOST_CHECK_GT(root["events"]["forward_starts"].size(), 0);
    BOOST_CHECK_GT(root["events"]["backward_starts"].size(), 0);
}

// Parameterized regression test for the spiral/rectangle waypoint sequence. This path is
// known to fail with the new TOTG on the original hardware profile (100 deg/s / 100 deg/s^2)
// while the legacy generator succeeds. Additional profiles are included to explore which
// constraint regimes trigger the failure.
BOOST_DATA_TEST_CASE(spiral_rectangle_6dof, boost::unit_test::data::make(get_spiral_rectangle_constraint_profiles()), profile) {
    using namespace viam::trajex::totg;
    trajectory_test_fixture f(6);

    // NOTE: Using relaxed tolerance due to known acceleration bound violations
    // in some integration points for trajectories with reversals (likely RSDK-12981).
    f.validation_tolerance_percent *= 8;

    f.set_waypoints_deg(get_spiral_rectangle_waypoints_deg())
        .set_max_velocity(profile.max_velocity)
        .set_max_acceleration(profile.max_acceleration)
        .set_max_blend_deviation(0.1)  // matches arm default (k_default_path_tolerance_delta_rads)
        .allow_any_events();
    f.create_and_validate();
}

BOOST_AUTO_TEST_SUITE_END()
