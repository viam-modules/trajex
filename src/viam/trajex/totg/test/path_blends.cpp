// Path circular blend tests
// Extracted from test.cpp lines 152-189 (helpers) and 1106-1560 (tests)

#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/test/test_utils.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/angles.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/xt.hpp>

#if __has_include(<xtensor/reducers/xnorm.hpp>)
#include <xtensor/reducers/xnorm.hpp>
#else
#include <xtensor/xnorm.hpp>
#endif

#include <boost/test/unit_test.hpp>

using viam::trajex::degrees_to_radians;
using viam::trajex::xmatrix;
using viam::trajex::xvector;
using viam::trajex::totg::test::configs_close;
using viam::trajex::totg::test::curvature_refit_bound;
using viam::trajex::totg::test::path_type_sequence;
using viam::trajex::totg::test::validate_path_quality;
using viam::trajex::totg::test::verify_path_visits_waypoints;

BOOST_AUTO_TEST_SUITE(path_tests)
// Circular blend tests
// These tests verify circular arc blending at corners that cannot be coalesced

BOOST_AUTO_TEST_CASE(circular_blend_right_angle_basic) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Simple 90-degree corner with blend
    const xmatrix<> waypoints = {{0.0, 0.0},
                                 {1.0, 0.0},  // Corner at 90 degrees
                                 {1.0, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Should have 3 segments: trimmed linear, circular blend, trimmed linear
    BOOST_CHECK_EQUAL(p.size(), 3U);

    // Verify path starts and ends at exact waypoints
    auto config_start = p.configuration(arc_length{0.0});
    auto config_end = p.configuration(p.length());
    BOOST_CHECK(configs_close(config_start, xvector<>{0.0, 0.0}));
    BOOST_CHECK(configs_close(config_end, xvector<>{1.0, 1.0}));

    // Verify corner waypoint is visited within blend deviation
    verify_path_visits_waypoints(p, waypoints, 0.1);

    // Check that middle segment is circular
    auto it = p.begin();
    BOOST_CHECK((*it).is<path::segment::linear>());  // First segment
    ++it;
    BOOST_CHECK((*it).is<path::segment::circular>());  // Blend segment
    ++it;
    BOOST_CHECK((*it).is<path::segment::linear>());  // Last segment
}

BOOST_AUTO_TEST_CASE(circular_blend_obtuse_angle) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // 135-degree turn (obtuse angle)
    const xmatrix<> waypoints = {{0.0, 0.0},
                                 {1.0, 0.0},  // Corner at 135 degrees
                                 {0.0, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Should create blend at obtuse corner
    BOOST_CHECK_EQUAL(p.size(), 3U);

    // Verify endpoints
    auto config_start = p.configuration(arc_length{0.0});
    auto config_end = p.configuration(p.length());
    BOOST_CHECK(configs_close(config_start, xvector<>{0.0, 0.0}));
    BOOST_CHECK(configs_close(config_end, xvector<>{0.0, 1.0}));

    // Verify corner is visited within tolerance
    verify_path_visits_waypoints(p, waypoints, 0.1);
}

BOOST_AUTO_TEST_CASE(circular_blend_acute_angle) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // 45-degree turn (acute angle)
    const xmatrix<> waypoints = {{0.0, 0.0},
                                 {1.0, 0.0},  // Corner at 45 degrees
                                 {2.0, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Should create blend at acute corner
    BOOST_CHECK_EQUAL(p.size(), 3U);

    // Verify endpoints
    auto config_start = p.configuration(arc_length{0.0});
    auto config_end = p.configuration(p.length());
    BOOST_CHECK(configs_close(config_start, xvector<>{0.0, 0.0}));
    BOOST_CHECK(configs_close(config_end, xvector<>{2.0, 1.0}));

    verify_path_visits_waypoints(p, waypoints, 0.1);
}

BOOST_AUTO_TEST_CASE(circular_blend_multiple_corners) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Path with two 90-degree corners
    const xmatrix<> waypoints = {{0.0, 0.0},
                                 {1.0, 0.0},  // First corner
                                 {1.0, 1.0},  // Second corner
                                 {0.0, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Should have: linear, blend, linear, blend, linear = 5 segments
    BOOST_CHECK_EQUAL(p.size(), 5U);

    // Verify endpoints
    auto config_start = p.configuration(arc_length{0.0});
    auto config_end = p.configuration(p.length());
    BOOST_CHECK(configs_close(config_start, xvector<>{0.0, 0.0}));
    BOOST_CHECK(configs_close(config_end, xvector<>{0.0, 1.0}));

    // All corners should be visited within tolerance
    verify_path_visits_waypoints(p, waypoints, 0.1);

    // Verify blend segments are circular
    auto it = p.begin();
    BOOST_CHECK((*it).is<path::segment::linear>());  // Start segment
    ++it;
    BOOST_CHECK((*it).is<path::segment::circular>());  // First blend
    ++it;
    BOOST_CHECK((*it).is<path::segment::linear>());  // Middle segment
    ++it;
    BOOST_CHECK((*it).is<path::segment::circular>());  // Second blend
    ++it;
    BOOST_CHECK((*it).is<path::segment::linear>());  // End segment
}

BOOST_AUTO_TEST_CASE(circular_blend_zero_deviation_no_blends) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // With zero blend deviation, should get sharp corners (no blends)
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.0));

    // Should have 2 linear segments, no blends
    BOOST_CHECK_EQUAL(p.size(), 2U);

    // Both segments should be linear
    for (auto seg : p) {
        BOOST_CHECK(seg.is<path::segment::linear>());
    }

    // Path goes exactly through corner
    auto corner_config = p.configuration(arc_length{1.0});
    BOOST_CHECK(configs_close(corner_config, xvector<>{1.0, 0.0}));
}

BOOST_AUTO_TEST_CASE(circular_blend_preserves_coalescing) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Mix of coalesceable points and blend corners
    const xmatrix<> waypoints = {{0.0, 0.0},
                                 {0.5, 0.0},  // Should coalesce
                                 {1.0, 0.0},  // Corner requiring blend
                                 {1.0, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_linear_deviation(0.1).set_max_blend_deviation(0.1));

    // Should coalesce first segment, then blend at corner: linear, blend, linear = 3 segments
    BOOST_CHECK_EQUAL(p.size(), 3U);

    verify_path_visits_waypoints(p, waypoints, 0.1);
}

BOOST_AUTO_TEST_CASE(circular_blend_insufficient_segment_length) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Corner with segments too short for requested blend
    const xmatrix<> waypoints = {
        {0.0, 0.0},
        {0.1, 0.0},  // Very short segment
        {0.1, 0.1}   // Very short segment
    };

    // Request large blend that won't fit fully
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.5));

    // Should create a blend limited by segment length (ℓ = min(0.05, 0.05, ...))
    // Result: trimmed linear, circular blend, trimmed linear = 3 segments
    BOOST_CHECK_EQUAL(p.size(), 3U);

    // Verify endpoints
    auto config_start = p.configuration(arc_length{0.0});
    auto config_end = p.configuration(p.length());
    BOOST_CHECK(configs_close(config_start, xvector<>{0.0, 0.0}));
    BOOST_CHECK(configs_close(config_end, xvector<>{0.1, 0.1}));
}

BOOST_AUTO_TEST_CASE(circular_blend_collinear_no_blend) {
    using namespace viam::trajex::totg;

    // Collinear points should not create blend (angle is 180 degrees)
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Should have 2 linear segments, no blend (points are collinear)
    BOOST_CHECK_EQUAL(p.size(), 2U);

    for (auto seg : p) {
        BOOST_CHECK(seg.is<path::segment::linear>());
    }
}

// This is the regression test for RSDK-13338. Three 6-DOF waypoints from a GP12 realtime plan form
// a corner that turns by less than acos can resolve (its floor is 1.5e-8 rad). The corner must
// blend at the curvature cap and the path must meet its contract.
BOOST_AUTO_TEST_CASE(circular_blend_near_collinear_dense_waypoints_blends_at_curvature_cap) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {
        {0.8133007953857693, 0.07784110371418973, 0.0616186421878857, -0.5531657430677487, 0.03481448567149956, 3.7011475425398173},
        {0.8120016837615454, 0.0789811538169164, 0.06568920373661674, -0.5526286189764857, 0.03142004812961326, 3.7009971572529694},
        {0.8107025721373212, 0.08012120391964306, 0.06975976528534782, -0.5520914948852225, 0.028025610587726943, 3.7008467719661216},
    };

    // Production config from the failed-plan replay: blend tolerance 0.1, no colinearization,
    // default curvature limits.
    const auto opts = path::options{}.set_max_blend_deviation(0.1);
    const path p = path::create(waypoints, opts);

    BOOST_REQUIRE_EQUAL(path_type_sequence(p), "LCL");

    auto it = p.begin();
    const auto arc = *++it;
    const auto curvature = xt::norm_l2(arc.curvature(arc.start()))();
    constexpr auto k_min_curvature = path::options::k_default_min_blend_curvature;
    BOOST_CHECK_LE(std::abs(curvature - k_min_curvature), k_min_curvature * curvature_refit_bound(arc));

    validate_path_quality(p, waypoint_accumulator{waypoints}, opts);
}

// This is the regression test for RSDK-14911. Three consecutive waypoints from a GP8 sanding move
// form a corner that turns by 1.467e-13 rad, far below what acos can resolve, and the radius sits
// at the min_blend_curvature cap of 1e5. The arc must stay on the path.
//
// The expected geometry is what the bounds imply for the corner's true angle. The deviation budget
// doesn't bind, so the trim is the min_blend_curvature cap, tan(angle / 2) / kmin, the curvature is
// kmin up to the refit of the radius to the arc's span on the path, and the arc sweeps the true angle.
// The angle and arc length get a 1% tolerance, which is how closely doubles resolve a turn this
// small. Any change in which case these bounds produce here should be a deliberate change to this
// test.
BOOST_AUTO_TEST_CASE(circular_blend_at_min_curvature_cap_stays_on_path) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {
        {-0.13380480958993365, 1.0989793178952696, -0.15096002912280768, 3.29841079653297, 0.2920799930934756, -3.6821355908152187},
        {-0.12555390621904083, 1.0971638332748608, -0.15472811893622324, 3.2964569348239867, 0.28958239753330123, -3.6891124956202073},
        {-0.11730300284814804, 1.0953483486544517, -0.15849620874963882, 3.294503073115002, 0.28708480197312675, -3.696089400425196},
    };

    // This is the production config: blend tolerance 0.1, no colinearization, and the default curvature limits.
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    BOOST_REQUIRE_EQUAL(path_type_sequence(p), "LCL");

    // The trim is tan(angle / 2) / kmin for the true angle, computed exactly from these waypoints.
    constexpr double k_expected_angle = 1.46716194234704914e-13;
    constexpr double k_expected_arc_length = 1.46716194234704914e-8;
    constexpr double k_expected_first_linear_length = 1.20126797700223382e-2;
    constexpr double k_expected_last_linear_length = 1.20126797700226734e-2;

    auto it = p.begin();
    const auto first = *it++;
    const auto arc = *it++;
    const auto last = *it;

    BOOST_CHECK_SMALL(static_cast<double>(first.end() - first.start()) - k_expected_first_linear_length, 1e-10);
    BOOST_CHECK_SMALL(static_cast<double>(last.end() - last.start()) - k_expected_last_linear_length, 1e-10);
    BOOST_CHECK_CLOSE(static_cast<double>(arc.end() - arc.start()), k_expected_arc_length, 1.0);
    const auto curvature = xt::norm_l2(arc.curvature(arc.start()))();
    constexpr auto k_min_curvature = path::options::k_default_min_blend_curvature;
    BOOST_CHECK_LE(std::abs(curvature - k_min_curvature), k_min_curvature * curvature_refit_bound(arc));
    arc.visit([&](const auto& data) {
        if constexpr (std::is_same_v<std::decay_t<decltype(data)>, path::segment::circular>) {
            BOOST_CHECK_CLOSE(data.angle_rads, k_expected_angle, 1.0);
        }
    });

    // Every segment meets its neighbours, and the path runs from the first waypoint to the last.
    BOOST_CHECK(configs_close(first.configuration(first.end()), arc.configuration(arc.start()), 1e-12));
    BOOST_CHECK(configs_close(arc.configuration(arc.end()), last.configuration(last.start()), 1e-12));
    BOOST_CHECK(configs_close(p.configuration(arc_length{0.0}), xvector<>{xt::view(waypoints, 0, xt::all())}, 1e-12));
    BOOST_CHECK(configs_close(p.configuration(p.length()), xvector<>{xt::view(waypoints, 2, xt::all())}, 1e-12));
}

// Corners that nearly reverse, blended into arcs. An arc's frame and angle come from the sum and
// the difference of the incoming and outgoing directions, and at a near reversal it is the sum that
// cancels, so this is where the frame is hardest to get right. Each end must meet its line within
// the path's contract, which validate_path_quality enforces, and the arc must sweep the true angle.
BOOST_AUTO_TEST_CASE(circular_blend_near_reversal_meets_its_lines) {
    using namespace viam::trajex::totg;

    // The angle is twice the arctangent of a ratio of two norms, each a sum of dof squares of
    // coordinates that each rounded once, so its relative error is a few times dof eps, carried
    // across an angle of at most pi.
    constexpr std::size_t k_dof = 3;
    constexpr double k_angle_bound = (k_dof + 4) * std::numbers::pi * std::numeric_limits<double>::epsilon();

    // The directions are off the axes so that every coordinate rounds.
    const xvector<> in = xvector<>{1.0, 2.0, 3.0} / std::sqrt(14.0);
    const xvector<> perpendicular = xvector<>{3.0, 0.0, -1.0} / std::sqrt(10.0);
    const xvector<> corner = {0.3, -0.7, 1.1};

    for (const double gap : {1e-1, 1e-2, 1e-3}) {
        BOOST_TEST_CONTEXT("pi - angle = " << gap) {
            const xvector<> out = (-std::cos(gap) * in) + (std::sin(gap) * perpendicular);
            xmatrix<> waypoints = xt::zeros<double>({std::size_t{3}, k_dof});
            xt::view(waypoints, 0, xt::all()) = corner - in;
            xt::view(waypoints, 1, xt::all()) = corner;
            xt::view(waypoints, 2, xt::all()) = corner + out;

            const auto opts = path::options{}.set_max_blend_deviation(0.1);
            const path p = path::create(waypoints, opts);
            BOOST_REQUIRE_EQUAL(path_type_sequence(p), "LCL");

            auto it = p.begin();
            const auto arc = *++it;
            arc.visit([&](const auto& data) {
                if constexpr (std::is_same_v<std::decay_t<decltype(data)>, path::segment::circular>) {
                    BOOST_CHECK_SMALL(data.angle_rads - (std::numbers::pi - gap), k_angle_bound);
                }
            });

            validate_path_quality(p, waypoint_accumulator{waypoints}, opts);
        }
    }
}

BOOST_AUTO_TEST_CASE(circular_blend_very_sharp_angle) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Nearly 180-degree turn (very sharp bend back)
    const xmatrix<> waypoints = {
        {0.0, 0.0}, {1.0, 0.0}, {0.1, 0.1}  // Nearly reverses direction
    };

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Should create blend even for sharp angle
    BOOST_CHECK_GE(p.size(), 3U);

    verify_path_visits_waypoints(p, waypoints, 0.1);
}

BOOST_AUTO_TEST_CASE(circular_blend_higher_dimensions) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // 3D path with corner
    const xmatrix<> waypoints = {{0.0, 0.0, 0.0},
                                 {1.0, 0.0, 0.0},  // Corner in XY plane
                                 {1.0, 1.0, 0.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    BOOST_CHECK_EQUAL(p.size(), 3U);
    BOOST_CHECK_EQUAL(p.dof(), 3U);

    verify_path_visits_waypoints(p, waypoints, 0.1);
}

BOOST_AUTO_TEST_CASE(circular_blend_blend_radius_geometry) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Verify blend geometry: 90-degree corner with known max_deviation
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const double max_dev = 0.1;
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(max_dev));

    // For 90-degree turn using Kunz & Stilman formulas (eq. 3-4):
    // Trim: ℓ = δ * sin(α/2) / (1 - cos(α/2))
    // Radius: r = ℓ / tan(α/2)
    const double half_angle = degrees_to_radians(45.0);
    const double expected_trim = max_dev * std::sin(half_angle) / (1.0 - std::cos(half_angle));
    const double expected_radius = expected_trim / std::tan(half_angle);

    // Path length should be: (1 - trim) + (radius * π/2) + (1 - trim)
    const double expected_length = (2.0 * (1.0 - expected_trim)) + ((expected_radius * std::numbers::pi) / 2.0);
    BOOST_CHECK_CLOSE(static_cast<double>(p.length()), expected_length, 0.1);  // 0.1% tolerance
}

BOOST_AUTO_TEST_CASE(circular_blend_start_end_waypoints_exact) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Start and end waypoints must be exact even with blends
    const xmatrix<> waypoints = {
        {0.5, 0.3},  // Arbitrary start
        {1.7, 0.8},  // Corner
        {2.1, 1.9}   // Arbitrary end
    };

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.2));

    auto config_start = p.configuration(arc_length{0.0});
    auto config_end = p.configuration(p.length());

    // Start and end must be exact (machine precision)
    BOOST_CHECK_CLOSE(config_start(0), 0.5, 1e-10);
    BOOST_CHECK_CLOSE(config_start(1), 0.3, 1e-10);
    BOOST_CHECK_CLOSE(config_end(0), 2.1, 1e-10);
    BOOST_CHECK_CLOSE(config_end(1), 1.9, 1e-10);
}

BOOST_AUTO_TEST_CASE(circular_blend_symmetric_corners) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Symmetric path - blends should be symmetric
    const xmatrix<> waypoints = {
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}  // Square path
    };

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Should have 4 corners with 4 blends: 4 linear + 4 circular = 8 segments
    // (last segment closes back to start, but no blend at start/end)
    BOOST_CHECK_GE(p.size(), 7U);  // At least 4 corners blended
}

BOOST_AUTO_TEST_CASE(circular_blend_different_blend_and_linear_tolerances) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Use different tolerances for coalescing and blending
    const xmatrix<> waypoints = {{0.0, 0.0},
                                 {0.5, 0.05},  // Slightly off line
                                 {1.0, 0.0},   // Corner
                                 {1.0, 1.0}};

    // Large linear deviation to coalesce, small blend deviation
    const path p = path::create(waypoints,
                                path::options{}
                                    .set_max_linear_deviation(0.1)    // Coalesce middle point
                                    .set_max_blend_deviation(0.05));  // Small blend

    // Middle point should coalesce, corner should blend
    BOOST_CHECK_EQUAL(p.size(), 3U);
    verify_path_visits_waypoints(p, waypoints, 0.1);
}

BOOST_AUTO_TEST_CASE(circular_blend_no_coalescing_only_blending) {
    using namespace viam::trajex::totg;

    // Test blending without coalescing enabled
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints,
                                path::options{}
                                    .set_max_linear_deviation(0.0)   // No coalescing
                                    .set_max_blend_deviation(0.1));  // Only blending

    // Should have blend at corner
    BOOST_CHECK_EQUAL(p.size(), 3U);

    // Verify second segment is circular
    auto it = p.begin();
    ++it;
    BOOST_CHECK((*it).is<path::segment::circular>());
}

BOOST_AUTO_TEST_CASE(circular_blend_consecutive_corners) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Two corners very close together
    const xmatrix<> waypoints = {{0.0, 0.0},
                                 {1.0, 0.0},  // First corner
                                 {1.1, 0.1},  // Second corner very close
                                 {1.1, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.05));

    // Should handle consecutive corners appropriately
    BOOST_CHECK_GE(p.size(), 3U);
    verify_path_visits_waypoints(p, waypoints, 0.05);
}

BOOST_AUTO_TEST_CASE(circular_blend_path_continuity) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Verify path is continuous across blend boundaries
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Sample at segment boundaries - should be continuous
    auto it = p.begin();
    auto seg1_end = (*it).end();
    ++it;
    auto seg2_start = (*it).start();
    auto seg2_end = (*it).end();
    ++it;
    auto seg3_start = (*it).start();

    // Boundaries should match
    BOOST_CHECK_CLOSE(static_cast<double>(seg1_end), static_cast<double>(seg2_start), 1e-10);
    BOOST_CHECK_CLOSE(static_cast<double>(seg2_end), static_cast<double>(seg3_start), 1e-10);

    // Configurations at boundaries should match
    auto config_boundary1 = p.configuration(seg1_end);
    auto config_boundary1_next = p.configuration(seg2_start);
    BOOST_CHECK(configs_close(config_boundary1, config_boundary1_next));
}

BOOST_AUTO_TEST_CASE(circular_blend_tangent_continuity) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Verify tangent vectors are continuous at blend boundaries
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    // Get tangents just before and just after segment boundaries
    auto it = p.begin();
    auto seg1_end = (*it).end();
    ++it;
    auto seg2_start = (*it).start();

    // Tangent at end of linear segment
    auto tangent1 = p.tangent(arc_length{static_cast<double>(seg1_end) - 1e-6});
    // Tangent at start of blend
    auto tangent2 = p.tangent(arc_length{static_cast<double>(seg2_start) + 1e-6});

    // Tangents should be close (C1 continuity)
    const double dot_product = xt::sum(tangent1 * tangent2)();
    BOOST_CHECK_CLOSE(dot_product, 1.0, 1.0);  // Within 1% (small angle tolerance)
}

// Regression: blend trim must cap against the original (waypoint-to-waypoint) segment length,
// not the remaining length after a previous blend consumed its share. Kunz & Stilman Section IV:
// "We also need to ensure that the circular segment does not replace more than half of each of
// the neighboring linear segments." Neighboring means the original path segments.
//
// Z-staircase: (0,0) -> (0,1) -> (1,1) -> (1,2). Three unit segments, two 90-degree corners.
// With geometry-limited deviation (100.0 >> needed), each blend trims min(L1/2, L2/2) = 0.5 from
// the shared middle segment. Together they consume it entirely, so no linear segment appears
// between the two circular blends -- the path is LCCL.
BOOST_AUTO_TEST_CASE(circular_blend_trim_respects_original_segment_length) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 2.0}};
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(100.0));

    // The two blends together consume the full middle segment: LCCL, no linear between them.
    BOOST_REQUIRE_EQUAL(p.size(), 4U);

    auto it = p.begin();
    BOOST_CHECK((*it).is<path::segment::linear>());
    ++it;
    BOOST_CHECK((*it).is<path::segment::circular>());
    ++it;
    BOOST_CHECK((*it).is<path::segment::circular>());
    ++it;
    BOOST_CHECK((*it).is<path::segment::linear>());

    BOOST_CHECK(configs_close(p.configuration(arc_length{0.0}), xvector<>{0.0, 0.0}));
    BOOST_CHECK(configs_close(p.configuration(p.length()), xvector<>{1.0, 2.0}));
}

// Regression (RSDK-12771): blend trim can leave an incoming linear whose arc-length
// contribution is representable relative to dist_to_locus (passing the existing guard)
// but not relative to cumulative_length, producing a zero-width segment view that traps
// path::cursor in an infinite loop during forward integration.
//
// Waypoints extracted from a real 7000-waypoint dense trajectory capture that triggered
// the hang. The blend at waypoint [20] trims the [19]->[20] incoming segment down to
// ~6.9e-18 radians, which is below ULP of the cumulative arc length (~0.118 rad) at
// that point.
BOOST_AUTO_TEST_CASE(blend_trim_zero_width_segment) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {
        {0.82739107946303503, 0.88282048800603485, 0.3039375847975771, -3.136198473590786, 0.98778065081918609, 0.74039620121758376},
        {0.82629088242947146, 0.88015395319311995, 0.29929061711574401, -3.1361612990406651, 0.98576635111271682, 0.74146624876782663},
        {0.82518902514739223, 0.87750093630902959, 0.29467042227130935, -3.1361212806604346, 0.98376585935290217, 0.7425369173880072},
        {0.82408439141808032, 0.87486007035523905, 0.29007430703974835, -3.136083340709396, 0.98177575096874348, 0.74361269537785513},
        {0.82297766704596664, 0.87223282218850162, 0.28550514470579086, -3.1360432784270205, 0.97980134663434992, 0.7446863591792573},
        {0.82175093188676929, 0.86934079066277903, 0.28047842559826797, -3.1359989040577294, 0.97762778247494508, 0.74587804588963502},
        {0.82052122478846157, 0.86646396202641407, 0.27548181447209474, -3.1359544227250566, 0.97546946137410784, 0.74707256825793555},
        {0.81928851589217333, 0.86360202999044611, 0.27051468583278532, -3.1359098886631349, 0.97332587508569313, 0.74827007078641894},
        {0.8180527967645177, 0.86075441708409506, 0.26557607257769117, -3.135865526464737, 0.97119590782824416, 0.74947043923950929},
        {0.81691547078297266, 0.85815210566197009, 0.26106578102812716, -3.1358240998866274, 0.96925181642027836, 0.75057491150423716},
        {0.81577544695593207, 0.85556188147923184, 0.25657946439116519, -3.1357834811994563, 0.9673198495138623, 0.75168229175030732},
        {0.81463322511546443, 0.85298334430198652, 0.25211648865248959, -3.135741171041222, 0.96539921603621814, 0.75279123497030176},
        {0.8134882050117741, 0.8504162266389127, 0.24767611227211728, -3.1356999702528503, 0.96348915314636985, 0.75390332968205465},
        {0.81239977300000277, 0.8479915334672864, 0.24348418047826675, -3.1356649955963296, 0.96168408895129498, 0.75496095169515587},
        {0.81130993554969921, 0.84557935592151812, 0.23931712067998823, -3.135625154121938, 0.95989451529514969, 0.75601903359253686},
        {0.81021853068193705, 0.84317907848638818, 0.23517379397324567, -3.135581307674193, 0.95811895749286435, 0.75707826471125217},
        {0.80912400325497646, 0.8407869536014575, 0.23104650713052508, -3.1355416437658326, 0.95634888710197274, 0.75814095295035044},
        {0.8079804871517845, 0.8383033321852208, 0.2267640376073147, -3.135500013460045, 0.95451298609643331, 0.75925112318759469},
        {0.80683444819414352, 0.83583023284298352, 0.22250259711775061, -3.1354584901952416, 0.95268732223556085, 0.76036387690474327},
        {0.80568580309839632, 0.83336896720205433, 0.2182642674204768, -3.1354171697807547, 0.95087545998514833, 0.76147925931929172},
        {0.80453379071711051, 0.8309134890629436, 0.21403804528617396, -3.1353803509698968, 0.94906219693260463, 0.76259848675258046},
        {0.80348035729964573, 0.82868511774530973, 0.21020648152308605, -3.1353351970286405, 0.94743226524784308, 0.76362086319744726},
        {0.80242316594544671, 0.82645762028543435, 0.20637754406775707, -3.1352985688288966, 0.94579144498181067, 0.76464635430345096},
        {0.80136455609434587, 0.82424371145004272, 0.20257472793898343, -3.1352581155276646, 0.94417228036033596, 0.76567419014892324},
        {0.80030343340426746, 0.82203557017046813, 0.19878386991668273, -3.1352196593643331, 0.94255472609856017, 0.7667040771997794},
        {0.79914710824348445, 0.8196442013430919, 0.19468049525035699, -3.1351802801979831, 0.94080381320088813, 0.76782674729827727},
        {0.79798873121757508, 0.81726319384717827, 0.19059777012593726, -3.1351382211247683, 0.93906420782762101, 0.7689510613676801},
    };

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));

    for (const auto& seg : p) {
        BOOST_CHECK(seg.end() > seg.start());
    }
}

// Regression: when a circular blend is followed by colinearization, the tube extends in
// the direction of the far endpoint (segment_start → locus), which may differ from the
// blend's outgoing tangent (toward the first skipped waypoint). The merged linear segment
// then runs between two directions that don't match the blend boundaries on either side,
// producing tangent discontinuities at both the C→L and L→C junctions.
//
// Geometry (2-DOF):
//   W0=(0,3) → W1=(0,0): blend here (~128° corner)
//   W2=(0.5,0.4): within the W1→W3 colinearization tube (deviation=0.4 < tube_half=0.5)
//                 but NOT on the W1→W3 axis — blend at W1 exits along normalize(W2-W1)
//   W3=(3,0): next corner (90° turn to W4)
//   W4=(3,3): exit
//
// With max_linear_deviation=1.0, W2 is absorbed. The merged linear connects the W1 blend
// exit (on the W1→W2 axis) to the W3 blend entry (on the W1→W3 axis). The C→L junction
// after W1 has a ~42° tangent mismatch; the L→C junction before W3 has a ~4° mismatch.
BOOST_AUTO_TEST_CASE(colinearization_after_blend_preserves_tangent_continuity) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {
        {0.0, 3.0},  // W0
        {0.0, 0.0},  // W1: corner
        {0.5, 0.4},  // W2: in tube of W1->W3, off the W1->W3 axis
        {3.0, 0.0},  // W3: corner
        {3.0, 3.0},  // W4
    };

    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.5).set_max_linear_deviation(1.0));

    BOOST_REQUIRE_EQUAL(p.size(), 5U);
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LCLCL");

    // At every segment junction the path must be C1: the tangent just before and just
    // after the boundary must agree to within a tight tolerance.
    auto prev = p.begin();
    auto curr = std::next(prev);
    while (curr != p.end()) {
        const auto t_before = (*prev).tangent((*prev).end());
        const auto t_after = (*curr).tangent((*curr).start());
        const double dot = xt::sum(t_before * t_after)();
        BOOST_CHECK_CLOSE(dot, 1.0, 0.01);
        prev = curr;
        ++curr;
    }
}

BOOST_AUTO_TEST_SUITE_END()

// Tests for path construction at degenerate and near-degenerate waypoint geometries.
// These verify the path-layer decisions (segment types, dropping waypoints) that are
// prerequisites for correct TOTG behavior at cusps and near-cusps. All tests express
// desired end-state behavior; most will fail until the implementation is complete.

BOOST_AUTO_TEST_SUITE(extremal_path_construction_tests)

// Exact collinear: middle point lies exactly on the chord. The blend radius is a
// singularity (infinite), so no blend is emitted. Two linear segments remain.
BOOST_AUTO_TEST_CASE(exact_collinear_no_blend) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}};
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LL");
}

// Near-collinear: the middle waypoint deviates from the chord by 0.001, well within
// max_blend_deviation=0.1. The natural blend radius is enormous; min_curvature caps it
// at 1/min_curvature, producing a small-radius arc that retains C1 continuity.
// Structure is LCL; the arc radius is capped, not the path topology.
BOOST_AUTO_TEST_CASE(near_collinear_emits_capped_arc) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.001}, {2.0, 0.0}};
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LCL");
}

// Same near-collinear geometry, but with a tight min_blend_curvature=1.0 (max radius=1.0).
// The natural blend radius is about 500; the cap clamps it to 1.0, which the refit of the radius to
// the arc's span on the path may move by its bound and no more.
BOOST_AUTO_TEST_CASE(near_collinear_arc_radius_respects_min_curvature) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.001}, {2.0, 0.0}};
    constexpr double k_min_curvature = 1.0;
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1).set_min_blend_curvature(k_min_curvature));
    BOOST_REQUIRE_EQUAL(path_type_sequence(p), "LCL");

    const auto arc = *std::next(p.begin());
    const auto curvature = xt::norm_l2(arc.curvature(arc.start()))();
    BOOST_CHECK_LE(std::abs(curvature - k_min_curvature), k_min_curvature * curvature_refit_bound(arc));
}

// Near-collinear outside tolerance: cross-track deviation (0.5) exceeds max_blend_deviation
// (0.1), so the waypoint represents a genuine direction change and a blend is emitted.
BOOST_AUTO_TEST_CASE(near_collinear_outside_deviation_produces_blend) {
    using namespace viam::trajex::totg;
    // Cross-track deviation of (1, 0.5) from chord (0,0)-(2,0) is 0.5 > max_blend_deviation=0.1.
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.5}, {2.0, 0.0}};
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LCL");
}

// Exact reversal: path goes A -> B -> A. No blend arc is geometrically valid at a true
// reversal; the path must be L-L.
BOOST_AUTO_TEST_CASE(exact_reversal_emits_ll) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1));
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LL");
}

// Near-reversal within the curvature band: a 170-degree turn with max_blend_curvature=100
// produces a blend with curvature ~105 (radius ~0.0096), which exceeds the limit.
// The near-reversal is too tight to blend usefully; emit L-L.
BOOST_AUTO_TEST_CASE(near_reversal_within_curvature_band_emits_ll) {
    using namespace viam::trajex::totg;
    // Outgoing direction at 170 degrees from incoming (1,0): corner at (1,0), third waypoint
    // at (1+cos(170deg), sin(170deg)).
    const double angle_rad = degrees_to_radians(170.0);
    const xmatrix<> waypoints = {
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0 + std::cos(angle_rad), std::sin(angle_rad)},
    };
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1).set_max_blend_curvature(100.0));
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LL");
}

// Non-reversal corner with max_blend_curvature set: a 90-degree turn has curvature ~4,
// well below max_blend_curvature=100, so a blend is emitted normally.
BOOST_AUTO_TEST_CASE(non_reversal_corner_outside_curvature_band_emits_lcl) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.1).set_max_blend_curvature(100.0));
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LCL");
}

// Z-staircase: three equal-length segments with two 90-degree corners. With a large
// deviation budget, each blend consumes its full half-segment. Together the two blends
// consume the middle segment entirely, producing L-C-C-L (4 segments).
BOOST_AUTO_TEST_CASE(z_staircase_adjacent_blends_emit_lccl) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 2.0}};
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(100.0));
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LCCL");
}

// The same topology as the Z-staircase, but on waypoints where the rounding does not work
// out. Both blends take exactly half of the short middle leg, so nothing of it remains, yet
// subtracting the second trim from the distance left after the first blend comes out two ULPs
// long, which the representability guard keeps. Field waypoints from orbsanding_1d6726c5,
// where that sliver became the start of an infeasible switching point.
BOOST_AUTO_TEST_CASE(blends_taking_half_of_the_same_leg_emit_adjacent_arcs) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {
        {4.035506923596518, -0.6872436019104348, -1.1330972471743401, 3.1511543100874206, 0.44919960780984386, -0.010755407914024963},
        {4.038372872954921, -0.6849832425285033, -1.1291366551604054, 3.1510064128249624, 0.44741562156588255, -0.010581739767723697},
        {4.038593051686501, -0.6848457368400501, -1.1288382298157824, 3.1509948730460766, 0.44724917122787927, -0.010571710262542519},
        {4.043258608017244, -0.6822986225264079, -1.124314005536449, 3.142798195912856, 0.4445389556470222, -0.0013979901641557936},
    };
    const path p = path::create(waypoints, path::options{}.set_max_blend_deviation(0.008726646259971648));
    BOOST_CHECK_EQUAL(path_type_sequence(p), "LCCL");
}

// This corner's arc would span less than the path can resolve where it falls. A turn of 1e-6 at the
// radius a min_blend_curvature of 1e12 allows is an arc 1e-18 long, starting at arc length 1, where
// adjacent doubles are 2.2e-16 apart. There is nowhere on the path to put that arc, so the corner
// must be left unblended rather than emitted as a segment with no interior.
BOOST_AUTO_TEST_CASE(corner_whose_arc_has_no_span_is_left_unblended) {
    using namespace viam::trajex::totg;
    constexpr auto k_turn = 1e-6;
    constexpr auto k_curvature = 1e12;
    const auto waypoints = xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.0 + std::cos(k_turn), std::sin(k_turn)}};
    const auto opts =
        path::options{}.set_max_blend_deviation(0.1).set_min_blend_curvature(k_curvature).set_max_blend_curvature(k_curvature);
    const auto p = path::create(waypoints, opts);

    BOOST_CHECK_EQUAL(path_type_sequence(p), "LL");
    validate_path_quality(p, waypoint_accumulator{waypoints}, opts);
}

// Two corners are joined by a short leg whose blends leave a line of 1e-6 between them. The line starts
// at the first blend's exit point, which is rounded, so a direction rebuilt from there to the next
// corner, about half the leg away, would be off by about eps |q| over half the leg: hundreds of eps
// for a 1e-3 leg. The line takes the leg's direction instead, so path::k_tangent_error_bound_ulps
// holds it as it holds each arc's end, and the two sides of each join are within twice that of each
// other. The two corners turn by the same angle, so one deviation sets both trims.
BOOST_AUTO_TEST_CASE(short_line_between_blends_meets_its_arcs) {
    using namespace viam::trajex::totg;
    using viam::trajex::totg::test::angle_between;
    constexpr auto k_join_tangent_bound = 2.0 * path::k_tangent_error_bound_ulps * std::numeric_limits<double>::epsilon();
    constexpr auto k_leg = 1e-3;
    constexpr auto k_leftover = 1e-6;

    // The directions are off the axes so that every coordinate rounds. The third is the first
    // reflected through the second, so both corners turn by the angle between the first two.
    const xvector<> first_direction = xvector<>{1.0, 2.0, 3.0} / std::sqrt(14.0);
    const xvector<> second_direction = xvector<>{3.0, 1.0, -1.0} / std::sqrt(11.0);
    const xvector<> third_direction = (2.0 * xt::sum(first_direction * second_direction)() * second_direction) - first_direction;
    const auto corner = xvector<>{0.3, -0.7, 1.1};

    xmatrix<> waypoints = xt::zeros<double>({std::size_t{4}, std::size_t{3}});
    xt::view(waypoints, 0, xt::all()) = corner - first_direction;
    xt::view(waypoints, 1, xt::all()) = corner;
    xt::view(waypoints, 2, xt::all()) = corner + (k_leg * second_direction);
    xt::view(waypoints, 3, xt::all()) = corner + (k_leg * second_direction) + third_direction;

    // Each blend trims deviation / tan(turn / 4) from the middle leg, and together they leave
    // k_leftover of it.
    const auto turn = angle_between(first_direction, second_direction);
    const auto deviation = ((k_leg - k_leftover) / 2.0) * std::tan(turn / 4.0);
    const auto opts = path::options{}.set_max_blend_deviation(deviation);
    const auto p = path::create(waypoints, opts);
    BOOST_REQUIRE_EQUAL(path_type_sequence(p), "LCLCL");

    auto it = std::next(p.begin());
    const auto first_arc = *it++;
    const auto line = *it++;
    const auto second_arc = *it;
    BOOST_REQUIRE_LT(static_cast<double>(line.length()), 2 * k_leftover);

    BOOST_CHECK(configs_close(first_arc.tangent(first_arc.end()), line.tangent(line.start()), k_join_tangent_bound));
    BOOST_CHECK(configs_close(line.tangent(line.end()), second_arc.tangent(second_arc.start()), k_join_tangent_bound));
    validate_path_quality(p, waypoint_accumulator{waypoints}, opts);
}

// This corner turns by 1e-9, which the dot product of its two directions cannot see, since cos(1e-9)
// rounds to exactly 1. Here the outgoing direction (1, 1e-9) is its own normalization, so the dot
// product is exactly 1 although the directions differ. The corner turns, so it must be blended.
// path::k_tangent_error_bound_ulps holds each side of a join to within that many ULPs of the tangent
// they share, and so to twice that of each other.
BOOST_AUTO_TEST_CASE(corner_turning_below_dot_product_resolution_is_blended) {
    using namespace viam::trajex::totg;
    constexpr auto k_join_tangent_bound = 2.0 * path::k_tangent_error_bound_ulps * std::numeric_limits<double>::epsilon();
    const auto waypoints = xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {2.0, 1e-9}};
    const auto opts = path::options{}.set_max_blend_deviation(0.1);
    const auto p = path::create(waypoints, opts);
    BOOST_REQUIRE_EQUAL(path_type_sequence(p), "LCL");

    auto it = p.begin();
    const auto first = *it++;
    const auto arc = *it++;
    const auto last = *it;
    BOOST_CHECK(configs_close(first.tangent(first.end()), arc.tangent(arc.start()), k_join_tangent_bound));
    BOOST_CHECK(configs_close(arc.tangent(arc.end()), last.tangent(last.start()), k_join_tangent_bound));
    validate_path_quality(p, waypoint_accumulator{waypoints}, opts);
}

BOOST_AUTO_TEST_SUITE_END()  // extremal_path_construction_tests
