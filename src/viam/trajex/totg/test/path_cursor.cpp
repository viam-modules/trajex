// Path cursor tests: sequential traversal and seeking
// Extracted from test.cpp lines 2259-2798

#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/xt.hpp>

#if __has_include(<xtensor/reducers/xnorm.hpp>)
#include <xtensor/reducers/xnorm.hpp>
#else
#include <xtensor/xnorm.hpp>
#endif

#include <boost/test/unit_test.hpp>

using viam::trajex::xmatrix;
using viam::trajex::xvector;

BOOST_AUTO_TEST_SUITE(path_cursor_tests)

BOOST_AUTO_TEST_CASE(construct_and_get_path_reference) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 0.0}};
    const path p = path::create(waypoints);

    const path::cursor cursor = p.create_cursor();

    // Should have reference to path
    BOOST_CHECK_EQUAL(&cursor.path(), &p);
    BOOST_CHECK_EQUAL(cursor.path().length(), p.length());
}

BOOST_AUTO_TEST_CASE(construct_with_minimal_path) {
    using namespace viam::trajex::totg;

    // Create minimal valid path (2 distinct waypoints)
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    // Should be able to construct cursor with minimal path
    BOOST_CHECK_NO_THROW({
        auto c = p.create_cursor();
        (void)c;
    });
}

BOOST_AUTO_TEST_CASE(initial_position_at_start) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 1.0}};
    const path p = path::create(waypoints);

    const path::cursor cursor = p.create_cursor();

    BOOST_CHECK_EQUAL(cursor.position(), arc_length{0.0});
    BOOST_CHECK(cursor.position() != p.length());
    BOOST_CHECK_EQUAL(static_cast<double>(cursor.position()), 0.0);
}

BOOST_AUTO_TEST_CASE(forward_integration_small_steps) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    // Take several small forward steps
    cursor.seek_by(arc_length{0.1});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.1, 0.01);
    BOOST_CHECK(cursor.position() != arc_length{0.0});
    BOOST_CHECK(cursor.position() != p.length());

    cursor.seek_by(arc_length{0.3});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.4, 0.01);

    cursor.seek_by(arc_length{0.6});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 1.0, 0.01);
    BOOST_CHECK_EQUAL(cursor.position(), p.length());
}

BOOST_AUTO_TEST_CASE(forward_integration_to_end) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 1.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    // Advance past end (should overflow to sentinel)
    cursor.seek_by(p.length() + arc_length{10.0});

    BOOST_CHECK(cursor == cursor.end());
}

BOOST_AUTO_TEST_CASE(backward_integration_small_steps) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    // Start at end
    cursor.seek(p.length());
    BOOST_CHECK_EQUAL(cursor.position(), p.length());

    // Take several small backward steps (negative delta)
    cursor.seek_by(arc_length{-0.2});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.8, 0.01);
    BOOST_CHECK(cursor.position() != arc_length{0.0});
    BOOST_CHECK(cursor.position() != p.length());

    cursor.seek_by(arc_length{-0.3});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.5, 0.01);

    cursor.seek_by(arc_length{-0.5});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.0, 0.01);
    BOOST_CHECK_EQUAL(cursor.position(), arc_length{0.0});
}

BOOST_AUTO_TEST_CASE(backward_integration_to_start) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 1.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    cursor.seek(p.length());

    // Advance past start (should underflow to sentinel)
    cursor.seek_by(arc_length{-100.0});

    BOOST_CHECK(cursor == cursor.end());
}

BOOST_AUTO_TEST_CASE(bidirectional_integration) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    // Forward then backward
    cursor.seek_by(arc_length{0.6});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.6, 0.01);

    cursor.seek_by(arc_length{-0.2});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.4, 0.01);

    cursor.seek_by(arc_length{0.5});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.9, 0.01);

    // This goes slightly past 0, so should underflow to sentinel
    cursor.seek_by(arc_length{-0.9});
    BOOST_CHECK(cursor == cursor.end());
}

BOOST_AUTO_TEST_CASE(reset_to_start_and_end) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    cursor.seek_by(arc_length{0.5});

    cursor.seek(arc_length{0.0});
    BOOST_CHECK_EQUAL(cursor.position(), arc_length{0.0});
    BOOST_CHECK_EQUAL(static_cast<double>(cursor.position()), 0.0);

    cursor.seek(p.length());
    BOOST_CHECK_EQUAL(cursor.position(), p.length());
    BOOST_CHECK_EQUAL(static_cast<double>(cursor.position()), static_cast<double>(p.length()));
}

BOOST_AUTO_TEST_CASE(reset_to_specific_position) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    cursor.seek(arc_length{0.3});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.3, 0.01);

    // Underflows to sentinel at start, overflows to sentinel at end
    cursor.seek(arc_length{-5.0});
    BOOST_CHECK(cursor == cursor.end());

    cursor.seek(arc_length{0.5});  // Reset to valid position
    BOOST_CHECK(cursor != cursor.end());

    cursor.seek(p.length() + arc_length{10.0});
    BOOST_CHECK(cursor == cursor.end());
}

BOOST_AUTO_TEST_CASE(configuration_query_at_positions) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 1.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    // At start
    auto config_start = cursor.configuration();
    BOOST_CHECK_CLOSE(config_start(0), 0.0, 0.01);
    BOOST_CHECK_CLOSE(config_start(1), 0.0, 0.01);

    // At midpoint
    cursor.seek(p.length() / 2.0);
    auto config_mid = cursor.configuration();
    BOOST_CHECK_CLOSE(config_mid(0), 0.5, 0.01);
    BOOST_CHECK_CLOSE(config_mid(1), 0.5, 0.01);

    // At end
    cursor.seek(p.length());
    auto config_end = cursor.configuration();
    BOOST_CHECK_CLOSE(config_end(0), 1.0, 0.01);
    BOOST_CHECK_CLOSE(config_end(1), 1.0, 0.01);
}

BOOST_AUTO_TEST_CASE(tangent_query) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    // Tangent should be constant along linear segment
    auto tangent_start = cursor.tangent();
    cursor.seek_by(arc_length{0.5});
    auto tangent_mid = cursor.tangent();

    BOOST_CHECK_CLOSE(tangent_start(0), 1.0, 0.01);  // Unit vector in x direction
    BOOST_CHECK_CLOSE(tangent_start(1), 0.0, 0.01);
    BOOST_CHECK_CLOSE(tangent_mid(0), 1.0, 0.01);
    BOOST_CHECK_CLOSE(tangent_mid(1), 0.0, 0.01);
}

BOOST_AUTO_TEST_CASE(curvature_query) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    const path::cursor cursor = p.create_cursor();

    // Curvature should be zero for linear segment
    auto curvature = cursor.curvature();
    BOOST_CHECK_SMALL(curvature(0), 1e-6);
    BOOST_CHECK_SMALL(curvature(1), 1e-6);
}

BOOST_AUTO_TEST_CASE(integration_across_multiple_segments) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Create path with circular blend
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
    path::options opts;
    opts.set_max_blend_deviation(0.1);
    const path p = path::create(waypoints, opts);

    // Path should have 3 segments: linear, circular, linear
    BOOST_CHECK_EQUAL(p.size(), 3U);

    path::cursor cursor = p.create_cursor();

    // Integrate forward across all segments
    const double step = 0.05;
    while (cursor != cursor.end()) {
        // Should be able to query at any position
        auto config = cursor.configuration();
        auto tangent = cursor.tangent();
        auto curvature = cursor.curvature();

        BOOST_CHECK_EQUAL(config.shape(0), 2U);
        BOOST_CHECK_EQUAL(tangent.shape(0), 2U);
        BOOST_CHECK_EQUAL(curvature.shape(0), 2U);

        cursor.seek_by(arc_length{step});
    }
}

BOOST_AUTO_TEST_CASE(backward_integration_across_multiple_segments) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Create path with circular blend
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
    path::options opts;
    opts.set_max_blend_deviation(0.1);
    const path p = path::create(waypoints, opts);

    path::cursor cursor = p.create_cursor();
    cursor.seek(p.length());

    // Integrate backward across all segments
    const double step = -0.05;  // Negative for backward
    while (cursor != cursor.end()) {
        // Should be able to query at any position
        auto config = cursor.configuration();
        BOOST_CHECK_EQUAL(config.shape(0), 2U);

        cursor.seek_by(arc_length{step});
    }
}

BOOST_AUTO_TEST_CASE(hint_optimization_forward_sequential) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Create path with many segments
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
    const path p = path::create(waypoints);

    path::cursor cursor = p.create_cursor();

    // Sequential forward access should benefit from hints
    // We can't measure performance directly, but verify correctness
    const double step = 0.1;
    while (cursor != cursor.end()) {
        // Verify queries work correctly (hint must be valid)
        auto config = cursor.configuration();
        BOOST_CHECK_EQUAL(config.shape(0), 2U);

        cursor.seek_by(arc_length{step});
    }
}

BOOST_AUTO_TEST_CASE(advance_by_and_sentinel_detection) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);
    path::cursor cursor = p.create_cursor();

    // Normal advancement - should advance full amount
    cursor.seek_by(arc_length{0.5});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.5, 0.01);
    BOOST_CHECK(cursor != cursor.end());

    // Advance to exactly the end - should be at length, not sentinel
    cursor.seek_by(arc_length{0.5});
    BOOST_CHECK_EQUAL(cursor.position(), p.length());
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 1.0, 0.01);
    BOOST_CHECK(cursor != cursor.end());

    // Try to advance past end - should overflow to sentinel
    cursor.seek_by(arc_length{0.1});
    BOOST_CHECK(cursor == cursor.end());
}

BOOST_AUTO_TEST_CASE(advance_by_negative_and_start_clamping) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);
    path::cursor cursor = p.create_cursor(p.length());  // Start at end

    // Normal backward advancement - should advance full amount (negative)
    cursor.seek_by(arc_length{-0.5});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.5, 0.01);
    BOOST_CHECK(cursor != cursor.end());

    // Try to advance past start - should underflow to sentinel
    cursor.seek_by(arc_length{-1.0});
    BOOST_CHECK(cursor == cursor.end());

    // Reset to start
    cursor.seek(arc_length{0.0});
    BOOST_CHECK_EQUAL(cursor.position(), arc_length{0.0});
    BOOST_CHECK(cursor != cursor.end());

    // Try to advance when at start - should underflow to sentinel
    cursor.seek_by(arc_length{-0.1});
    BOOST_CHECK(cursor == cursor.end());
}

BOOST_AUTO_TEST_CASE(seek_to_position_and_clamping) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);
    path::cursor cursor = p.create_cursor();

    // Seek to valid position
    cursor.seek(arc_length{0.5});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), 0.5, 0.01);
    BOOST_CHECK(cursor != cursor.end());

    // Try to seek past end - should overflow to sentinel
    cursor.seek(arc_length{10.0});
    BOOST_CHECK(cursor == cursor.end());

    // Seek before start - should underflow to sentinel
    cursor.seek(arc_length{-5.0});
    BOOST_CHECK(cursor == cursor.end());
}

BOOST_AUTO_TEST_CASE(create_cursor_at_various_positions) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const path p = path::create(waypoints);

    // Create at start (default)
    const path::cursor cursor1 = p.create_cursor();
    BOOST_CHECK_EQUAL(cursor1.position(), arc_length{0.0});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor1.position()), 0.0, 0.01);

    // Create at middle
    const path::cursor cursor2 = p.create_cursor(arc_length{0.5});
    BOOST_CHECK_CLOSE(static_cast<double>(cursor2.position()), 0.5, 0.01);
    BOOST_CHECK(cursor2.position() != arc_length{0.0});
    BOOST_CHECK(cursor2.position() != p.length());

    // Create at end
    const path::cursor cursor3 = p.create_cursor(p.length());
    BOOST_CHECK_EQUAL(cursor3.position(), p.length());
    BOOST_CHECK_CLOSE(static_cast<double>(cursor3.position()), 1.0, 0.01);

    // Create past end - should clamp
    const path::cursor cursor4 = p.create_cursor(arc_length{10.0});
    BOOST_CHECK_EQUAL(cursor4.position(), p.length());

    // Create before start - should clamp
    const path::cursor cursor5 = p.create_cursor(arc_length{-5.0});
    BOOST_CHECK_EQUAL(cursor5.position(), arc_length{0.0});
}

BOOST_AUTO_TEST_CASE(large_jumps_across_many_segments) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Create path with many segments (12 segments: 6 linear + 6 circular blends)
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {3.0, 1.0}, {3.0, 2.0}, {2.0, 2.0}, {1.0, 2.0}};
    path::options opts;
    opts.set_max_blend_deviation(0.1);
    const path p = path::create(waypoints, opts);

    // Should have multiple segments (linear + circular blends)
    BOOST_CHECK_GT(p.size(), 5U);  // At least 6 segments to test large jumps

    path::cursor cursor = p.create_cursor();

    // Perform large non-sequential jumps and verify correctness
    // These jumps exercise the binary search fallback in update_hint()

    // Jump to near end
    const arc_length pos1 = p.length() * 0.9;
    cursor.seek(pos1);
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), static_cast<double>(pos1), 0.01);
    auto config1 = cursor.configuration();
    BOOST_CHECK_EQUAL(config1.shape(0), 2U);

    // Jump back to near beginning
    const arc_length pos2 = p.length() * 0.1;
    cursor.seek(pos2);
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), static_cast<double>(pos2), 0.01);
    auto config2 = cursor.configuration();
    BOOST_CHECK_EQUAL(config2.shape(0), 2U);

    // Jump to middle
    const arc_length pos3 = p.length() * 0.5;
    cursor.seek(pos3);
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), static_cast<double>(pos3), 0.01);
    auto config3 = cursor.configuration();
    BOOST_CHECK_EQUAL(config3.shape(0), 2U);

    // Jump to near start
    const arc_length pos4 = p.length() * 0.05;
    cursor.seek(pos4);
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), static_cast<double>(pos4), 0.01);
    auto config4 = cursor.configuration();
    BOOST_CHECK_EQUAL(config4.shape(0), 2U);

    // Jump to near end again
    const arc_length pos5 = p.length() * 0.95;
    cursor.seek(pos5);
    BOOST_CHECK_CLOSE(static_cast<double>(cursor.position()), static_cast<double>(pos5), 0.01);
    auto config5 = cursor.configuration();
    BOOST_CHECK_EQUAL(config5.shape(0), 2U);

    // Verify tangent and curvature queries also work after jumps
    auto tangent = cursor.tangent();
    BOOST_CHECK_EQUAL(tangent.shape(0), 2U);
    auto curvature = cursor.curvature();
    BOOST_CHECK_EQUAL(curvature.shape(0), 2U);
}

BOOST_AUTO_TEST_CASE(dereference_operator_returns_segment_view) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Create path with circular blend to have different segment types
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
    path::options opts;
    opts.set_max_blend_deviation(0.1);
    const path p = path::create(waypoints, opts);

    // Should have 3 segments: linear, circular, linear
    BOOST_CHECK_EQUAL(p.size(), 3U);

    path::cursor cursor = p.create_cursor();

    // Start on first segment (linear)
    auto view1 = *cursor;
    BOOST_CHECK(view1.is<path::segment::linear>());
    BOOST_CHECK_EQUAL(view1.start(), arc_length{0.0});

    // Seek to middle - should be on circular blend
    cursor.seek(p.length() * 0.5);
    auto view2 = *cursor;
    BOOST_CHECK(view2.is<path::segment::circular>());

    // Seek to end - should be on last linear segment
    cursor.seek(p.length() * 0.95);
    auto view3 = *cursor;
    BOOST_CHECK(view3.is<path::segment::linear>());

    // Verify view gives access to segment bounds
    BOOST_CHECK(view3.start() < view3.end());
    BOOST_CHECK_EQUAL(view3.length(), view3.end() - view3.start());

    // Can use dereference result to query geometry
    auto config = view3.configuration(cursor.position());
    BOOST_CHECK_EQUAL(config.shape(0), 2U);
}

//
// Test that cursor advances to next segment when positioned exactly at segment boundary.
//
// This validates the semantics required for acceleration switching point detection:
// when we seek to a boundary, we should land in the segment that starts at that boundary,
// not the segment that ends there.
//
BOOST_AUTO_TEST_CASE(cursor_at_boundary_advances_to_next_segment) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex;

    // Create a simple path with three linear segments (no blends)
    // to ensure we have well-defined boundaries between distinct segments
    path::options opts;
    opts.set_max_deviation(0.0);  // No blending - hard corners

    const xmatrix<> waypoints = {
        {0.0, 0.0},  // Start
        {1.0, 0.0},  // First corner
        {1.0, 1.0},  // Second corner
        {2.0, 1.0}   // End
    };

    const path p = path::create(waypoints, opts);

    // Path should have 3 segments
    BOOST_REQUIRE_EQUAL(p.size(), 3);

    // Get segment boundaries by iterating
    std::vector<arc_length> boundaries;
    for (const auto& view : p) {
        boundaries.push_back(view.start());
        boundaries.push_back(view.end());
    }

    // Should have: segment0.start, segment0.end (=segment1.start), segment1.end (=segment2.start), segment2.end
    BOOST_REQUIRE_GE(boundaries.size(), 4);

    // Pick the boundary between segment 0 and segment 1
    const arc_length boundary_0_1 = boundaries[1];

    // Position cursor at boundary
    auto cursor = p.create_cursor(boundary_0_1);

    // Cursor should be in segment 1 (the segment that starts at boundary), not segment 0
    auto view_at_boundary = *cursor;

    // Verify cursor is in the segment that starts at this boundary
    BOOST_CHECK_EQUAL(view_at_boundary.start(), boundary_0_1);

    // Verify this is not the segment that ends at this boundary (segment 0)
    // by checking that the view doesn't extend backward from boundary
    BOOST_CHECK_GE(cursor.position(), view_at_boundary.start());
}

//
// Test that segment views accept queries at their start() point.
//
// This validates that we can query both segments adjacent to a boundary at the exact
// boundary arc length: the segment ending at the boundary can be queried at its end(),
// and the segment starting at the boundary can be queried at its start().
//
BOOST_AUTO_TEST_CASE(segment_view_accepts_query_at_start) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex;

    path::options opts;
    opts.set_max_deviation(0.0);

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints, opts);
    BOOST_REQUIRE_GE(p.size(), 2);

    // Get first segment
    auto it = p.begin();
    auto segment0 = *it;

    // Should be able to query at start
    const arc_length start = segment0.start();
    BOOST_CHECK_NO_THROW(segment0.configuration(start));
    BOOST_CHECK_NO_THROW(segment0.tangent(start));
    BOOST_CHECK_NO_THROW(segment0.curvature(start));
}

//
// Test that segment views accept queries at their end() point.
//
// The segment boundary semantics are [start, end) for containment checks, but
// end() should still be a valid query point for geometry (tangent, curvature)
// to support sampling both sides of a discontinuity.
//
BOOST_AUTO_TEST_CASE(segment_view_accepts_query_at_end) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex;

    path::options opts;
    opts.set_max_deviation(0.0);

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints, opts);
    BOOST_REQUIRE_GE(p.size(), 2);

    // Get first segment
    auto it = p.begin();
    auto segment0 = *it;

    // Should be able to query at end
    const arc_length end = segment0.end();
    BOOST_CHECK_NO_THROW(segment0.configuration(end));
    BOOST_CHECK_NO_THROW(segment0.tangent(end));
    BOOST_CHECK_NO_THROW(segment0.curvature(end));
}

//
// Test that both segments adjacent to a boundary return different geometry at the boundary.
//
// This is the key property for detecting discontinuities: if we query segment_before at
// boundary and segment_after at boundary (same arc length, different segments), we should
// get different tangent/curvature vectors, reflecting the geometric discontinuity.
//
BOOST_AUTO_TEST_CASE(adjacent_segments_differ_at_boundary) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex;

    path::options opts;
    opts.set_max_deviation(0.0);

    // Create path with a sharp corner to ensure geometric discontinuity
    const xmatrix<> waypoints = {
        {0.0, 0.0},  // Start - moving in +x direction
        {1.0, 0.0},  // Corner - 90 degree turn
        {1.0, 1.0}   // End - now moving in +y direction
    };

    const path p = path::create(waypoints, opts);
    BOOST_REQUIRE_EQUAL(p.size(), 2);

    // Get the boundary between the two segments
    auto it = p.begin();
    auto segment_before = *it;
    ++it;
    auto segment_after = *it;

    const arc_length boundary = segment_before.end();
    BOOST_CHECK_EQUAL(boundary, segment_after.start());  // Verify they meet

    // Query geometry from both segments at the exact boundary point
    auto tangent_before = segment_before.tangent(boundary);
    auto tangent_after = segment_after.tangent(boundary);

    // Tangents should differ at the corner (90 degree turn)
    // segment_before points in +x direction: [1, 0]
    // segment_after points in +y direction: [0, 1]
    BOOST_CHECK_CLOSE(tangent_before(0), 1.0, 1e-6);
    BOOST_CHECK_SMALL(tangent_before(1), 1e-6);

    BOOST_CHECK_SMALL(tangent_after(0), 1e-6);
    BOOST_CHECK_CLOSE(tangent_after(1), 1.0, 1e-6);

    // Verify they're actually different (dot product should be ~0 for perpendicular)
    const double dot_product = xt::sum(tangent_before * tangent_after)();
    BOOST_CHECK_SMALL(dot_product, 1e-6);
}

//
// Test cursor behavior with seek operations around boundaries.
//
// Validates that:
// 1. Seeking slightly before boundary keeps cursor in segment ending at boundary
// 2. Seeking exactly to boundary advances cursor to segment starting at boundary
// 3. Seeking slightly after boundary places cursor in segment starting at boundary
//
BOOST_AUTO_TEST_CASE(cursor_seek_behavior_around_boundary) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex;

    path::options opts;
    opts.set_max_deviation(0.0);

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints, opts);
    BOOST_REQUIRE_EQUAL(p.size(), 2);

    // Find boundary between segments
    auto it = p.begin();
    auto segment0 = *it;
    ++it;
    auto segment1 = *it;

    const arc_length boundary = segment0.end();
    BOOST_CHECK_EQUAL(boundary, segment1.start());

    constexpr double small_offset = 1e-10;

    // Test: slightly before boundary should be in segment 0
    {
        auto cursor = p.create_cursor(boundary - arc_length{small_offset});
        auto view = *cursor;
        BOOST_CHECK_EQUAL(view.start(), segment0.start());
        BOOST_CHECK_EQUAL(view.end(), segment0.end());
    }

    // Test: exactly at boundary should be in segment 1
    {
        auto cursor = p.create_cursor(boundary);
        auto view = *cursor;
        BOOST_CHECK_EQUAL(view.start(), segment1.start());
        BOOST_CHECK_EQUAL(view.end(), segment1.end());
    }

    // Test: slightly after boundary should be in segment 1
    {
        auto cursor = p.create_cursor(boundary + arc_length{small_offset});
        auto view = *cursor;
        BOOST_CHECK_EQUAL(view.start(), segment1.start());
        BOOST_CHECK_EQUAL(view.end(), segment1.end());
    }
}

//
// Test that we can safely sample geometry on both sides of a boundary using cursor.
//
// This test validates the complete workflow needed for switching point detection:
// 1. Position cursor somewhere in segment N
// 2. Get segment view and find its end (the boundary)
// 3. Query that view at the boundary (segment N's perspective)
// 4. Seek cursor to boundary (advances to segment N+1)
// 5. Get new segment view
// 6. Query new view at boundary (segment N+1's perspective)
// 7. Verify we got different geometry (discontinuity detected)
//
BOOST_AUTO_TEST_CASE(safe_boundary_sampling_workflow) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex;

    path::options opts;
    opts.set_max_deviation(0.0);

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints, opts);

    // Start cursor in first segment
    auto cursor = p.create_cursor(arc_length{0.5});

    // Get current segment view
    auto segment_before = *cursor;
    const arc_length boundary = segment_before.end();

    // Sample geometry from segment ending at boundary
    auto tangent_before = segment_before.tangent(boundary);
    auto curvature_before = segment_before.curvature(boundary);

    // Advance cursor to boundary (should move to next segment)
    cursor.seek(boundary);
    auto segment_after = *cursor;

    // Verify cursor advanced to next segment
    BOOST_CHECK_EQUAL(segment_after.start(), boundary);
    BOOST_CHECK_NE(segment_after.start(), segment_before.start());

    // Sample geometry from segment starting at boundary
    auto tangent_after = segment_after.tangent(boundary);
    auto curvature_after = segment_after.curvature(boundary);

    // Verify we got different geometry (detecting the discontinuity)
    const double tangent_diff = xt::norm_l2(tangent_before - tangent_after)();
    BOOST_CHECK_GT(tangent_diff, 0.1);  // Should differ significantly at 90 degree corner
}

//
// Test with circular blend segment to ensure boundary behavior works with both segment types.
//
BOOST_AUTO_TEST_CASE(cursor_boundary_behavior_with_circular_blends) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex;

    path::options opts;
    opts.set_max_deviation(0.1);  // Enable blending

    // Create path that will have blend
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};

    const path p = path::create(waypoints, opts);

    // With blending enabled, should have 3 segments: linear, circular, linear
    BOOST_REQUIRE_EQUAL(p.size(), 3);

    auto it = p.begin();
    auto segment0 = *it;  // Linear
    ++it;
    auto segment1 = *it;  // Circular blend
    ++it;
    auto segment2 = *it;  // Linear

    // Test boundary between linear and circular
    {
        const arc_length boundary = segment0.end();
        BOOST_CHECK_EQUAL(boundary, segment1.start());

        auto cursor = p.create_cursor(boundary);
        auto view = *cursor;

        // Should be in circular segment
        BOOST_CHECK(view.is<path::segment::circular>());
        BOOST_CHECK_EQUAL(view.start(), boundary);
    }

    // Test boundary between circular and linear
    {
        const arc_length boundary = segment1.end();
        BOOST_CHECK_EQUAL(boundary, segment2.start());

        auto cursor = p.create_cursor(boundary);
        auto view = *cursor;

        // Should be in second linear segment
        BOOST_CHECK(view.is<path::segment::linear>());
        BOOST_CHECK_EQUAL(view.start(), boundary);
    }
}

namespace {

// The filling and value-returning accessors run the same arithmetic over the same inputs,
// so anything short of bit equality means one of them has been reimplemented independently.
void check_exactly_equal(const xvector<>& value, std::span<const double> filled) {
    BOOST_REQUIRE_EQUAL(value.size(), filled.size());

    for (std::size_t i = 0; i < filled.size(); ++i) {
        BOOST_CHECK_EQUAL(value(i), filled[i]);
    }
}

// Two linear runs joined by a circular blend, so both segment kinds get exercised.
viam::trajex::totg::path make_blended_path() {
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 0.0}};
    return viam::trajex::totg::path::create(waypoints);
}

viam::trajex::arc_length fraction_of(viam::trajex::arc_length length, double f) {
    return viam::trajex::arc_length{static_cast<double>(length) * f};
}

}  // namespace

BOOST_AUTO_TEST_CASE(fill_accessors_match_value_accessors) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    path::cursor c = p.create_cursor();

    std::vector<double> filled(p.dof());

    for (int step = 0; step <= 20; ++step) {
        c.seek(fraction_of(p.length(), static_cast<double>(step) / 20.0));

        c.configuration(filled);
        check_exactly_equal(c.configuration(), filled);

        c.tangent(filled);
        check_exactly_equal(c.tangent(), filled);

        c.curvature(filled);
        check_exactly_equal(c.curvature(), filled);
    }
}

BOOST_AUTO_TEST_CASE(fill_accessors_reject_wrong_sized_span) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    const path::cursor c = p.create_cursor();

    std::vector<double> too_small(p.dof() - 1);
    std::vector<double> too_large(p.dof() + 1);

    BOOST_CHECK_THROW(c.configuration(too_small), std::invalid_argument);
    BOOST_CHECK_THROW(c.tangent(too_small), std::invalid_argument);
    BOOST_CHECK_THROW(c.curvature(too_small), std::invalid_argument);

    BOOST_CHECK_THROW(c.configuration(too_large), std::invalid_argument);
    BOOST_CHECK_THROW(c.tangent(too_large), std::invalid_argument);
    BOOST_CHECK_THROW(c.curvature(too_large), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(fill_accessors_throw_at_sentinel) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const path p = make_blended_path();
    path::cursor c = p.create_cursor();
    c.seek(p.length() + arc_length{1.0});

    BOOST_REQUIRE(c == c.end());

    std::vector<double> filled(p.dof());

    BOOST_CHECK_THROW(c.configuration(filled), std::out_of_range);
    BOOST_CHECK_THROW(c.tangent(filled), std::out_of_range);
    BOOST_CHECK_THROW(c.curvature(filled), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(rich_matches_plain_cursor_at_many_positions) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    path::cursor plain = p.create_cursor();
    path::cursor::rich r = p.create_cursor().enrich();

    for (int step = 0; step <= 20; ++step) {
        const auto s = fraction_of(p.length(), static_cast<double>(step) / 20.0);

        plain.seek(s);
        r.seek(s);

        BOOST_CHECK_EQUAL(r.position(), plain.position());

        const auto& configuration = r.configuration();
        const auto& tangent = r.tangent();
        const auto& curvature = r.curvature();

        check_exactly_equal(plain.configuration(), {configuration.data(), configuration.size()});
        check_exactly_equal(plain.tangent(), {tangent.data(), tangent.size()});
        check_exactly_equal(plain.curvature(), {curvature.data(), curvature.size()});
    }
}

BOOST_AUTO_TEST_CASE(rich_storage_address_stable_across_seek) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    path::cursor::rich r = p.create_cursor().enrich();

    // Invalidation must discard the cached *values* without releasing the storage holding
    // them. Were the fill ever rewritten as an assignment from a returned array, the
    // move-assignment would swap buffers and this would fail -- along with the guarantee
    // that a reference handed to a caller keeps referring to live storage.
    const double* const configuration_storage = r.configuration().data();
    const double* const tangent_storage = r.tangent().data();
    const double* const curvature_storage = r.curvature().data();

    for (int step = 1; step <= 5; ++step) {
        r.seek(fraction_of(p.length(), static_cast<double>(step) / 5.0));

        BOOST_CHECK_EQUAL(r.configuration().data(), configuration_storage);
        BOOST_CHECK_EQUAL(r.tangent().data(), tangent_storage);
        BOOST_CHECK_EQUAL(r.curvature().data(), curvature_storage);
    }
}

BOOST_AUTO_TEST_CASE(rich_accessors_lazy_in_any_order) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    const auto s = fraction_of(p.length(), 0.4);

    path::cursor plain = p.create_cursor();
    plain.seek(s);

    // Each component fills on first use, so the order they are first touched after a move
    // must not matter. Take them in reverse and compare against a plain cursor.
    path::cursor::rich r = p.create_cursor().enrich();
    r.seek(s);

    const auto& curvature = r.curvature();
    check_exactly_equal(plain.curvature(), {curvature.data(), curvature.size()});

    const auto& tangent = r.tangent();
    check_exactly_equal(plain.tangent(), {tangent.data(), tangent.size()});

    const auto& configuration = r.configuration();
    check_exactly_equal(plain.configuration(), {configuration.data(), configuration.size()});
}

BOOST_AUTO_TEST_CASE(rich_reference_is_window_not_snapshot) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    const auto start = fraction_of(p.length(), 0.25);
    const auto moved = fraction_of(p.length(), 0.75);

    path::cursor::rich r = p.create_cursor().enrich();
    r.seek(start);

    const auto& configuration = r.configuration();
    const std::vector<double> at_start(configuration.begin(), configuration.end());

    // Seeking clears the validity bits but leaves the storage holding the old values, so a
    // reference taken before the move still reads the old position.
    r.seek(moved);
    check_exactly_equal(xvector<>{configuration}, at_start);

    // Asking again refills the same storage in place, at which point the reference taken
    // before the move begins reporting the new position. This is the documented hazard; it
    // is pinned here so that changing it cannot pass silently.
    const auto& refilled = r.configuration();
    BOOST_CHECK_EQUAL(&refilled, &configuration);

    path::cursor plain = p.create_cursor();
    plain.seek(moved);
    check_exactly_equal(plain.configuration(), {configuration.data(), configuration.size()});
}

BOOST_AUTO_TEST_CASE(rich_plain_is_independent_copy) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    const auto start = fraction_of(p.length(), 0.3);

    path::cursor::rich r = p.create_cursor().enrich();
    r.seek(start);

    path::cursor extracted = r.plain();
    BOOST_CHECK_EQUAL(extracted.position(), r.position());
    BOOST_CHECK_EQUAL(&extracted.path(), &p);

    // Moving the extracted cursor must not drag the rich one along with it.
    extracted.seek(fraction_of(p.length(), 0.9));
    BOOST_CHECK_EQUAL(r.position(), start);
    BOOST_CHECK(extracted.position() != r.position());
}

BOOST_AUTO_TEST_CASE(rich_enrich_leaves_source_unchanged) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    const auto start = fraction_of(p.length(), 0.6);

    path::cursor source = p.create_cursor();
    source.seek(start);

    path::cursor::rich r = source.enrich();
    BOOST_CHECK_EQUAL(r.position(), start);

    r.seek(fraction_of(p.length(), 0.1));
    BOOST_CHECK_EQUAL(source.position(), start);
}

BOOST_AUTO_TEST_CASE(rich_sentinel_comparison) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const path p = make_blended_path();
    path::cursor::rich r = p.create_cursor().enrich();

    BOOST_CHECK(r != r.end());
    BOOST_CHECK(r != end(r));

    r.seek(p.length() + arc_length{1.0});

    BOOST_CHECK(r == r.end());
    BOOST_CHECK(r == end(r));
    BOOST_CHECK(std::default_sentinel == r);
}

BOOST_AUTO_TEST_CASE(rich_failed_query_is_not_cached) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const path p = make_blended_path();
    const auto reachable = fraction_of(p.length(), 0.5);

    path::cursor::rich r = p.create_cursor().enrich();
    r.seek(p.length() + arc_length{1.0});

    BOOST_CHECK_THROW(static_cast<void>(r.configuration()), std::out_of_range);

    // A throwing fill must leave the validity bit clear. Were it set, this query would hand
    // back whatever the storage happened to contain instead of computing the position.
    r.seek(reachable);

    path::cursor plain = p.create_cursor();
    plain.seek(reachable);

    const auto& configuration = r.configuration();
    check_exactly_equal(plain.configuration(), {configuration.data(), configuration.size()});
}

BOOST_AUTO_TEST_CASE(rich_copy_has_independent_storage) {
    using namespace viam::trajex::totg;

    const path p = make_blended_path();
    const auto start = fraction_of(p.length(), 0.45);

    path::cursor::rich original = p.create_cursor().enrich();
    original.seek(start);
    const auto& original_configuration = original.configuration();

    path::cursor::rich copy = original;
    const auto& copy_configuration = copy.configuration();

    BOOST_CHECK(copy_configuration.data() != original_configuration.data());
    check_exactly_equal(xvector<>{copy_configuration}, {original_configuration.data(), original_configuration.size()});

    // Moving the copy must not disturb the original's cached values.
    copy.seek(fraction_of(p.length(), 0.05));
    static_cast<void>(copy.configuration());

    path::cursor plain = p.create_cursor();
    plain.seek(start);
    check_exactly_equal(plain.configuration(), {original_configuration.data(), original_configuration.size()});
}

BOOST_AUTO_TEST_SUITE_END()
