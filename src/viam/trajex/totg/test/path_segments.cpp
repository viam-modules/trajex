// Path segment tests: linear and circular segment types
// Extracted from test.cpp lines 1755-2060

#include <cmath>
#include <limits>
#include <numbers>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/types/angles.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/xt.hpp>

#if __has_include(<xtensor/reducers/xnorm.hpp>)
#include <xtensor/reducers/xnorm.hpp>
#else
#include <xtensor/xnorm.hpp>
#endif

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(segment_tests)

using viam::trajex::degrees_to_radians;
using viam::trajex::xmatrix;
using viam::trajex::xvector;
BOOST_AUTO_TEST_CASE(linear_constructor) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Should be able to create linear segments
    const xvector<> start = {1.0, 2.0, 3.0};
    const xvector<> end = {4.0, 5.0, 6.0};
    BOOST_CHECK_NO_THROW((path::segment::linear{start, end}));
}

BOOST_AUTO_TEST_CASE(circular_constructor) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Should be able to create circular segments
    const xvector<> apex = {0.0, 0.0, 0.0};
    const xvector<> t = {1.0, 0.0, 0.0};
    const xvector<> n = {0.0, 1.0, 0.0};
    BOOST_CHECK_NO_THROW((path::segment::circular{apex, t, n, 1.0, std::numbers::pi / 2}));

    // The arc is parameterized by arc length only if t and n are orthonormal, and TOTG relies on
    // its tangent having unit length. Arcs built by path::create meet that to within rounding, so
    // the constructor allows only rounding, and a basis off by more is the caller's error. A
    // tangent too long by 1e-12, or a normal leaning toward it by 1e-12, would pass any tolerance
    // one might choose, but is far beyond rounding at three coordinates.
    const xvector<> long_t = {1.0 + 1e-12, 0.0, 0.0};
    BOOST_CHECK_THROW((path::segment::circular{apex, long_t, n, 1.0, std::numbers::pi / 2}), std::invalid_argument);
    const xvector<> leaning_n = {1e-12, 1.0, 0.0};
    BOOST_CHECK_THROW((path::segment::circular{apex, t, leaning_n, 1.0, std::numbers::pi / 2}), std::invalid_argument);

    // Non-finite geometry is rejected, including NaN, which every comparison against a bound lets through.
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    constexpr auto inf = std::numeric_limits<double>::infinity();
    const xvector<> nan_apex = {nan, 0.0, 0.0};
    BOOST_CHECK_THROW((path::segment::circular{nan_apex, t, n, 1.0, std::numbers::pi / 2}), std::invalid_argument);
    const xvector<> nan_t = {nan, 0.0, 0.0};
    BOOST_CHECK_THROW((path::segment::circular{apex, nan_t, n, 1.0, std::numbers::pi / 2}), std::invalid_argument);
    const xvector<> inf_n = {0.0, inf, 0.0};
    BOOST_CHECK_THROW((path::segment::circular{apex, t, inf_n, 1.0, std::numbers::pi / 2}), std::invalid_argument);
    BOOST_CHECK_THROW((path::segment::circular{apex, t, n, nan, std::numbers::pi / 2}), std::invalid_argument);
    BOOST_CHECK_THROW((path::segment::circular{apex, t, n, 1.0, inf}), std::invalid_argument);

    // The radius must be positive, and a blend must turn by less than pi, which is what leaves each
    // joint at most one extremum on it. Both bounds are open.
    BOOST_CHECK_THROW((path::segment::circular{apex, t, n, 0.0, std::numbers::pi / 2}), std::invalid_argument);
    BOOST_CHECK_THROW((path::segment::circular{apex, t, n, -1.0, std::numbers::pi / 2}), std::invalid_argument);
    BOOST_CHECK_THROW((path::segment::circular{apex, t, n, 1.0, 0.0}), std::invalid_argument);
    BOOST_CHECK_THROW((path::segment::circular{apex, t, n, 1.0, -std::numbers::pi / 2}), std::invalid_argument);
    BOOST_CHECK_THROW((path::segment::circular{apex, t, n, 1.0, std::numbers::pi}), std::invalid_argument);
    BOOST_CHECK_THROW((path::segment::circular{apex, t, n, 1.0, 2 * std::numbers::pi}), std::invalid_argument);
    BOOST_CHECK_NO_THROW((path::segment::circular{apex, t, n, std::numeric_limits<double>::denorm_min(), std::numbers::pi / 2}));
    BOOST_CHECK_NO_THROW((path::segment::circular{apex, t, n, 1.0, std::numeric_limits<double>::denorm_min()}));
    BOOST_CHECK_NO_THROW((path::segment::circular{apex, t, n, 1.0, std::nextafter(std::numbers::pi, 0.0)}));
}

BOOST_AUTO_TEST_CASE(segment_view_type_check) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Create linear segment
    const path::segment::linear linear_data{xvector<>{1.0, 2.0, 3.0}, xvector<>{4.0, 5.0, 6.0}};
    const path::segment linear_seg{linear_data};
    const path::segment::view linear_view{linear_seg, arc_length{0.0}, linear_data.length};

    BOOST_CHECK(linear_view.is<path::segment::linear>());
    BOOST_CHECK(!linear_view.is<path::segment::circular>());

    // Create circular segment
    const path::segment::circular circular_data{
        xvector<>{0.0, 0.0, 0.0}, xvector<>{1.0, 0.0, 0.0}, xvector<>{0.0, 1.0, 0.0}, 1.0, std::numbers::pi / 2};
    const path::segment circular_seg{circular_data};
    const path::segment::view circular_view{circular_seg, arc_length{0.0}, arc_length{std::numbers::pi / 2}};

    BOOST_CHECK(circular_view.is<path::segment::circular>());
    BOOST_CHECK(!circular_view.is<path::segment::linear>());
}

BOOST_AUTO_TEST_CASE(segment_view_visit) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Create linear segment
    const path::segment::linear linear_data{xvector<>{1.0, 2.0, 3.0}, xvector<>{4.0, 5.0, 6.0}};
    const path::segment linear_seg{linear_data};
    const path::segment::view linear_view{linear_seg, arc_length{0.0}, linear_data.length};

    // Visit linear segment - direction (3,3,3) normalizes to (1/√3, 1/√3, 1/√3)
    bool visited_linear = false;
    linear_view.visit([&visited_linear](const auto& seg_data) {
        using T = std::decay_t<decltype(seg_data)>;
        if constexpr (std::is_same_v<T, path::segment::linear>) {
            visited_linear = true;
            BOOST_CHECK_CLOSE(seg_data.unit_direction(0), 1.0 / std::sqrt(3.0), 1e-6);
            BOOST_CHECK_CLOSE(seg_data.unit_direction(1), 1.0 / std::sqrt(3.0), 1e-6);
            BOOST_CHECK_CLOSE(seg_data.unit_direction(2), 1.0 / std::sqrt(3.0), 1e-6);
        }
    });
    BOOST_CHECK(visited_linear);

    // Create circular segment
    const path::segment::circular circular_data{
        xvector<>{0.0, 0.0, 0.0}, xvector<>{1.0, 0.0, 0.0}, xvector<>{0.0, 1.0, 0.0}, 2.5, std::numbers::pi / 2};
    const path::segment circular_seg{circular_data};
    const path::segment::view circular_view{circular_seg, arc_length{0.0}, arc_length{std::numbers::pi / 2}};

    // Visit circular segment
    bool visited_circular = false;
    circular_view.visit([&visited_circular](const auto& seg_data) {
        using T = std::decay_t<decltype(seg_data)>;
        if constexpr (std::is_same_v<T, path::segment::circular>) {
            visited_circular = true;
            BOOST_CHECK_CLOSE(seg_data.radius, 2.5, 1e-6);
        }
    });
    BOOST_CHECK(visited_circular);
}

BOOST_AUTO_TEST_CASE(linear_segment_configuration) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Linear segment from (0,0) to (3,4) - length = 5, direction = (0.6, 0.8)
    const path::segment::linear data{xvector<>{0.0, 0.0}, xvector<>{3.0, 4.0}};
    const path::segment seg{data};
    const path::segment::view view{seg, arc_length{0.0}, data.length};

    // At start (s=0): should be (0,0)
    auto config_start = view.configuration(arc_length{0.0});
    BOOST_CHECK_CLOSE(config_start(0), 0.0, 1e-6);
    BOOST_CHECK_CLOSE(config_start(1), 0.0, 1e-6);

    // At middle (s=2.5): should be (1.5, 2.0)
    auto config_mid = view.configuration(arc_length{2.5});
    BOOST_CHECK_CLOSE(config_mid(0), 1.5, 1e-6);
    BOOST_CHECK_CLOSE(config_mid(1), 2.0, 1e-6);

    // At end (s=5): should be (3,4)
    auto config_end = view.configuration(arc_length{5.0});
    BOOST_CHECK_CLOSE(config_end(0), 3.0, 1e-6);
    BOOST_CHECK_CLOSE(config_end(1), 4.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(linear_segment_tangent) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Linear segment from (0,0) to (3,4) - length = 5
    // Unit tangent should be (0.6, 0.8)
    const path::segment::linear data{xvector<>{0.0, 0.0}, xvector<>{3.0, 4.0}};
    const path::segment seg{data};
    const path::segment::view view{seg, arc_length{0.0}, data.length};

    // Tangent should be constant along the segment
    auto tangent_start = view.tangent(arc_length{0.0});
    BOOST_CHECK_CLOSE(tangent_start(0), 0.6, 1e-6);
    BOOST_CHECK_CLOSE(tangent_start(1), 0.8, 1e-6);

    auto tangent_mid = view.tangent(arc_length{2.5});
    BOOST_CHECK_CLOSE(tangent_mid(0), 0.6, 1e-6);
    BOOST_CHECK_CLOSE(tangent_mid(1), 0.8, 1e-6);

    auto tangent_end = view.tangent(arc_length{5.0});
    BOOST_CHECK_CLOSE(tangent_end(0), 0.6, 1e-6);
    BOOST_CHECK_CLOSE(tangent_end(1), 0.8, 1e-6);
}

BOOST_AUTO_TEST_CASE(linear_segment_curvature) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Linear segments have zero curvature
    const path::segment::linear data{xvector<>{0.0, 0.0}, xvector<>{3.0, 4.0}};
    const path::segment seg{data};
    const path::segment::view view{seg, arc_length{0.0}, data.length};

    auto curvature = view.curvature(arc_length{2.5});
    BOOST_CHECK_SMALL(curvature(0), 1e-10);
    BOOST_CHECK_SMALL(curvature(1), 1e-10);
}

BOOST_AUTO_TEST_CASE(circular_segment_configuration) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Quarter circle in xy plane, radius 1, centered at origin, arc from (1,0) to (0,1)
    const double h = std::sqrt(0.5);
    const path::segment::circular data{
        xvector<>{h, h},          // apex
        xvector<>{-h, h},         // t
        xvector<>{-h, -h},        // n
        1.0,                      // radius
        degrees_to_radians(90.0)  // angle_rads (90 degrees)
    };
    const path::segment seg{data};
    const path::segment::view view{seg, arc_length{0.0}, arc_length{degrees_to_radians(90.0)}};

    // At start (angle=0): should be (1, 0)
    auto config_start = view.configuration(arc_length{0.0});
    BOOST_CHECK_CLOSE(config_start(0), 1.0, 1e-6);
    BOOST_CHECK_SMALL(config_start(1), 1e-10);  // Near zero, use SMALL not CLOSE

    // At middle (angle=π/4): should be (√2/2, √2/2)
    auto config_mid = view.configuration(arc_length{degrees_to_radians(45.0)});
    BOOST_CHECK_CLOSE(config_mid(0), std::sqrt(2.0) / 2.0, 1e-6);
    BOOST_CHECK_CLOSE(config_mid(1), std::sqrt(2.0) / 2.0, 1e-6);

    // At end (angle=π/2): should be (0, 1)
    auto config_end = view.configuration(arc_length{degrees_to_radians(90.0)});
    BOOST_CHECK_SMALL(config_end(0), 1e-10);  // Near zero, use SMALL not CLOSE
    BOOST_CHECK_CLOSE(config_end(1), 1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(circular_segment_tangent) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Quarter circle centered at origin from (1,0) to (0,1), tangent should be perpendicular to radius
    const double h = std::sqrt(0.5);
    const path::segment::circular data{
        xvector<>{h, h},          // apex
        xvector<>{-h, h},         // t
        xvector<>{-h, -h},        // n
        1.0,                      // radius
        degrees_to_radians(90.0)  // angle_rads
    };
    const path::segment seg{data};
    const path::segment::view view{seg, arc_length{0.0}, arc_length{degrees_to_radians(90.0)}};

    // At start (angle=0): tangent should be (0, 1)
    auto tangent_start = view.tangent(arc_length{0.0});
    BOOST_CHECK_SMALL(tangent_start(0), 1e-10);  // Near zero
    BOOST_CHECK_CLOSE(tangent_start(1), 1.0, 1e-6);

    // At end (angle=π/2): tangent should be (-1, 0)
    auto tangent_end = view.tangent(arc_length{degrees_to_radians(90.0)});
    BOOST_CHECK_CLOSE(tangent_end(0), -1.0, 1e-6);
    BOOST_CHECK_SMALL(tangent_end(1), 1e-10);  // Near zero
}

BOOST_AUTO_TEST_CASE(circular_segment_curvature) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Curvature vector points toward center with magnitude 1/radius. Quarter circle centered at
    // origin from (2,0) to (0,2).
    const double h = std::sqrt(0.5);
    const path::segment::circular data{
        xvector<>{2.0 * h, 2.0 * h},  // apex
        xvector<>{-h, h},             // t
        xvector<>{-h, -h},            // n
        2.0,                          // radius = 2
        degrees_to_radians(90.0)      // angle_rads
    };
    const path::segment seg{data};
    const path::segment::view view{seg, arc_length{0.0}, arc_length{degrees_to_radians(180.0)}};

    // At start (angle=0): curvature should point from (2,0) to (0,0), magnitude 1/2
    auto curvature_start = view.curvature(arc_length{0.0});
    BOOST_CHECK_CLOSE(curvature_start(0), -0.5, 1e-6);  // -1/radius
    BOOST_CHECK_SMALL(curvature_start(1), 1e-10);       // Near zero, use SMALL not CLOSE

    // Magnitude should be 1/radius = 0.5
    const double mag = xt::norm_l2(curvature_start)();
    BOOST_CHECK_CLOSE(mag, 0.5, 1e-6);
}

BOOST_AUTO_TEST_CASE(coordinate_extremum) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // A linear segment's tangent is constant, so no coordinate has an extremum on it.
    const path::segment::linear linear_data{xvector<>{0.0, 0.0, 0.0}, xvector<>{3.0, 4.0, 0.0}};
    const path::segment linear_seg{linear_data};
    const path::segment::view linear_view{linear_seg, arc_length{1.0}, arc_length{1.0} + linear_data.length};
    BOOST_CHECK(!linear_view.coordinate_extremum(0));
    BOOST_CHECK_THROW(linear_view.coordinate_extremum(3), std::out_of_range);

    // These are quarter-turn arcs of unit radius placed at s = 1 on their path, with the frame rotated
    // so that the first coordinate's tangent component is sin(phi - root), which puts its extremum at
    // phi = root, and the second's is cos(phi - root), with its root a quarter turn away. The third
    // coordinate never moves.
    const double quarter = degrees_to_radians(90.0);
    const arc_length start{1.0};
    const auto arc_with_root_at = [&](double root) {
        return path::segment{path::segment::circular{xvector<>{0.0, 0.0, 0.0},
                                                     xvector<>{-std::sin(root), std::cos(root), 0.0},
                                                     xvector<>{std::cos(root), std::sin(root), 0.0},
                                                     1.0,
                                                     quarter}};
    };
    const auto expect_root_at = [&](const path::segment::view& view, std::size_t joint, double root) {
        const auto extremum = view.coordinate_extremum(joint);
        BOOST_REQUIRE(extremum);
        BOOST_CHECK_CLOSE(static_cast<double>(*extremum), static_cast<double>(start) + root + (quarter / 2.0), 1e-12);
        return *extremum;
    };

    // The root is at the apex. The second coordinate's root would be a quarter turn either side,
    // which is not within the interval roots are taken from.
    const auto apex_seg = arc_with_root_at(0.0);
    const path::segment::view apex_view{apex_seg, start, start + arc_length{quarter}};
    BOOST_CHECK_SMALL(apex_view.tangent(expect_root_at(apex_view, 0, 0.0))(0), 1e-15);
    BOOST_CHECK(!apex_view.coordinate_extremum(1));
    BOOST_CHECK(!apex_view.coordinate_extremum(2));
    BOOST_CHECK_THROW(apex_view.coordinate_extremum(3), std::out_of_range);

    // A root near the end is on the arc. The second coordinate's root, a quarter turn earlier, is
    // off the arc, before its start, and is reported there all the same.
    constexpr double k_margin = 0.01;
    const auto near_end_seg = arc_with_root_at((quarter / 2.0) - k_margin);
    const path::segment::view near_end_view{near_end_seg, start, start + arc_length{quarter}};
    BOOST_CHECK_SMALL(near_end_view.tangent(expect_root_at(near_end_view, 0, (quarter / 2.0) - k_margin))(0), 1e-15);
    BOOST_CHECK(expect_root_at(near_end_view, 1, -(quarter / 2.0) - k_margin) < near_end_view.start());

    // This is the mirror image: the root is near the start, and the second coordinate's is past the end.
    const auto near_start_seg = arc_with_root_at(k_margin - (quarter / 2.0));
    const path::segment::view near_start_view{near_start_seg, start, start + arc_length{quarter}};
    BOOST_CHECK_SMALL(near_start_view.tangent(expect_root_at(near_start_view, 0, k_margin - (quarter / 2.0)))(0), 1e-15);
    BOOST_CHECK(expect_root_at(near_start_view, 1, (quarter / 2.0) + k_margin) > near_start_view.end());
}

BOOST_AUTO_TEST_CASE(nearest) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const auto nearest_to = [](const path::segment::view& view, const xvector<>& point) {
        return static_cast<double>(view.nearest(std::span<const double>{point.data(), point.size()}));
    };

    // The line runs from (0,0,0) to (3,4,0) and is placed at s = 1. A point off the line projects
    // onto it, and points beyond either end are nearest that end.
    const path::segment::linear linear_data{xvector<>{0.0, 0.0, 0.0}, xvector<>{3.0, 4.0, 0.0}};
    const path::segment linear_seg{linear_data};
    const path::segment::view linear_view{linear_seg, arc_length{1.0}, arc_length{1.0} + linear_data.length};
    BOOST_CHECK_CLOSE(nearest_to(linear_view, xvector<>{1.5, 2.0, 5.0}), 3.5, 1e-12);
    BOOST_CHECK_EQUAL(nearest_to(linear_view, xvector<>{-1.0, -1.0, 0.0}), 1.0);
    BOOST_CHECK_EQUAL(nearest_to(linear_view, xvector<>{10.0, 10.0, 0.0}), static_cast<double>(linear_view.end()));

    // The arc is a quarter circle of unit radius about the origin, from angle -pi/4 to pi/4, placed
    // at s = 1. A point off the circle is nearest the arc at its own angle, a point past an end is
    // nearest that end, and so is a point on the far side of the center.
    const double quarter = std::numbers::pi / 2;
    const path::segment arc_seg{path::segment::circular{xvector<>{1.0, 0.0}, xvector<>{0.0, 1.0}, xvector<>{-1.0, 0.0}, 1.0, quarter}};
    const path::segment::view arc_view{arc_seg, arc_length{1.0}, arc_length{1.0 + quarter}};
    BOOST_CHECK_CLOSE(nearest_to(arc_view, xvector<>{2.0, 0.0}), 1.0 + (quarter / 2.0), 1e-12);
    BOOST_CHECK_CLOSE(nearest_to(arc_view, xvector<>{3.0 * std::cos(0.3), 3.0 * std::sin(0.3)}), 1.0 + (quarter / 2.0) + 0.3, 1e-12);
    BOOST_CHECK_EQUAL(nearest_to(arc_view, xvector<>{std::cos(1.2), std::sin(1.2)}), static_cast<double>(arc_view.end()));
    BOOST_CHECK_EQUAL(nearest_to(arc_view, xvector<>{std::cos(-1.2), std::sin(-1.2)}), static_cast<double>(arc_view.start()));
    BOOST_CHECK_EQUAL(nearest_to(arc_view, xvector<>{-1.0, 0.1}), static_cast<double>(arc_view.end()));

    // The configuration must have the path's degrees of freedom.
    BOOST_CHECK_THROW(nearest_to(arc_view, xvector<>{1.0, 0.0, 0.0}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(has_interior) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // A segment has an interior when a representable arc length lies strictly between its ends: two
    // ULPs apart leaves exactly one, one ULP apart leaves none.
    const path::segment::linear data{xvector<>{0.0, 0.0}, xvector<>{3.0, 4.0}};
    const path::segment seg{data};
    const double start = 1.0;
    const double one_ulp = std::nextafter(start, 2.0);
    const double two_ulps = std::nextafter(one_ulp, 2.0);
    BOOST_CHECK((path::segment::view{seg, arc_length{start}, arc_length{two_ulps}}.has_interior()));
    BOOST_CHECK(!(path::segment::view{seg, arc_length{start}, arc_length{one_ulp}}.has_interior()));
    BOOST_CHECK(!(path::segment::view{seg, arc_length{start}, arc_length{start}}.has_interior()));
    BOOST_CHECK((path::segment::view{seg, arc_length{0.0}, data.length}.has_interior()));
}

BOOST_AUTO_TEST_CASE(segment_view_bounds_checking) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    const path::segment::linear data{xvector<>{0.0, 0.0}, xvector<>{3.0, 4.0}};
    const path::segment seg{data};
    const path::segment::view view{seg, arc_length{0.0}, data.length};

    // Negative arc_length should throw
    BOOST_CHECK_THROW(view.configuration(arc_length{-1.0}), std::out_of_range);
    BOOST_CHECK_THROW(view.tangent(arc_length{-0.5}), std::out_of_range);
    BOOST_CHECK_THROW(view.curvature(arc_length{-0.1}), std::out_of_range);

    // Arc_length beyond segment end should throw
    BOOST_CHECK_THROW(view.configuration(arc_length{5.1}), std::out_of_range);
    BOOST_CHECK_THROW(view.tangent(arc_length{6.0}), std::out_of_range);
    BOOST_CHECK_THROW(view.curvature(arc_length{10.0}), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(path_container_semantics) {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_length;

    // Create a path with 3 waypoints -> 2 linear segments
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
    auto p = path::create(waypoints);

    // Test size and empty
    BOOST_CHECK_EQUAL(p.size(), 2U);
    BOOST_CHECK(!p.empty());

    // Test iteration - iterator yields segment::view
    size_t count = 0;
    for (const auto& view : p) {
        // Each iteration yields a segment::view
        BOOST_CHECK(view.is<path::segment::linear>());
        BOOST_CHECK(view.start() >= arc_length{0.0});
        BOOST_CHECK(view.length() > arc_length{0.0});
        ++count;
    }
    BOOST_CHECK_EQUAL(count, 2U);

    // Test iterator accessors
    auto it = p.begin();
    BOOST_CHECK(it != p.end());

    auto view1 = *it;
    BOOST_CHECK(view1.is<path::segment::linear>());
    BOOST_CHECK_EQUAL(static_cast<double>(view1.start()), 0.0);
    BOOST_CHECK_EQUAL(static_cast<double>(view1.end()), 1.0);

    // Can query the view immediately
    auto config = view1.configuration(arc_length{0.5});
    BOOST_CHECK_EQUAL(config.shape(0), 2U);

    ++it;
    BOOST_CHECK(it != p.end());

    auto view2 = *it;
    BOOST_CHECK_EQUAL(static_cast<double>(view2.start()), 1.0);
    BOOST_CHECK_EQUAL(static_cast<double>(view2.end()), 2.0);  // second segment length is 1.0

    ++it;
    BOOST_CHECK(it == p.end());

    // Test const_iterator
    auto cit = p.cbegin();
    BOOST_CHECK(cit != p.cend());
    ++cit;
    BOOST_CHECK(cit != p.cend());
    ++cit;
    BOOST_CHECK(cit == p.cend());
}

BOOST_AUTO_TEST_SUITE_END()
