#include <viam/trajex/totg/compat/path.hpp>
#include <viam/trajex/totg/compat/trajectory.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/types/arc_length.hpp>

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
#include <third_party/trajectories/Path.h>
#include <third_party/trajectories/Trajectory.h>
#endif

// We qualify names throughout rather than bringing them in with using-declarations, because the
// reference headers above define global `Path` and `Trajectory`, which would make unqualified
// names ambiguous.
namespace compat = viam::trajex::totg::compat;

namespace {

Eigen::VectorXd vec(const std::vector<double>& values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    std::copy(values.begin(), values.end(), result.data());
    return result;
}

std::list<Eigen::VectorXd> waypoints(const std::vector<std::vector<double>>& points) {
    std::list<Eigen::VectorXd> result;
    for (const auto& point : points) {
        result.push_back(vec(point));
    }
    return result;
}

bool close(const Eigen::VectorXd& actual, const Eigen::VectorXd& expected, double tolerance) {
    return actual.size() == expected.size() && (actual - expected).cwiseAbs().maxCoeff() <= tolerance;
}

template <typename Exception>
bool holds(const std::exception_ptr& error) {
    if (!error) {
        return false;
    }
    try {
        std::rethrow_exception(error);
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(compat_path)

BOOST_AUTO_TEST_CASE(polyline_geometry) {
    const compat::Path path(waypoints({{0.0, 0.0}, {3.0, 0.0}, {3.0, 4.0}}));
    BOOST_REQUIRE(!path.getError());
    BOOST_CHECK_CLOSE(path.getLength(), 7.0, 1e-9);
    BOOST_CHECK(close(path.getConfig(0.0), vec({0.0, 0.0}), 1e-12));
    BOOST_CHECK(close(path.getConfig(3.0), vec({3.0, 0.0}), 1e-12));
    BOOST_CHECK(close(path.getConfig(7.0), vec({3.0, 4.0}), 1e-12));
    BOOST_CHECK(close(path.getTangent(1.0), vec({1.0, 0.0}), 1e-12));
    BOOST_CHECK(close(path.getTangent(5.0), vec({0.0, 1.0}), 1e-12));
    BOOST_CHECK(close(path.getCurvature(1.0), vec({0.0, 0.0}), 1e-12));
}

BOOST_AUTO_TEST_CASE(arc_length_is_clamped) {
    const compat::Path path(waypoints({{0.0}, {2.0}}));
    BOOST_CHECK(close(path.getConfig(-1.0), vec({0.0}), 1e-12));
    BOOST_CHECK(close(path.getConfig(5.0), vec({2.0}), 1e-12));
    BOOST_CHECK_EQUAL(path.getConfig(std::numeric_limits<double>::quiet_NaN()).size(), 0);
}

// The middle waypoint is within the deviation of the line through its neighbours, so trajex's
// collinear merging would drop it and straighten the path. Through compat, maxDeviation only
// blends, so the path still bends toward the middle waypoint. Building the same path natively,
// with merging, checks that the test can tell the two apart.
BOOST_AUTO_TEST_CASE(max_deviation_blends_without_merging) {
    const auto points = waypoints({{0.0, 0.0}, {1.0, 5e-3}, {2.0, 0.0}});
    constexpr double k_deviation = 0.01;

    const compat::Path path(points, k_deviation);
    BOOST_REQUIRE(!path.getError());
    BOOST_CHECK_GT(path.getConfig(path.getLength() / 2.0)[1], 1e-3);

    namespace totg = viam::trajex::totg;
    viam::trajex::xmatrix<> native_points = {{0.0, 0.0}, {1.0, 5e-3}, {2.0, 0.0}};
    const auto merged = totg::path::create(native_points, totg::path::options{}.set_max_deviation(k_deviation));
    const auto midpoint = merged.configuration(viam::trajex::arc_length{static_cast<double>(merged.length()) / 2.0});
    BOOST_CHECK_LT(std::abs(midpoint(1)), 1e-9);
}

BOOST_AUTO_TEST_CASE(switching_points_are_not_reported) {
    const compat::Path path(waypoints({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}}), 0.1);
    BOOST_CHECK(path.getSwitchingPoints().empty());
    bool discontinuity = false;
    BOOST_CHECK_EQUAL(path.getNextSwitchingPoint(0.0, discontinuity), path.getLength());
    BOOST_CHECK(discontinuity);
}

BOOST_AUTO_TEST_CASE(too_few_waypoints_is_captured) {
    const compat::Path path(waypoints({{1.0, 2.0}}));
    BOOST_CHECK(holds<std::invalid_argument>(path.getError()));
    BOOST_CHECK_EQUAL(path.getLength(), 0.0);
    BOOST_CHECK_EQUAL(path.getConfig(0.0).size(), 0);
    BOOST_CHECK_EQUAL(path.getTangent(0.0).size(), 0);
    BOOST_CHECK_EQUAL(path.getCurvature(0.0).size(), 0);
}

BOOST_AUTO_TEST_CASE(mismatched_dimensions_are_captured) {
    const compat::Path path(waypoints({{0.0, 0.0}, {1.0}}));
    BOOST_CHECK(holds<std::invalid_argument>(path.getError()));
    BOOST_CHECK_EQUAL(path.getLength(), 0.0);
}

BOOST_AUTO_TEST_CASE(copies_are_independent) {
    auto original = std::make_unique<compat::Path>(waypoints({{0.0}, {2.0}}));
    const compat::Path copied(*original);
    compat::Path assigned(waypoints({{0.0}, {5.0}}));
    assigned = *original;
    original.reset();

    BOOST_CHECK_CLOSE(copied.getLength(), 2.0, 1e-9);
    BOOST_CHECK_CLOSE(assigned.getLength(), 2.0, 1e-9);

    compat::Path& alias = assigned;
    assigned = alias;
    BOOST_CHECK_CLOSE(assigned.getLength(), 2.0, 1e-9);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(compat_trajectory)

// A rest to rest move over unit distance at unit acceleration never reaches the velocity limit,
// so the velocity profile is a triangle that peaks at unit velocity halfway and lasts two seconds.
BOOST_AUTO_TEST_CASE(triangular_profile) {
    const compat::Trajectory trajectory(compat::Path(waypoints({{0.0}, {1.0}})), vec({10.0}), vec({1.0}));
    BOOST_REQUIRE(trajectory.isValid());
    BOOST_CHECK(!trajectory.getError());
    BOOST_CHECK_CLOSE(trajectory.getDuration(), 2.0, 1.0);
    BOOST_CHECK(close(trajectory.getPosition(0.0), vec({0.0}), 1e-9));
    BOOST_CHECK(close(trajectory.getPosition(trajectory.getDuration()), vec({1.0}), 1e-9));
    BOOST_CHECK(close(trajectory.getVelocity(trajectory.getDuration() / 2.0), vec({1.0}), 1e-2));
    BOOST_CHECK(close(trajectory.getVelocity(0.0), vec({0.0}), 1e-9));
}

BOOST_AUTO_TEST_CASE(time_is_clamped) {
    const compat::Trajectory trajectory(compat::Path(waypoints({{0.0}, {1.0}})), vec({1.0}), vec({1.0}));
    BOOST_REQUIRE(trajectory.isValid());
    BOOST_CHECK(close(trajectory.getPosition(-1.0), vec({0.0}), 1e-9));
    BOOST_CHECK(close(trajectory.getPosition(trajectory.getDuration() + 1.0), vec({1.0}), 1e-9));
    BOOST_CHECK(close(trajectory.getVelocity(trajectory.getDuration() + 1.0), vec({0.0}), 1e-9));
    BOOST_CHECK_EQUAL(trajectory.getPosition(std::numeric_limits<double>::quiet_NaN()).size(), 0);
}

BOOST_AUTO_TEST_CASE(invalid_path_makes_invalid_trajectory) {
    const compat::Path path(waypoints({{1.0}}));
    const compat::Trajectory trajectory(path, vec({1.0}), vec({1.0}));
    BOOST_CHECK(!trajectory.isValid());
    BOOST_CHECK(trajectory.getError() == path.getError());
    BOOST_CHECK_EQUAL(trajectory.getDuration(), 0.0);
    BOOST_CHECK_EQUAL(trajectory.getPosition(0.0).size(), 0);
    BOOST_CHECK_EQUAL(trajectory.getVelocity(0.0).size(), 0);
}

BOOST_AUTO_TEST_CASE(mismatched_limits_are_captured) {
    const compat::Trajectory trajectory(compat::Path(waypoints({{0.0, 0.0}, {1.0, 1.0}})), vec({1.0}), vec({1.0}));
    BOOST_CHECK(!trajectory.isValid());
    BOOST_CHECK(holds<std::invalid_argument>(trajectory.getError()));
}

BOOST_AUTO_TEST_CASE(copies_are_independent) {
    auto original = std::make_unique<compat::Trajectory>(compat::Path(waypoints({{0.0}, {1.0}})), vec({10.0}), vec({1.0}));
    const double duration = original->getDuration();
    const compat::Trajectory copied(*original);
    compat::Trajectory assigned(compat::Path(waypoints({{0.0}})), vec({1.0}), vec({1.0}));
    assigned = *original;
    original.reset();

    BOOST_CHECK(copied.isValid());
    BOOST_CHECK(assigned.isValid());
    BOOST_CHECK_EQUAL(copied.getDuration(), duration);
    BOOST_CHECK_EQUAL(assigned.getDuration(), duration);
    BOOST_CHECK(close(assigned.getPosition(duration), vec({1.0}), 1e-9));
}

BOOST_AUTO_TEST_SUITE_END()

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)

// Runs the reference implementation and compat in one program, on the same inputs, through the
// same interface, which is the side by side use the namespaced classes exist for. We don't check
// that the two agree, since compat is a different implementation and may behave differently.
BOOST_AUTO_TEST_SUITE(compat_alongside_legacy)

namespace {

void run_both(const std::list<Eigen::VectorXd>& points,
              double max_deviation,
              const Eigen::VectorXd& max_velocity,
              const Eigen::VectorXd& max_acceleration) {
    const ::Trajectory legacy(::Path(points, max_deviation), max_velocity, max_acceleration);
    const compat::Trajectory compat(compat::Path(points, max_deviation), max_velocity, max_acceleration);
    BOOST_REQUIRE(legacy.isValid());
    BOOST_REQUIRE(compat.isValid());

    BOOST_TEST_MESSAGE("duration: legacy " << legacy.getDuration() << "s, compat " << compat.getDuration() << "s");
    BOOST_CHECK(close(legacy.getPosition(legacy.getDuration()), points.back(), 1e-6));
    BOOST_CHECK(close(compat.getPosition(compat.getDuration()), points.back(), 1e-9));
}

}  // namespace

// The waypoints from the example program that ships with the reference implementation.
BOOST_AUTO_TEST_CASE(reference_example) {
    run_both(
        waypoints({{0.0, 0.0, 0.0}, {0.0, 0.2, 1.0}, {0.0, 3.0, 0.5}, {1.1, 2.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}),
        0.1,
        vec({1.0, 1.0, 1.0}),
        vec({1.0, 1.0, 1.0}));
}

BOOST_AUTO_TEST_CASE(square_with_unequal_limits) {
    run_both(waypoints({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.5}}), 0.05, vec({1.0, 0.5}), vec({2.0, 1.0}));
}

BOOST_AUTO_TEST_SUITE_END()

#endif
