#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <utility>

#include <boost/test/unit_test.hpp>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/test/test_utils.hpp>
#include <viam/trajex/totg/tools/replay.hpp>
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

}  // namespace

BOOST_AUTO_TEST_SUITE(online_extension_verification)

// Confirms that integration points generated from a prefix of a waypoint set form a
// bit-exact prefix of the integration points generated from the full set, except for a
// short divergent tail around the prefix's terminal where the path coalesces differently.
BOOST_AUTO_TEST_CASE(lab_sander_prefix_integration_points_bit_exact) {
    const auto fixture_path =
        std::filesystem::path(VIAM_TRAJEX_TEST_DATA_DIR) / "lab_sander_backward_integration_exceeded-20260507.trajex-totg-replay.json";

    // The lab_sander fixture has 3920 waypoints. Using the first half gives a non-trivial
    // 1960-waypoint prefix while leaving plenty of path beyond it in the full trajectory.
    constexpr std::size_t k_fixture_waypoint_count = 3920;
    constexpr std::size_t k_prefix_waypoint_count = k_fixture_waypoint_count / 2;

    auto run = [&fixture_path](std::optional<std::size_t> prefix) {
        auto planner = viam::trajex::totg::replay_planner::create(fixture_path, prefix);
        auto outcome = planner.execute([](const auto&, auto tx, const auto&) { return tx; });
        BOOST_REQUIRE_MESSAGE(outcome.receiver.has_value(), "trajectory generation failed");
        BOOST_REQUIRE(outcome.receiver->traj.has_value());
        return std::make_pair(std::move(*outcome.receiver->traj), planner.processed_waypoint_count());
    };

    auto [traj_full, full_waypoint_count] = run(std::nullopt);
    // Tripwire: if the fixture's waypoint count ever changes, force the test author back here.
    BOOST_REQUIRE_EQUAL(full_waypoint_count, k_fixture_waypoint_count);

    auto [traj_prefix, prefix_waypoint_count] = run(k_prefix_waypoint_count);
    BOOST_REQUIRE_EQUAL(prefix_waypoint_count, k_prefix_waypoint_count);

    const auto& points_full = traj_full.get_integration_points();
    const auto& points_prefix = traj_prefix.get_integration_points();

    const auto [it_prefix, it_full] = std::ranges::mismatch(points_prefix, points_full);
    const auto match_count = static_cast<std::size_t>(it_prefix - points_prefix.begin());
    const double match_fraction = static_cast<double>(match_count) / static_cast<double>(points_prefix.size());

    BOOST_TEST_MESSAGE("integration points (full):        " << points_full.size());
    BOOST_TEST_MESSAGE("integration points (prefix):      " << points_prefix.size());
    BOOST_TEST_MESSAGE("bit-exact common prefix length:   " << match_count);
    BOOST_TEST_MESSAGE("bit-exact common prefix fraction: " << match_fraction);

    // The match fraction on this fixture is empirically ~96.5%.
    BOOST_CHECK_CLOSE(match_fraction, 0.96, 1.0);
}

BOOST_AUTO_TEST_SUITE_END()  // online_extension_verification
