#include <viam/trajex/totg/tools/planner.hpp>

#include <sstream>
#include <string>

#include <boost/test/unit_test.hpp>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/totg/tools/replay.hpp>

#include "../../test/test_utils.hpp"

namespace {

using namespace viam::trajex::totg;
using viam::trajex::totg::test::gp12_model_table;

struct test_receiver {
    int segment_count = 0;
    double total_duration = 0.0;
    std::chrono::microseconds total_gen_time{};
};

planner<test_receiver>::config simple_config() {
    return {
        .velocity_limits = xt::xarray<double>{1.0, 1.0, 1.0},
        .acceleration_limits = xt::xarray<double>{1.0, 1.0, 1.0},
        .path_blend_tolerance = 0.001,
        .colinearization_ratio = std::nullopt,
        .min_blend_curvature = std::nullopt,
        .max_blend_curvature = std::nullopt,
    };
}

// waypoint_accumulator views data, doesn't own it, so the xarray must
// outlive the accumulator. Stash is the mechanism for that.
waypoint_accumulator stash_waypoints(planner<test_receiver>& p, xt::xarray<double> wp) {
    auto data = p.stash(std::move(wp));
    return waypoint_accumulator{*data};
}

}  // namespace

BOOST_AUTO_TEST_SUITE(planner_tests)

BOOST_AUTO_TEST_CASE(missing_waypoint_provider_throws) {
    auto p = planner<test_receiver>(simple_config());

    BOOST_CHECK_THROW(p.execute([](const auto&, const auto&, const auto&) -> std::optional<test_receiver> { return std::nullopt; }),
                      std::logic_error);
}

BOOST_AUTO_TEST_CASE(fewer_than_two_waypoints_skips_algorithms) {
    bool decider_called = false;

    auto result = planner<test_receiver>(simple_config())
                      .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{1.0, 2.0, 3.0}}); })
                      .with_totg([](const auto&, test_receiver&, const waypoint_accumulator&, const trajectory&, auto) {})
                      .execute([&](const auto& p, const auto& totg, const auto& legacy) -> std::optional<test_receiver> {
                          decider_called = true;
                          BOOST_CHECK_EQUAL(p.processed_waypoint_count(), 1U);
                          BOOST_CHECK(!totg.receiver);
                          BOOST_CHECK(!totg.error);
                          BOOST_CHECK(!legacy.receiver);
                          BOOST_CHECK(!legacy.error);
                          return std::nullopt;
                      });

    BOOST_CHECK(decider_called);
    BOOST_CHECK(!result);
}

BOOST_AUTO_TEST_CASE(totg_only_success) {
    bool success_called = false;

    auto result = planner<test_receiver>(simple_config())
                      .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}); })
                      .with_totg([&](const auto&, test_receiver& acc, const waypoint_accumulator&, trajectory&& traj, auto elapsed) {
                          success_called = true;
                          acc.segment_count++;
                          acc.total_duration += traj.duration().count();
                          acc.total_gen_time += elapsed;
                      })
                      .execute([](const auto&, const auto& totg, const auto& legacy) -> std::optional<test_receiver> {
                          BOOST_CHECK(totg.receiver.has_value());
                          BOOST_CHECK(!legacy.receiver);
                          BOOST_CHECK(!legacy.error);
                          return std::move(totg.receiver);
                      });

    BOOST_CHECK(success_called);
    BOOST_REQUIRE(result);
    BOOST_CHECK_EQUAL(result->segment_count, 1);
    BOOST_CHECK_GT(result->total_duration, 0.0);
}

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
BOOST_AUTO_TEST_CASE(legacy_only_success) {
    bool success_called = false;

    auto result =
        planner<test_receiver>(simple_config())
            .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}); })
            .with_legacy(
                [&](const auto&, test_receiver& acc, const waypoint_accumulator&, const Path&, const Trajectory& traj, auto elapsed) {
                    success_called = true;
                    acc.segment_count++;
                    acc.total_duration += traj.getDuration();
                    acc.total_gen_time += elapsed;
                })
            .execute([](const auto&, const auto& totg, const auto& legacy) -> std::optional<test_receiver> {
                BOOST_CHECK(!totg.receiver);
                BOOST_CHECK(legacy.receiver.has_value());
                return std::move(legacy.receiver);
            });

    BOOST_CHECK(success_called);
    BOOST_REQUIRE(result);
    BOOST_CHECK_EQUAL(result->segment_count, 1);
    BOOST_CHECK_GT(result->total_duration, 0.0);
}

BOOST_AUTO_TEST_CASE(both_algorithms_success) {
    bool totg_called = false;
    bool legacy_called = false;

    auto result =
        planner<test_receiver>(simple_config())
            .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}); })
            .with_totg([&](const auto&, test_receiver& acc, const waypoint_accumulator&, trajectory&& traj, auto) {
                totg_called = true;
                acc.segment_count++;
                acc.total_duration += traj.duration().count();
            })
            .with_legacy([&](const auto&, test_receiver& acc, const waypoint_accumulator&, const Path&, const Trajectory& traj, auto) {
                legacy_called = true;
                acc.segment_count++;
                acc.total_duration += traj.getDuration();
            })
            .execute([](const auto&, const auto& totg, const auto& legacy) -> std::optional<test_receiver> {
                BOOST_CHECK(totg.receiver.has_value());
                BOOST_CHECK(legacy.receiver.has_value());
                return std::move(totg.receiver);
            });

    BOOST_CHECK(totg_called);
    BOOST_CHECK(legacy_called);
    BOOST_REQUIRE(result);
}
#endif

BOOST_AUTO_TEST_CASE(preprocessor_runs_before_algorithms) {
    auto result = planner<test_receiver>(simple_config())
                      .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}); })
                      .with_waypoint_preprocessor(
                          [](auto&, waypoint_accumulator& accumulator) { accumulator = deduplicate_waypoints(accumulator, 1000.0); })
                      .with_totg([](const auto&, test_receiver&, const waypoint_accumulator&, const trajectory&, auto) {
                          BOOST_FAIL("totg should not run on < 2 waypoints");
                      })
                      .execute([](const auto& p, const auto& totg, const auto&) -> std::optional<test_receiver> {
                          BOOST_CHECK_EQUAL(p.processed_waypoint_count(), 1U);
                          BOOST_CHECK(!totg.receiver);
                          return std::nullopt;
                      });

    BOOST_CHECK(!result);
}

BOOST_AUTO_TEST_CASE(validator_can_reject_move) {
    BOOST_CHECK_THROW(planner<test_receiver>(simple_config())
                          .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}); })
                          .with_move_validator([](auto&, const waypoint_accumulator&) { throw std::runtime_error("move rejected"); })
                          .with_totg([](const auto&, test_receiver&, const waypoint_accumulator&, const trajectory&, auto) {})
                          .execute([](const auto&, const auto&, const auto&) -> std::optional<test_receiver> {
                              BOOST_FAIL("decider should not be called after validation failure");
                              return std::nullopt;
                          }),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(segmenter_produces_multiple_segments) {
    int totg_segment_count = 0;

    auto result =
        planner<test_receiver>(simple_config())
            .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}); })
            .with_segmenter([](auto&, waypoint_accumulator accumulator) { return segment_at_reversals(std::move(accumulator)); })
            .with_totg([&](const auto&, test_receiver& acc, const waypoint_accumulator&, const trajectory&, auto) {
                totg_segment_count++;
                acc.segment_count++;
            })
            .execute([](const auto&, const auto& totg, const auto&) -> std::optional<test_receiver> { return std::move(totg.receiver); });

    BOOST_REQUIRE(result);
    BOOST_CHECK_EQUAL(totg_segment_count, 2);
    BOOST_CHECK_EQUAL(result->segment_count, 2);
}

BOOST_AUTO_TEST_CASE(failure_disengages_receiver_for_remaining_segments) {
    int success_count = 0;
    bool failure_called = false;

    auto result =
        planner<test_receiver>(simple_config())
            .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}); })
            .with_segmenter([](auto&, waypoint_accumulator accumulator) { return segment_at_reversals(std::move(accumulator)); })
            .with_totg(
                [&](const auto&, test_receiver&, const waypoint_accumulator&, const trajectory&, auto) {
                    success_count++;
                    throw std::runtime_error("synthetic failure");
                },
                [&](const auto&, const test_receiver&, const waypoint_accumulator&, const std::exception& e) {
                    failure_called = true;
                    BOOST_CHECK_EQUAL(std::string(e.what()), "synthetic failure");
                })
            .execute([](const auto&, const auto& totg, const auto&) -> std::optional<test_receiver> {
                BOOST_CHECK(!totg.receiver);
                BOOST_CHECK(totg.error != nullptr);
                return std::nullopt;
            });

    BOOST_CHECK_EQUAL(success_count, 1);
    BOOST_CHECK(failure_called);
    BOOST_CHECK(!result);
}

BOOST_AUTO_TEST_CASE(stash_extends_data_lifetime) {
    auto result =
        planner<test_receiver>(simple_config())
            .with_waypoint_provider([](auto& p) {
                auto data = p.stash(xt::xarray<double>{{0.0, 0.0, 0.0}});
                waypoint_accumulator accumulator{*data};
                auto more = p.stash(xt::xarray<double>{{1.0, 0.0, 0.0}});
                accumulator.add_waypoints(*more);
                return accumulator;
            })
            .with_totg([](const auto&, test_receiver& acc, const waypoint_accumulator&, const trajectory&, auto) { acc.segment_count++; })
            .execute([](const auto&, const auto& totg, const auto&) -> std::optional<test_receiver> { return std::move(totg.receiver); });

    BOOST_REQUIRE(result);
    BOOST_CHECK_EQUAL(result->segment_count, 1);
}

BOOST_AUTO_TEST_CASE(segment_totg_false_passes_unsegmented_to_totg) {
    int totg_segments_seen = 0;
    int legacy_segments_seen [[maybe_unused]] = 0;

    auto cfg = simple_config();
    cfg.segment_totg = false;

    planner<test_receiver>(std::move(cfg))
        .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}); })
        .with_segmenter([](auto&, waypoint_accumulator accumulator) { return segment_at_reversals(std::move(accumulator)); })
        .with_totg([&](const auto&, test_receiver&, const waypoint_accumulator&, const trajectory&, auto) { totg_segments_seen++; })
#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
        .with_legacy([&](const auto&, test_receiver&, const waypoint_accumulator&, Path&&, Trajectory&&, auto) { legacy_segments_seen++; })
#endif
        .execute([](const auto&, const auto&, const auto&) -> std::optional<test_receiver> { return std::nullopt; });

    BOOST_CHECK_EQUAL(totg_segments_seen, 1);

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
    BOOST_CHECK_EQUAL(legacy_segments_seen, 2);
#endif
}

BOOST_AUTO_TEST_CASE(processed_waypoint_count_reflects_preprocessing) {
    planner<test_receiver>(simple_config())
        .with_waypoint_provider(
            [](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}); })
        .with_waypoint_preprocessor(
            [](auto&, waypoint_accumulator& accumulator) { accumulator = deduplicate_waypoints(accumulator, 1e-6); })
        .with_totg([](const auto&, test_receiver&, const waypoint_accumulator&, const trajectory&, auto) {})
        .execute([](const auto& p, const auto&, const auto&) -> std::optional<test_receiver> {
            BOOST_CHECK_EQUAL(p.processed_waypoint_count(), 3U);
            return std::nullopt;
        });
}

BOOST_AUTO_TEST_CASE(decider_return_type_is_flexible) {
    const int result = planner<test_receiver>(simple_config())
                           .with_waypoint_provider([](auto& p) { return stash_waypoints(p, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}); })
                           .with_totg([](const auto&, test_receiver&, const waypoint_accumulator&, const trajectory&, auto) {})
                           .execute([](const auto&, const auto& totg, const auto&) -> int { return totg.receiver ? 42 : -1; });

    BOOST_CHECK_EQUAL(result, 42);
}

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
// The legacy generator cannot enforce a TCP speed limit, so it must never run as a fallback
// for one: a planner whose config carries a TCP limit refuses to register legacy at all.
BOOST_AUTO_TEST_CASE(legacy_registration_with_tcp_limit_throws) {
    auto cfg = simple_config();
    cfg.tcp = trajectory::tcp_limits::from(gp12_model_table(), 0.5);

    planner<test_receiver> p(std::move(cfg));
    BOOST_CHECK_THROW(p.with_legacy([](const auto&, test_receiver&, const waypoint_accumulator&, Path&&, Trajectory&&, auto) {}),
                      std::logic_error);
}

// Legacy replay of a TCP-carrying record is a deliberate uncapped comparison run: the record's
// TCP limit is dropped explicitly (visible as an empty config field) rather than carried into a
// generator that would silently ignore it.
BOOST_AUTO_TEST_CASE(legacy_replay_of_tcp_record_drops_tcp_limit) {
    const auto table = gp12_model_table();

    planner<test_receiver>::config cfg{
        .velocity_limits = xt::xarray<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        .acceleration_limits = xt::xarray<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        .path_blend_tolerance = 0.01,
        .colinearization_ratio = std::nullopt,
    };
    cfg.tcp = trajectory::tcp_limits::from(table, 0.5);
    cfg.model_table = table;

    planner<test_receiver> p(std::move(cfg));
    auto data = p.stash(xt::xarray<double>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.3, 0.1, 0.0, 0.0, 0.0, 0.0}});
    const std::string record = p.serialize_for_replay(waypoint_accumulator{*data});

    std::istringstream in(record);
    auto replayed = legacy_replay_planner::create(in);
    BOOST_CHECK(!replayed.get_config().tcp.has_value());

    auto outcome = replayed.execute([](const auto&, const auto&, auto legacy) { return legacy; });
    BOOST_REQUIRE(outcome.receiver.has_value());
    BOOST_CHECK(outcome.receiver->result.has_value());
}
#endif

// A malformed model_table must be rejected where the config is accepted. Deferring the check to
// serialize_for_replay would throw on the failure paths that call it to record diagnostics,
// destroying the replay record for the original error.
BOOST_AUTO_TEST_CASE(malformed_model_table_rejected_at_construction) {
    {
        auto cfg = simple_config();
        cfg.model_table = xt::xarray<double>{1.0, 2.0, 3.0};  // 1-D, not (n, 10)
        BOOST_CHECK_THROW(static_cast<void>(planner<test_receiver>(cfg)), std::invalid_argument);
    }
    {
        auto cfg = simple_config();
        cfg.model_table = xt::xarray<double>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};  // (2, 3), not (n, 10)
        BOOST_CHECK_THROW(static_cast<void>(planner<test_receiver>(cfg)), std::invalid_argument);
    }
}

// A config carrying a TCP limit and its model-table provenance survives a
// serialize_for_replay -> replay_planner round-trip: the scalar cap is preserved, the model-table is
// recorded, and the opaque jacobian callback is rebuilt from it.
BOOST_AUTO_TEST_CASE(replay_record_round_trips_tcp_limit) {
    const auto table = gp12_model_table();

    planner<test_receiver>::config cfg{
        .velocity_limits = xt::xarray<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        .acceleration_limits = xt::xarray<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        .path_blend_tolerance = 0.01,
        .colinearization_ratio = std::nullopt,
    };
    cfg.tcp = trajectory::tcp_limits::from(table, 0.5);
    cfg.model_table = table;

    planner<test_receiver> p(std::move(cfg));
    auto data = p.stash(xt::xarray<double>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.3, 0.1, 0.0, 0.0, 0.0, 0.0}});
    const std::string record = p.serialize_for_replay(waypoint_accumulator{*data});

    std::istringstream in(record);
    auto replayed = replay_planner::create(in);
    const auto& rc = replayed.get_config();

    BOOST_REQUIRE(rc.tcp.has_value());
    BOOST_CHECK_CLOSE(rc.tcp->max_linear_velocity, 0.5, 1e-9);
    BOOST_REQUIRE(rc.model_table.has_value());
    BOOST_REQUIRE_EQUAL(rc.model_table->dimension(), 2U);
    BOOST_CHECK_EQUAL(rc.model_table->shape(0), 7U);
    BOOST_CHECK_EQUAL(rc.model_table->shape(1), 10U);

    // The rebuilt jacobian is callable and returns the 3xN linear-velocity block for the six
    // actuated joints, confirming the callback was reconstructed (not merely the scalar copied).
    BOOST_REQUIRE(static_cast<bool>(rc.tcp->linear_jacobian));
    const auto J = rc.tcp->linear_jacobian(xt::xarray<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    BOOST_REQUIRE_EQUAL(J.dimension(), 2U);
    BOOST_CHECK_EQUAL(J.shape(0), 3U);
    BOOST_CHECK_EQUAL(J.shape(1), 6U);

    // The reconstructed limit generates a trajectory end to end.
    auto outcome = replayed.execute([](const auto&, auto tx, const auto&) { return tx; });
    BOOST_REQUIRE(outcome.receiver.has_value());
    BOOST_CHECK(outcome.receiver->traj.has_value());
}

// Without model-table provenance the TCP jacobian cannot be reconstructed, so serialize_for_replay
// records neither the cap nor the table, and the round-tripped config carries no TCP limit.
BOOST_AUTO_TEST_CASE(replay_record_omits_tcp_without_model_table) {
    planner<test_receiver>::config cfg{
        .velocity_limits = xt::xarray<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        .acceleration_limits = xt::xarray<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        .path_blend_tolerance = 0.01,
        .colinearization_ratio = std::nullopt,
    };
    cfg.tcp = trajectory::tcp_limits::from(gp12_model_table(), 0.5);
    // cfg.model_table intentionally left unset.

    planner<test_receiver> p(std::move(cfg));
    auto data = p.stash(xt::xarray<double>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.3, 0.1, 0.0, 0.0, 0.0, 0.0}});
    const std::string record = p.serialize_for_replay(waypoint_accumulator{*data});

    std::istringstream in(record);
    auto replayed = replay_planner::create(in);
    BOOST_CHECK(!replayed.get_config().tcp.has_value());
    BOOST_CHECK(!replayed.get_config().model_table.has_value());
}

// A record that names a TCP cap but carries no model-table cannot rebuild the jacobian and is
// rejected rather than silently dropping the limit.
BOOST_AUTO_TEST_CASE(replay_tcp_speed_without_model_table_throws) {
    const std::string record = R"({
        "schema_version": 2,
        "max_velocity_vec_rads_per_sec": [1, 1, 1, 1, 1, 1],
        "max_acceleration_vec_rads_per_sec2": [1, 1, 1, 1, 1, 1],
        "path_tolerance_delta_rads": 0.01,
        "tcp_max_linear_velocity": 0.5,
        "waypoints_rads": [[0, 0, 0, 0, 0, 0], [0.3, 0.1, 0, 0, 0, 0]]
    })";
    std::istringstream in(record);
    BOOST_CHECK_THROW(replay_planner::create(in), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
