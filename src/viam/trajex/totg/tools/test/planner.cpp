#include <viam/trajex/totg/tools/planner.hpp>

#include <array>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <viam/trajex/totg/test/test_utils.hpp>
#include <viam/trajex/totg/tools/replay.hpp>
#include <viam/trajex/types/xt.hpp>

namespace {

using namespace viam::trajex::totg;
using viam::trajex::xmatrix;
using viam::trajex::xvector;
using viam::trajex::totg::test::gp12_chain;

struct test_receiver {
    int segment_count = 0;
    double total_duration = 0.0;
    std::chrono::microseconds total_gen_time{};
};

// Limits for the six joints of gp12_chain.
planner<test_receiver>::config gp12_config() {
    return {
        .velocity_limits = xvector<>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        .acceleration_limits = xvector<>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        .path_blend_tolerance = 0.01,
        .colinearization_ratio = std::nullopt,
    };
}

// The replay record a planner with this config writes for a short six-joint move.
std::string record_for(planner<test_receiver>::config cfg) {
    planner<test_receiver> p(std::move(cfg));
    auto data = p.stash(xmatrix<>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.3, 0.1, 0.0, 0.0, 0.0, 0.0}});
    return p.serialize_for_replay(waypoint_accumulator{*data});
}

// A chain's joints as visit_joints reports them, in a form that compares exactly. The first
// element is 0 for a fixed joint, 1 for a rotating one and 2 for a translating one.
using joint_description = std::tuple<int, std::array<double, 3>, std::array<double, 4>, std::array<double, 3>>;

std::vector<joint_description> describe(const viam::trajex::kinematics::chain& kinematics) {
    using chain = viam::trajex::kinematics::chain;
    struct collector {
        std::vector<joint_description> joints;
        void operator()(const chain::joint_origin& o) {
            joints.emplace_back(0, o.translation, o.rotation, std::array<double, 3>{});
        }
        void operator()(const chain::joint_origin& o, const chain::rotation_axis& a) {
            joints.emplace_back(1, o.translation, o.rotation, a.direction);
        }
        void operator()(const chain::joint_origin& o, const chain::translation_axis& a) {
            joints.emplace_back(2, o.translation, o.rotation, a.direction);
        }
    } c;
    kinematics.visit_joints(c);
    return c.joints;
}

planner<test_receiver>::config simple_config() {
    return {
        .velocity_limits = xvector<>{1.0, 1.0, 1.0},
        .acceleration_limits = xvector<>{1.0, 1.0, 1.0},
        .path_blend_tolerance = 0.001,
        .colinearization_ratio = std::nullopt,
        .min_blend_curvature = std::nullopt,
        .max_blend_curvature = std::nullopt,
    };
}

// waypoint_accumulator views data, doesn't own it, so the matrix must
// outlive the accumulator. Stash is the mechanism for that.
waypoint_accumulator stash_waypoints(planner<test_receiver>& p, xmatrix<> wp) {
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
                auto data = p.stash(xmatrix<>{{0.0, 0.0, 0.0}});
                waypoint_accumulator accumulator{*data};
                auto more = p.stash(xmatrix<>{{1.0, 0.0, 0.0}});
                accumulator.add_waypoints(*more);
                return accumulator;
            })
            .with_totg([](const auto&, test_receiver& acc, const waypoint_accumulator&, const trajectory&, auto) { acc.segment_count++; })
            .execute([](const auto&, const auto& totg, const auto&) -> std::optional<test_receiver> { return std::move(totg.receiver); });

    BOOST_REQUIRE(result);
    BOOST_CHECK_EQUAL(result->segment_count, 1);
}

// An outcome shares the planner's stash, so data the planner stashed outlives the planner for as
// long as an outcome does, and a receiver can keep the waypoint_accumulators its callbacks saw.
BOOST_AUTO_TEST_CASE(outcome_keeps_stash_alive_after_planner) {
    std::weak_ptr<xmatrix<>> stashed;
    std::optional<planner<test_receiver>::algorithm_outcome> outcome;

    // The planner is a temporary, gone at the end of this statement.
    outcome.emplace(
        planner<test_receiver>(simple_config())
            .with_waypoint_provider([&stashed](auto& p) {
                auto data = p.stash(xmatrix<>{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}});
                stashed = data;
                return waypoint_accumulator{*data};
            })
            .with_totg([](const auto&, test_receiver& acc, const waypoint_accumulator&, trajectory&&, auto) { acc.segment_count++; })
            .execute([](const auto&, auto totg, const auto&) { return totg; }));

    BOOST_REQUIRE(outcome->receiver.has_value());
    BOOST_CHECK(!stashed.expired());
    outcome.reset();
    BOOST_CHECK(stashed.expired());
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
    auto cfg = gp12_config();
    cfg.kinematics_chain = gp12_chain();
    cfg.tcp_max_linear_speed = 0.5;

    planner<test_receiver> p(std::move(cfg));
    BOOST_CHECK_THROW(p.with_legacy([](const auto&, test_receiver&, const waypoint_accumulator&, Path&&, Trajectory&&, auto) {}),
                      std::logic_error);
}

// Legacy replay of a TCP-carrying record is a deliberate uncapped comparison run: the record's
// cap is dropped explicitly (visible as an empty config field) rather than carried into a
// generator that would silently ignore it. The kinematics are kept, since they are harmless.
BOOST_AUTO_TEST_CASE(legacy_replay_of_tcp_record_drops_tcp_limit) {
    auto cfg = gp12_config();
    cfg.kinematics_chain = gp12_chain();
    cfg.tcp_max_linear_speed = 0.5;

    std::istringstream in(record_for(std::move(cfg)));
    auto replayed = legacy_replay_planner::create(in);
    BOOST_CHECK(!replayed.get_config().tcp_max_linear_speed.has_value());
    BOOST_CHECK(replayed.get_config().kinematics_chain.has_value());

    auto outcome = replayed.execute([](const auto&, const auto&, auto legacy) { return legacy; });
    BOOST_REQUIRE(outcome.receiver.has_value());
    BOOST_CHECK(outcome.receiver->result.has_value());
}
#endif

// A cap with nothing to apply it to must be rejected where the config is accepted. Deferring the
// check would let serialize_for_replay, which runs on failure paths to record diagnostics, write a
// record that cannot be replayed.
BOOST_AUTO_TEST_CASE(tcp_cap_without_kinematics_rejected_at_construction) {
    auto cfg = simple_config();
    cfg.tcp_max_linear_speed = 0.5;
    BOOST_CHECK_THROW(static_cast<void>(planner<test_receiver>(cfg)), std::invalid_argument);
}

// A chain whose moving joints do not match the joint limits is refused where the config is
// accepted, with or without a cap, rather than failing partway through the first trajectory.
BOOST_AUTO_TEST_CASE(kinematics_dof_mismatch_rejected_at_construction) {
    auto mismatched = simple_config();
    mismatched.kinematics_chain = gp12_chain();
    BOOST_CHECK_THROW(static_cast<void>(planner<test_receiver>(mismatched)), std::invalid_argument);

    mismatched.tcp_max_linear_speed = 0.5;
    BOOST_CHECK_THROW(static_cast<void>(planner<test_receiver>(mismatched)), std::invalid_argument);

    auto matched = gp12_config();
    matched.kinematics_chain = gp12_chain();
    matched.tcp_max_linear_speed = 0.5;
    BOOST_CHECK_NO_THROW(static_cast<void>(planner<test_receiver>(matched)));
}

// A config carrying kinematics and a cap survives a serialize_for_replay -> replay_planner round
// trip: the cap is preserved, the chain comes back joint for joint as it was built, and the
// rebuilt limit generates a trajectory.
BOOST_AUTO_TEST_CASE(replay_record_round_trips_tcp_limit) {
    auto cfg = gp12_config();
    cfg.kinematics_chain = gp12_chain();
    cfg.tcp_max_linear_speed = 0.5;

    std::istringstream in(record_for(std::move(cfg)));
    auto replayed = replay_planner::create(in);
    const auto& rc = replayed.get_config();

    BOOST_REQUIRE(rc.tcp_max_linear_speed.has_value());
    BOOST_CHECK_EQUAL(*rc.tcp_max_linear_speed, 0.5);
    BOOST_REQUIRE(rc.kinematics_chain.has_value());
    BOOST_CHECK(describe(*rc.kinematics_chain) == describe(gp12_chain()));

    auto outcome = replayed.execute([](const auto&, auto tx, const auto&) { return tx; });
    BOOST_REQUIRE(outcome.receiver.has_value());
    BOOST_CHECK(outcome.receiver->traj.has_value());
}

// Every joint kind, and quaternions and axes that are not unit length, come back exactly as they
// were given, since the record keeps what the builder was handed rather than what it derived.
BOOST_AUTO_TEST_CASE(replay_record_round_trips_every_joint_kind) {
    using chain = viam::trajex::kinematics::chain;
    const auto kinematics = chain::builder{}
                                .extend(chain::joint_origin{.translation = {0.1, -0.2, 0.3}, .rotation = {2.0, 0.1, -0.3, 0.4}},
                                        chain::rotation_axis{{0.0, 0.0, 3.0}})
                                .extend(chain::joint_origin{.translation = {0.5, 0.0, 0.0}}, chain::translation_axis{{1.0, 1.0, 0.0}})
                                .extend(chain::joint_origin{.translation = {0.0, 0.0, 0.1}, .rotation = {0.0, 1.0, 0.0, 0.0}})
                                .build();

    planner<test_receiver>::config cfg{
        .velocity_limits = xvector<>{1.0, 1.0},
        .acceleration_limits = xvector<>{1.0, 1.0},
        .path_blend_tolerance = 0.01,
        .colinearization_ratio = std::nullopt,
    };
    cfg.kinematics_chain = kinematics;

    planner<test_receiver> p(std::move(cfg));
    auto data = p.stash(xmatrix<>{{0.0, 0.0}, {0.3, 0.1}});
    std::istringstream in(p.serialize_for_replay(waypoint_accumulator{*data}));
    const auto replayed = replay_planner::create(in);

    BOOST_REQUIRE(replayed.get_config().kinematics_chain.has_value());
    BOOST_CHECK(describe(*replayed.get_config().kinematics_chain) == describe(kinematics));
    BOOST_CHECK(!replayed.get_config().tcp_max_linear_speed.has_value());
}

// A record that names a TCP cap but carries no kinematics_chain has nothing to apply the cap to and
// is rejected rather than silently dropping the limit. That is also what becomes of a schema v2
// record that described its kinematics as a model_table.
BOOST_AUTO_TEST_CASE(replay_tcp_speed_without_kinematics_chain_throws) {
    const std::string record = R"({
        "schema_version": 2,
        "max_velocity_vec_rads_per_sec": [1, 1, 1, 1, 1, 1],
        "max_acceleration_vec_rads_per_sec2": [1, 1, 1, 1, 1, 1],
        "path_tolerance_delta_rads": 0.01,
        "tcp_max_linear_speed": 0.5,
        "model_table": [[0, 0, 0.45, 0, 0, 0, 0, 0, 1, 0]],
        "waypoints_rads": [[0, 0, 0, 0, 0, 0], [0.3, 0.1, 0, 0, 0, 0]]
    })";
    std::istringstream in(record);
    BOOST_CHECK_THROW(replay_planner::create(in), std::runtime_error);
}

// A joint carrying both axis keys names two kinds at once, and the record is refused.
BOOST_AUTO_TEST_CASE(replay_joint_with_both_axes_throws) {
    const std::string record = R"({
        "schema_version": 3,
        "max_velocity_vec_rads_per_sec": [1],
        "max_acceleration_vec_rads_per_sec2": [1],
        "path_tolerance_delta_rads": 0.01,
        "kinematics_chain": [{"xyz": [0, 0, 0], "wxyz": [1, 0, 0, 0], "rotation_axis": [0, 0, 1], "translation_axis": [1, 0, 0]}],
        "waypoints_rads": [[0], [0.3]]
    })";
    std::istringstream in(record);
    BOOST_CHECK_THROW(replay_planner::create(in), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
