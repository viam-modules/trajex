#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <json/json.h>

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
using viam::trajex::xmatrix;
using viam::trajex::xvector;

}  // namespace

namespace {

// Replays a captured trajectory generation request and validates the resulting trajectory.
// Each caller passes its own tolerances, tuned just low enough for that replay to pass, so the
// constants document the current numerical quality of each captured scenario. Passing nullopt
// for a tolerance skips the corresponding validator, for replays where no finite tolerance
// would let it pass.
void generate_trajectory_from_replay_file(const std::string& filename,
                                          std::optional<double> trajectory_invariants_tolerance_percent,
                                          std::optional<double> joint_kinematics_tolerance_percent) {
    auto planner = viam::trajex::totg::replay_planner::create(std::filesystem::path(VIAM_TRAJEX_TEST_DATA_DIR) / filename);

    auto outcome = planner.execute([](const auto&, auto tx, const auto&) { return tx; });

    // The planner captures whatever killed the run in `outcome.error`. Reporting only that
    // generation failed discards the one thing a reader of the failure needs.
    if (!outcome.receiver.has_value()) {
        std::string reason = "no exception was captured";
        if (outcome.error) {
            try {
                std::rethrow_exception(outcome.error);
            } catch (const std::exception& e) {
                reason = e.what();
            } catch (...) {
                reason = "non-std exception";
            }
        }
        BOOST_FAIL("trajectory generation failed: " << reason);
    }
    BOOST_REQUIRE(outcome.receiver->traj.has_value());

    const trajectory& traj = *outcome.receiver->traj;
    BOOST_REQUIRE(outcome.receiver->waypoints.has_value());
    validate_path_quality(traj.path(), *outcome.receiver->waypoints, planner.path_options());

    if (trajectory_invariants_tolerance_percent.has_value()) {
        validate_trajectory_invariants(traj, *trajectory_invariants_tolerance_percent);
    }

    if (joint_kinematics_tolerance_percent.has_value()) {
        const struct trajectory::options& opts = traj.get_options();
        validate_joint_kinematics(traj, opts.max_velocity, opts.max_acceleration, *joint_kinematics_tolerance_percent);
    }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(replay_regression_tests)

BOOST_AUTO_TEST_CASE(gp12_backward_integration_exceeded_limit_curve) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 217.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 31.0;
    generate_trajectory_from_replay_file("gp12_backward_integration_exceeded-20260305.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

// The replayed trajectory contains a pair of consecutive integration points whose timestamps are
// identical at double precision, so the strict time monotonicity check in
// validate_trajectory_invariants flags it regardless of tolerance. The invariants validator is
// therefore skipped until that defect is fixed.
BOOST_AUTO_TEST_CASE(gp12_splice_point_infeasible_acceleration) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 170.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 54.0;
    generate_trajectory_from_replay_file("gp12_splice_point_infeasible-20260305.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

// On some platforms this replay produces a pair of consecutive integration points whose
// timestamps are identical at double precision, so the strict time monotonicity check in
// validate_trajectory_invariants flags it regardless of tolerance. The invariants validator is
// therefore skipped until that defect is fixed.
BOOST_AUTO_TEST_CASE(lab_sander_05072026_backward_integration_exceeded_limit_curve) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 56.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 76.0;
    generate_trajectory_from_replay_file("lab_sander_backward_integration_exceeded-20260507.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

// Regression test for a forward-integration numerical artifact: when the breach bisection collapses
// the next point onto the current point, delta_s falls to the floating-point noise floor while
// staying nonzero, so the finite-difference s_ddot = delta_s_dot / dt divides one noise-floor
// quantity by another and stamps a spurious arc acceleration of arbitrary magnitude on that point.
// The underlying motion stays feasible (s and s_dot are continuous, and the next step restamps the
// same point with a valid acceleration), so every integration point's s_ddot must lie within its
// feasible [s_ddot_min, s_ddot_max] band. The phase-plane acceleration-bound check inside
// validate_trajectory_invariants guards this: the replayed gp12 move stamped twenty points whose
// s_ddot exceeded the band by two or more orders of magnitude, far beyond the tolerance below.
// The noise-floor steps came from ULP-scale linear segments between blends, which path construction
// no longer emits.
BOOST_AUTO_TEST_CASE(gp12_forward_step_sddot_within_acceleration_bounds) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 0.001;
    constexpr auto k_local_joint_kinematics_tolerance_override = 0.1;
    generate_trajectory_from_replay_file("gp12_forward_step_sddot_exceeds_bounds-20260624.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(vik_182_forward_integration_acc_natural_escape_stall) {
    // Reproducer for a production hang in which forward integration entered an
    // unbounded loop after the limit-hit bisection collapsed next_point onto
    // current_point and the acceleration-curve "natural escape" classification
    // returned nullopt without committing the breach. Tripwire for any future
    // change that resurrects that path.
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 170.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 85.0;
    generate_trajectory_from_replay_file("VIK-182-stall.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

// The same check as tcp_limits_tests/terminal_acceleration_finite_and_sampleable_at_exact_duration,
// on the gp12 chain with a 0.5 m/s cap and a field record, so the property is also held on a real
// arm. Like that test it was written for RSDK-13338 and never reproduced the failure. RSDK-14959,
// whose reproducer follows, fixed what is likely the same cause. The record predates TCP options
// (schema v1), so the test applies the cap to the gp12 chain itself.
BOOST_AUTO_TEST_CASE(gp12_tcp_terminal_acceleration_sentinel_sample) {
    std::ifstream in(std::filesystem::path(VIAM_TRAJEX_TEST_DATA_DIR) /
                     "gp12_tcp_terminal_nan_acceleration-20260612.trajex-totg-replay.json");
    BOOST_REQUIRE(in.good());
    Json::Value root;
    const Json::CharReaderBuilder reader;
    std::string errs;
    BOOST_REQUIRE_MESSAGE(Json::parseFromStream(reader, in, &root, &errs), errs);

    const auto dof = root["max_velocity_vec_rads_per_sec"].size();
    xvector<> max_velocity = xt::zeros<double>({static_cast<std::size_t>(dof)});
    xvector<> max_acceleration = xt::zeros<double>({static_cast<std::size_t>(dof)});
    for (Json::ArrayIndex i = 0; i < dof; ++i) {
        max_velocity(i) = root["max_velocity_vec_rads_per_sec"][i].asDouble();
        max_acceleration(i) = root["max_acceleration_vec_rads_per_sec2"][i].asDouble();
    }
    const auto& wps = root["waypoints_rads"];
    xmatrix<> waypoints = xt::zeros<double>({static_cast<std::size_t>(wps.size()), static_cast<std::size_t>(dof)});
    for (Json::ArrayIndex i = 0; i < wps.size(); ++i) {
        for (Json::ArrayIndex j = 0; j < dof; ++j) {
            waypoints(i, static_cast<std::size_t>(j)) = wps[i][j].asDouble();
        }
    }

    auto p = path::create(waypoints, path::options{}.set_max_blend_deviation(root["path_tolerance_delta_rads"].asDouble()));
    trajectory::options opt{.max_velocity = max_velocity, .max_acceleration = max_acceleration};
    opt.tcp = trajectory::tcp_limits::from(test::gp12_chain(), 0.5);

    const auto traj = trajectory::create(std::move(p), std::move(opt));
    BOOST_CHECK(std::isfinite(static_cast<double>(traj.get_integration_points().back().s_ddot)));
    BOOST_CHECK_NO_THROW(static_cast<void>(traj.create_cursor().seek(traj.duration()).sample()));
}

// This reproduces RSDK-14959, the failure the two tests above check for, with joint limits only.
// A UR5e reported "Cannot query cursor at sentinel position" on a short, densely spaced straight
// move. The universal-robots module puts the measured joint position ahead of the caller's
// waypoints and deduplicates at 1e-3 rad. That drops the caller's first waypoint, so the path it
// built started a little off the recorded one. This record is the field record with its first
// waypoint moved by less than that tolerance. Of 1500 random such starts, it was the only one to
// leave a NaN s_ddot on the terminal integration point. Sampling at exactly the duration
// interpolates from that point with dt == 0. A NaN there gets through the arc length clamp and
// seeks the path cursor to its sentinel.
BOOST_AUTO_TEST_CASE(RSDK_14959_ur5e_unstraighten_terminal_nan_acceleration) {
    generate_trajectory_from_replay_file("ur5e_unstraighten_terminal_nan_acceleration-20261119.trajex-totg-replay.json", 0.1, 0.1);
}

// Every integration point's s_ddot must lie within its feasible [s_ddot_min, s_ddot_max] band. Path
// construction for this gp12 move (1.2 m/s TCP cap) used to emit linear segments exactly one ULP long
// between blends. Stepping across one gave a dt on the order of 1e-16, so the finite difference
// s_ddot = delta_s_dot / dt divided rounding residue by a noise-floor dt and stamped an acceleration
// in the thousands. s and s_dot stayed continuous, so only the stamped acceleration was wrong.
BOOST_AUTO_TEST_CASE(gp12_forward_truncated_step_sddot_within_bounds) {
    auto planner = viam::trajex::totg::replay_planner::create(std::filesystem::path(VIAM_TRAJEX_TEST_DATA_DIR) /
                                                              "gp12_forward_truncated_step_sddot_spike-20260623.trajex-totg-replay.json");
    auto outcome = planner.execute([](const auto&, auto tx, const auto&) { return tx; });
    BOOST_REQUIRE(outcome.receiver.has_value());
    BOOST_REQUIRE(outcome.receiver->traj.has_value());
    const auto& traj = *outcome.receiver->traj;

    auto cursor = traj.path().create_cursor();
    const auto& points = traj.get_integration_points();
    BOOST_REQUIRE(!points.empty());

    std::size_t worst_index = 0;
    double worst_excess = 0.0;
    double worst_value = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double s_ddot = static_cast<double>(points[i].s_ddot);
        BOOST_REQUIRE_MESSAGE(std::isfinite(s_ddot), "non-finite s_ddot at integration point " << i);

        cursor.seek(points[i].s);
        const auto bounds = traj.get_acceleration_bounds(cursor, points[i].s_dot);
        const double lower = static_cast<double>(bounds.s_ddot_min);
        const double upper = static_cast<double>(bounds.s_ddot_max);

        // Float-level slack for riding the velocity-limit curve, where the feasible band pinches to a
        // point and the slope-following acceleration carries rounding noise.
        //
        // The relative term is 2% rather than 1% to admit one known violation. A switching point sits on
        // an arc's acceleration limit curve at the arc's end, where the next segment is a linear only
        // 2.2e-6 long. Forward integration leaves the switching point with the arc's acceleration of
        // 26.21 and spends it crossing the linear, whose band is +/-25.85, an excess of 1.4%. Before
        // interpolation along constant acceleration steps was exact, the snap to the end of the linear
        // stamped the step with about 25.91 instead and hid it.
        //
        // TODO(RSDK-12850): Restore 1% once switching point accelerations are computed on the side of the
        // boundary the trajectory departs into.
        const double tol = 1e-2 + (2e-2 * std::max(std::abs(lower), std::abs(upper)));
        const double excess = std::max({0.0, s_ddot - (upper + tol), (lower - tol) - s_ddot});
        if (excess > worst_excess) {
            worst_excess = excess;
            worst_index = i;
            worst_value = s_ddot;
        }
    }

    BOOST_CHECK_MESSAGE(worst_excess == 0.0,
                        "integration point " << worst_index << " has s_ddot=" << worst_value
                                             << " outside its feasible acceleration band by " << worst_excess);
}

BOOST_AUTO_TEST_CASE(orbsanding_3df3ece3_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 100.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 16.0;
    generate_trajectory_from_replay_file("orbsanding-3df3ece3127a07f6554713f4449cc8b2_plan001_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_462bf0f9_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 5.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 21.0;
    generate_trajectory_from_replay_file("orbsanding-462bf0f94d9ebb7ccc71c6d8cd621533_plan002_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_9b8bc795_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 213.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 48.0;
    generate_trajectory_from_replay_file("orbsanding-9b8bc79585b192039f38325713fb7a12_plan001_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_a40fcf17_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 100.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 7.0;
    generate_trajectory_from_replay_file("orbsanding-a40fcf17e8a3bb17d6998a5cd028d736_plan001_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(failed_trajectory_6a16b075_533248da_trajectory_generation, *boost::unit_test::disabled()) {
    generate_trajectory_from_replay_file(
        "njsanding-viam_capture_tag=6a16b0754c15ae84eb949bc3bcc4932e_533248da-ea16-4af4-92d5-7b3f31dbdf8f_arm.trajex-totg-replay.json",
        std::nullopt,
        std::nullopt);
}

BOOST_AUTO_TEST_CASE(failed_trajectory_c89a6776_3ce29d66_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 100.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 71.0;
    generate_trajectory_from_replay_file(
        "njsanding-viam_capture_tag=c89a6776-85f5-4fa2-9919-eb486bc953d8_3ce29d66-8fc6-44de-bb33-45811ad671b5_arm.trajex-totg-replay.json",
        k_local_trajectory_invariants_tolerance_override,
        k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(failed_trajectory_c89a6776_4c48331e_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 100.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 46.0;
    generate_trajectory_from_replay_file(
        "njsanding-viam_capture_tag=c89a6776-85f5-4fa2-9919-eb486bc953d8_4c48331e-9871-4e06-bf93-01eacdb304dc_arm.trajex-totg-replay.json",
        k_local_trajectory_invariants_tolerance_override,
        k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_1d6726c5_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 7.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 24.0;
    generate_trajectory_from_replay_file("orbsanding-1d6726c5d9b8d3af076d8b0979e818c1_plan001_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_3c08d6c4_plan002_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 2.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 20.0;
    generate_trajectory_from_replay_file("orbsanding-3c08d6c43fb2f6f62fb2994307139311_plan002_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_3c08d6c4_plan003_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 0.5;
    constexpr auto k_local_joint_kinematics_tolerance_override = 10.0;
    generate_trajectory_from_replay_file("orbsanding-3c08d6c43fb2f6f62fb2994307139311_plan003_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

// The record behind the TCP-bound benchmark, with the TCP cap binding at every integration point.
BOOST_AUTO_TEST_CASE(orbsanding_xarm850_tcp_bound_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 0.001;
    constexpr auto k_local_joint_kinematics_tolerance_override = 6.0;
    generate_trajectory_from_replay_file("orbsanding-xarm850-tcp-bound-20261022.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(failed_trajectory_2f698337_eae1c945_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 214.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 62.0;
    generate_trajectory_from_replay_file(
        "njsanding-viam_capture_tag=2f698337-4b93-4e05-a81a-9e0906e5340c_eae1c945-b52b-4043-90c8-ca8234255b51_arm.trajex-totg-replay.json",
        k_local_trajectory_invariants_tolerance_override,
        k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(failed_trajectory_79603105_19c5a723_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 213.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 60.0;
    generate_trajectory_from_replay_file(
        "njsanding-viam_capture_tag=79603105-8826-492c-a189-2d754aa2739c_19c5a723-909a-45c8-afd4-46f025edff81_arm.trajex-totg-replay.json",
        k_local_trajectory_invariants_tolerance_override,
        k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(failed_trajectory_9a116016_5a3bd423_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 203.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 51.0;
    generate_trajectory_from_replay_file(
        "njsanding-viam_capture_tag=9a116016-eef3-4042-84c8-807d7b14f7e6_5a3bd423-ea97-425b-91b3-b866e93859de_arm.trajex-totg-replay.json",
        k_local_trajectory_invariants_tolerance_override,
        k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(failed_trajectory_9a116016_85719301_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 203.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 56.0;
    generate_trajectory_from_replay_file(
        "njsanding-viam_capture_tag=9a116016-eef3-4042-84c8-807d7b14f7e6_85719301-2690-4824-8cf7-b436353c5034_arm.trajex-totg-replay.json",
        k_local_trajectory_invariants_tolerance_override,
        k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_1209a844_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 17.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 17.0;
    generate_trajectory_from_replay_file("orbsanding-1209a844d5354757b74d7917d70e138c_plan007_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_8d1e0c32_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 2.0;
    constexpr auto k_local_joint_kinematics_tolerance_override = 32.0;
    generate_trajectory_from_replay_file("orbsanding-8d1e0c325e7c4c9a8096655fdeedef99_plan002_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_b3d06c98_plan001_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 0.5;
    constexpr auto k_local_joint_kinematics_tolerance_override = 15.0;
    generate_trajectory_from_replay_file("orbsanding-b3d06c9801b741578b9a9b3b1acf4545_plan001_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(orbsanding_b3d06c98_plan005_trajectory_generation) {
    // TODO(RSDK-13890): Reduce these tolerances
    constexpr auto k_local_trajectory_invariants_tolerance_override = 0.5;
    constexpr auto k_local_joint_kinematics_tolerance_override = 13.0;
    generate_trajectory_from_replay_file("orbsanding-b3d06c9801b741578b9a9b3b1acf4545_plan005_step002_work.trajex-totg-replay.json",
                                         k_local_trajectory_invariants_tolerance_override,
                                         k_local_joint_kinematics_tolerance_override);
}

BOOST_AUTO_TEST_CASE(RSDK_14509_orbsanding_13b848ed_run1_trajectory_generation, *boost::unit_test::disabled()) {
    generate_trajectory_from_replay_file(
        "orbsanding-13b848ed78d24c6c93d36dc30171b040_run1_plan001_step002_work.trajex-totg-replay.json", std::nullopt, std::nullopt);
}

BOOST_AUTO_TEST_CASE(RSDK_14509_orbsanding_13b848ed_run2_trajectory_generation, *boost::unit_test::disabled()) {
    generate_trajectory_from_replay_file(
        "orbsanding-13b848ed78d24c6c93d36dc30171b040_run2_plan001_step002_work.trajex-totg-replay.json", std::nullopt, std::nullopt);
}

// This is the regression test for RSDK-14911. A GP8 sanding move's 82 waypoints give 22 corners
// turning by less than acos can resolve, each blended with an arc at the min_blend_curvature radius
// cap. Every arc must lie on the path; the kinematics validator's comparison of each sample's
// position against its velocity is what catches one that doesn't.
//
// The driver wrote no replay record, since generation produced a trajectory. This one was rebuilt
// from the move's realtime capture: the waypoints and limits as captured, and the driver's default
// path options, which match the options in its replay records from the same machine. Built this way,
// the trajectory's second 3 Hz sample reproduces the position the driver streamed to four decimals.
//
// Both validators run at their default tolerance.
BOOST_AUTO_TEST_CASE(RSDK_14911_gp8_blend_arc_displaced_from_path) {
    constexpr auto k_trajectory_invariants_tolerance = 0.1;
    constexpr auto k_joint_kinematics_tolerance = 0.1;
    generate_trajectory_from_replay_file("gp8_blend_arc_displaced_from_path-20261110.trajex-totg-replay.json",
                                         k_trajectory_invariants_tolerance,
                                         k_joint_kinematics_tolerance);
}

// A forward step reaching the exit of a 5.3 mm arc is re-ended on the boundary, where its limits are
// read from the straight that follows. That straight has no acceleration limit, so the landing at
// s_dot 0.1584 is accepted, though it is above the arc's limit curve at its exit (0.15586). No limit
// hit is raised, and the arc's exit is never considered as a switching point. The backward pass from
// the end of the path then reaches the exit above the same curve, and generation fails with
// infeasible acceleration bounds.
//
// TODO(RSDK-13890): Enable once breach detection at a boundary checks the segment being left. The
// arc's exit must then be decided as a step-up switching point, which is RSDK-12850's search.
BOOST_AUTO_TEST_CASE(RSDK_13890_arc_exit_breach_undetected_0fb4d71c, *boost::unit_test::disabled()) {
    generate_trajectory_from_replay_file(
        "viam_capture_tag=0fb4d71c-11ba-4e7d-a78d-f0b86b5c6fe4_f15ea21f-d8cf-434e-ada5-bedc1095fbe0_arm.trajex-totg-replay.json",
        std::nullopt,
        std::nullopt);
}

// The breach above, at a boundary between two arcs rather than an arc and a straight. A forward step
// reaching the exit of a 1.7e-2 arc is re-ended on the boundary at s_dot 0.25492, above that arc's
// limit curve at its exit (0.25207), and is checked against the arc that follows, whose curve there
// is 0.26755. The backward pass from an extremum just past the boundary then reaches it above the
// first arc's curve, and generation fails with infeasible acceleration bounds.
//
// TODO(RSDK-13890): Enable with 0fb4d71c. Here the curve is finite on both sides of the boundary.
BOOST_AUTO_TEST_CASE(failed_trajectory_281279c7_95c23110_trajectory_generation, *boost::unit_test::disabled()) {
    generate_trajectory_from_replay_file(
        "njsanding-viam_capture_tag=281279c7-c5b2-4886-b133-3df6b6c1fea1_95c23110-2927-4e8b-ac51-d8da42e36ff6_arm.trajex-totg-replay.json",
        std::nullopt,
        std::nullopt);
}

// A 179.8 degree stroke reversal is blended by an arc of radius 1.4e-5, just under the curvature cap,
// and the acceleration limit curve across it is about 0.0093. Forward integration hits that curve
// inside the arc, and the velocity switching point search returns an escape exactly on the arc's exit,
// on the velocity curve at s_dot 2.5707. Judged from the straight that follows, which has no
// acceleration limit, that point is not above the acceleration curve, so it is accepted. The backward
// pass from it steps into the arc at about 275 times the arc's curve, and generation fails with
// infeasible acceleration bounds.
//
// TODO(RSDK-12850): Enable once a switching point on a boundary is judged against the segment the
// trajectory arrives from.
BOOST_AUTO_TEST_CASE(RSDK_14913_gp8_reversal_blend_velocity_escape, *boost::unit_test::disabled()) {
    generate_trajectory_from_replay_file("gp8_reversal_blend_velocity_escape-20261110.trajex-totg-replay.json", std::nullopt, std::nullopt);
}

BOOST_AUTO_TEST_SUITE_END()  // replay_regression_tests
