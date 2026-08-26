// Tests for viam::trajex::totg::streaming::session.

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/streaming/session.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/hertz.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using viam::trajex::totg::path;
using viam::trajex::totg::trajectory;
using viam::trajex::totg::waypoint_accumulator;
namespace streaming = viam::trajex::totg::streaming;
namespace types = viam::trajex::types;

// A small but real fixture: 2 DOF, modest velocity / acceleration limits, and a 100 Hz
// sample rate. The waypoint sets below produce trajectories long enough that pulling a
// dozen samples at 100 Hz stays well within the trajectory's duration.

constexpr double k_sample_rate_hz = 100.0;

types::hertz default_sample_rate() {
    return types::hertz{k_sample_rate_hz};
}

trajectory::options default_trajectory_options() {
    trajectory::options topt;
    topt.max_velocity = xt::xarray<double>{2.0, 2.0};
    topt.max_acceleration = xt::xarray<double>{5.0, 5.0};
    return topt;
}

path::options default_path_options() {
    path::options popt;
    popt.set_max_blend_deviation(0.05);
    return popt;
}

xt::xarray<double> three_waypoints() {
    return xt::xarray<double>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
}

xt::xarray<double> six_waypoints() {
    return xt::xarray<double>{
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {2.0, 1.0},
        {2.0, 2.0},
        {3.0, 2.0},
    };
}

// Builds a trajectory directly from waypoints with the same options the session uses.
// This is the reference any session sample stream should agree with where the active
// trajectory's geometry matches the full merged waypoint set.
trajectory reference_trajectory(const xt::xarray<double>& waypoints) {
    path p = path::create(waypoints, default_path_options());
    return trajectory::create(std::move(p), default_trajectory_options());
}

// Pins a waypoints xarray and a waypoint_accumulator over it together so the pair can be
// handed to session.extend() in a single expression without lifetime hazards.
//
// waypoint_accumulator holds views into its source xarray and explicitly deletes the
// rvalue-source constructor; ad-hoc factories returning an accumulator over a temporary
// xarray won't compile. This wrapper stores both pieces and pins itself.
//
// TODO(streaming-test-utils): hoist alongside other shared test helpers once a second
// streaming test file needs the same scaffolding. Until then, duplication here is cheap.
class pinned_waypoints {
   public:
    explicit pinned_waypoints(xt::xarray<double> data) : data_(std::move(data)), accumulator_(data_) {}

    pinned_waypoints(const pinned_waypoints&) = delete;
    pinned_waypoints& operator=(const pinned_waypoints&) = delete;
    pinned_waypoints(pinned_waypoints&&) = delete;
    pinned_waypoints& operator=(pinned_waypoints&&) = delete;

    const waypoint_accumulator& accumulator() const noexcept {
        return accumulator_;
    }
    const xt::xarray<double>& data() const noexcept {
        return data_;
    }

   private:
    xt::xarray<double> data_;
    waypoint_accumulator accumulator_;
};

// Returns true if every element of `a` is within `tolerance` of `b`. Used for direct
// configuration / velocity / acceleration comparisons between session samples and
// reference trajectory samples.
bool configs_match(const xt::xarray<double>& a, const xt::xarray<double>& b, double tolerance = 1e-9) {
    if (a.shape(0) != b.shape(0)) {
        return false;
    }
    for (std::size_t i = 0; i < a.shape(0); ++i) {
        if (std::abs(a(i) - b(i)) > tolerance) {
            return false;
        }
    }
    return true;
}

// Helper for the cornerstone equivalence test: verify that every sample the session
// emitted agrees with what a directly-built reference trajectory would produce at the
// same time. The reference is queried via `trajectory::sample(t)` which is random-access.
void check_samples_match_reference(const std::vector<struct trajectory::sample>& session_samples, const trajectory& reference) {
    for (const auto& s : session_samples) {
        if (s.time < trajectory::seconds{0.0} || s.time > reference.duration()) {
            continue;  // sample is past the reference's coverage; nothing to compare against
        }
        const auto expected = reference.sample(s.time);
        BOOST_CHECK(configs_match(s.configuration, expected.configuration));
        BOOST_CHECK(configs_match(s.velocity, expected.velocity));
        BOOST_CHECK(configs_match(s.acceleration, expected.acceleration));
    }
}

streaming::session fresh_session() {
    return streaming::session{default_path_options(), default_trajectory_options(), default_sample_rate()};
}

}  // namespace

BOOST_AUTO_TEST_SUITE(streaming_session_tests)

BOOST_AUTO_TEST_SUITE(empty_session)

BOOST_AUTO_TEST_CASE(fresh_session_has_zero_current_time) {
    auto sess = fresh_session();
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
}

BOOST_AUTO_TEST_CASE(fresh_session_has_no_active_trajectory) {
    auto sess = fresh_session();
    BOOST_CHECK(sess.active_trajectory() == nullptr);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 0U);
}

BOOST_AUTO_TEST_CASE(fresh_session_sample_next_returns_empty) {
    auto sess = fresh_session();
    const auto samples = sess.sample_next(5);
    BOOST_CHECK(samples.empty());
}

BOOST_AUTO_TEST_CASE(fresh_session_sample_at_least_returns_empty) {
    auto sess = fresh_session();
    const auto samples = sess.sample_at_least(trajectory::seconds{1.0});
    BOOST_CHECK(samples.empty());
}

BOOST_AUTO_TEST_SUITE_END()  // empty_session

BOOST_AUTO_TEST_SUITE(first_extend)

BOOST_AUTO_TEST_CASE(first_extend_with_valid_batch_creates_active_trajectory) {
    auto sess = fresh_session();
    const pinned_waypoints wp(three_waypoints());

    sess.extend(wp.accumulator());

    BOOST_CHECK(sess.active_trajectory() != nullptr);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);
}

BOOST_AUTO_TEST_CASE(first_extend_with_single_waypoint_propagates_invalid_argument) {
    auto sess = fresh_session();
    const pinned_waypoints wp(xt::xarray<double>{{0.0, 0.0}});

    BOOST_CHECK_THROW(sess.extend(wp.accumulator()), std::invalid_argument);

    // Failed extend leaves the session unchanged.
    BOOST_CHECK(sess.active_trajectory() == nullptr);
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 0U);
}

BOOST_AUTO_TEST_CASE(first_extend_with_dof_mismatch_against_options_propagates_invalid_argument) {
    // Default options have 2-DOF velocity / acceleration limits; provide 3-DOF waypoints
    // so that trajectory construction rejects the result.
    auto sess = fresh_session();
    const pinned_waypoints wp(xt::xarray<double>{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}});

    BOOST_CHECK_THROW(sess.extend(wp.accumulator()), std::invalid_argument);

    BOOST_CHECK(sess.active_trajectory() == nullptr);
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 0U);
}

BOOST_AUTO_TEST_SUITE_END()  // first_extend

BOOST_AUTO_TEST_SUITE(sampling_primitives)

BOOST_AUTO_TEST_CASE(sample_next_default_emits_one_sample) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto samples = sess.sample_next();
    BOOST_CHECK_EQUAL(samples.size(), 1U);
}

BOOST_AUTO_TEST_CASE(sample_next_n_emits_n_samples) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto samples = sess.sample_next(10);
    BOOST_CHECK_EQUAL(samples.size(), 10U);
}

BOOST_AUTO_TEST_CASE(sample_at_least_advances_at_least_horizon) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto horizon = trajectory::seconds{0.1};  // 100 ms
    const auto samples = sess.sample_at_least(horizon);
    BOOST_REQUIRE(!samples.empty());
    BOOST_CHECK_GE(sess.current_time().count(), horizon.count());
}

BOOST_AUTO_TEST_CASE(sample_at_least_zero_horizon_returns_exactly_one_sample) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    // Zero horizon: the first sample's time equals current_time + dt > current_time + 0,
    // so the stopping condition is satisfied after exactly one sample is emitted.
    const auto samples = sess.sample_at_least(trajectory::seconds{0.0});
    BOOST_CHECK_EQUAL(samples.size(), 1U);
}

BOOST_AUTO_TEST_CASE(current_time_tracks_last_emitted_sample) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto samples = sess.sample_next(7);
    BOOST_REQUIRE_EQUAL(samples.size(), 7U);
    BOOST_CHECK_EQUAL(sess.current_time().count(), samples.back().time.count());
}

BOOST_AUTO_TEST_SUITE_END()  // sampling_primitives

BOOST_AUTO_TEST_SUITE(single_trajectory_equivalence)

BOOST_AUTO_TEST_CASE(session_with_one_extend_matches_direct_trajectory) {
    auto sess = fresh_session();
    const auto waypoints = six_waypoints();
    const pinned_waypoints wp(waypoints);
    sess.extend(wp.accumulator());

    const auto reference = reference_trajectory(waypoints);
    const auto samples = sess.sample_at_least(reference.duration());
    BOOST_REQUIRE(!samples.empty());

    check_samples_match_reference(samples, reference);
}

BOOST_AUTO_TEST_SUITE_END()  // single_trajectory_equivalence

BOOST_AUTO_TEST_SUITE(seam_validation)

BOOST_AUTO_TEST_CASE(second_extend_with_seam_mismatch_throws_invalid_argument) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    // The last stored waypoint is {1.0, 1.0}. Provide a batch whose first waypoint differs.
    const pinned_waypoints mismatched(xt::xarray<double>{{9.0, 9.0}, {2.0, 2.0}});
    BOOST_CHECK_THROW(sess.extend(mismatched.accumulator()), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(second_extend_with_dof_mismatch_throws_invalid_argument) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    const pinned_waypoints wrong_dof(xt::xarray<double>{{1.0, 1.0, 0.0}, {2.0, 2.0, 0.0}});
    BOOST_CHECK_THROW(sess.extend(wrong_dof.accumulator()), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(second_extend_with_bit_exact_seam_matches_merged_reference) {
    // The merged waypoint set after seam-drop is {{0,0},{1,0},{1,1},{2,1},{2,2}}.
    // The sample stream after both extends should agree with a reference trajectory
    // built directly over that merged set.
    auto sess = fresh_session();
    const pinned_waypoints initial(xt::xarray<double>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}});
    sess.extend(initial.accumulator());

    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    const xt::xarray<double> merged{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}};
    const auto reference = reference_trajectory(merged);

    const auto samples = sess.sample_at_least(reference.duration());
    BOOST_REQUIRE(!samples.empty());

    check_samples_match_reference(samples, reference);
}

BOOST_AUTO_TEST_SUITE_END()  // seam_validation

BOOST_AUTO_TEST_SUITE(pivot)

BOOST_AUTO_TEST_CASE(extend_with_branch_ahead_of_watermark_pivots) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    // One sample's worth of watermark advancement: far below where the branch will lie
    // (the branch sits near the prefix's terminal blend, which is most of a trajectory away).
    sess.sample_next(1);

    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // Generation incremented: a new active trajectory was produced (pivot).
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK(sess.active_trajectory() != nullptr);
}

BOOST_AUTO_TEST_CASE(pivot_preserves_active_epoch) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(1);

    const auto pre_extend_epoch = sess.active_epoch();
    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // Confirm a pivot actually happened, then assert epoch is preserved across it.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), pre_extend_epoch.count());
}

BOOST_AUTO_TEST_CASE(pivot_preserves_current_time) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(3);

    const auto pre_extend_time = sess.current_time();
    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // Confirm a pivot actually happened, then assert current_time is preserved across it.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_EQUAL(sess.current_time().count(), pre_extend_time.count());
}

BOOST_AUTO_TEST_CASE(pivot_whose_resume_offset_overshoots_candidate_stages) {
    // A pivot resumes its new sampler one sample period past the last emitted sample. If the
    // candidate has less than one sample period of trajectory left after the branch, that
    // offset lands at or past the candidate's duration, and without a guard
    // quantized_for_trajectory throws (start >= duration). The right behavior is to stage
    // instead: with the branch that close to the end, a pivot would gain essentially no new
    // samples to emit, so staging (and then draining and rebasing) is correct.
    //
    // In production this arises at ordinary sample rates through corner-cutting. The active's
    // last waypoint is a hard endpoint (a full stop), but in the candidate that same waypoint
    // becomes interior and gets a circular blend, so the branch sits within one sample period
    // of the candidate's end. That exact geometry is fragile to reproduce deterministically,
    // so we force the same inequality with a slow sample rate instead: a small appended tail
    // leaves less than one (large) sample period of trajectory after the branch.
    //
    // To set it up, we park the watermark at D_act/2 (the middle of a three-sample grid, and
    // comfortably ahead of the branch, so the extend is admitted as a pivot) and choose the
    // period so the resume offset (D_act/2 + period) overshoots the candidate's duration by a
    // fixed margin. Setting period = D_cand - D_act/2 + margin makes that offset D_cand +
    // margin regardless of the actual durations.
    const auto d_act = reference_trajectory(three_waypoints()).duration();
    const xt::xarray<double> merged{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {1.0, 1.05}};
    const auto d_cand = reference_trajectory(merged).duration();

    constexpr double k_margin_sec = 0.05;
    const double period = d_cand.count() - (d_act.count() / 2.0) + k_margin_sec;
    // period lands in [D_act/2, D_act), so quantized_for_trajectory builds a three-sample grid
    // at { 0, D_act/2, D_act } and sample_next(2) parks the watermark exactly at D_act/2.
    streaming::session sess{default_path_options(), default_trajectory_options(), types::hertz{1.0 / period}};

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(2);

    // Precondition: we are set up so the pivot's resume offset overshoots the candidate.
    BOOST_REQUIRE_GE(sess.current_time().count() + period, d_cand.count());

    // The tiny appended tail would pivot (the branch is ahead of the watermark), but the
    // resume offset overshoots, so the session must stage instead of throwing.
    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {1.0, 1.05}});
    BOOST_CHECK_NO_THROW(sess.extend(extension.accumulator()));
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);
}

BOOST_AUTO_TEST_CASE(overshoot_stage_then_drain_consumes_the_staged_batch) {
    // Continuation of the staging test above. Once the overshoot guard declines the pivot and
    // stages the tiny post-branch tail, draining the session has to fold that tail in: its
    // waypoints are valid, reachable motion and must never be dropped. The rebuild (the active
    // terminal waypoint plus the staged tail) is a legitimate trajectory that just runs shorter
    // than one (slow) sample period, so rebase_'s resume-at-sample_period_ offset overshoots
    // and quantized_for_trajectory throws (start >= duration) inside sample_one_. The fix
    // handles this the same way the pivot side does: for a rebuild shorter than one sample
    // period, clamp the resume so the batch's terminal is emitted and the rebase completes.
    // This test asserts the batch is consumed; against the unfixed rebase it fails, because the
    // drain throws instead of delivering the endpoint.
    //
    // Construction mirrors the staging test above.
    const auto d_act = reference_trajectory(three_waypoints()).duration();
    const xt::xarray<double> merged{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {1.0, 1.05}};
    const auto d_cand = reference_trajectory(merged).duration();

    constexpr double k_margin_sec = 0.05;
    const double period = d_cand.count() - (d_act.count() / 2.0) + k_margin_sec;
    streaming::session sess{default_path_options(), default_trajectory_options(), types::hertz{1.0 / period}};

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(2);  // park the watermark at D_act/2: mid-grid and ahead of the branch

    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {1.0, 1.05}});
    sess.extend(extension.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // staged, not pivoted

    // Precondition: the rebuild rebase_ will attempt ({active terminal waypoint} + {staged
    // tail}) is shorter than one sample period, so the unfixed resume-at-sample_period_ offset
    // overshoots. This is the regime the fix must handle by consuming the batch, not dropping.
    const auto d_rebuild = reference_trajectory(xt::xarray<double>{{1.0, 1.0}, {1.0, 1.05}}).duration();
    BOOST_REQUIRE_GE(period, d_rebuild.count());

    // Draining past the active's terminal triggers the rebase. It must consume the staged tail,
    // neither throwing nor dropping it.
    std::vector<struct trajectory::sample> drained;
    BOOST_CHECK_NO_THROW(drained = sess.sample_next(8));

    // The rebase completed: a new trajectory was installed for the staged batch.
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);

    // The batch's motion was delivered: the last emitted sample reaches the final waypoint.
    BOOST_REQUIRE(!drained.empty());
    const auto& terminal = drained.back();
    const xt::xarray<double> final_waypoint{1.0, 1.05};
    BOOST_CHECK(configs_match(terminal.configuration, final_waypoint, 1e-3));

    // The short-rebuild terminal is a true rest-to-rest endpoint, so it has zero velocity and
    // acceleration, and it lands exactly at the rebased trajectory's end in global time (the
    // epoch, advanced by the old duration, plus the short rebuild's own duration).
    BOOST_CHECK_EQUAL(terminal.time.count(), (sess.active_epoch() + sess.active_trajectory()->duration()).count());
    BOOST_REQUIRE_EQUAL(terminal.velocity.shape(0), 2U);
    BOOST_REQUIRE_EQUAL(terminal.acceleration.shape(0), 2U);
    for (std::size_t i = 0; i < terminal.velocity.shape(0); ++i) {
        BOOST_CHECK_EQUAL(terminal.velocity(i), 0.0);
    }
    for (std::size_t i = 0; i < terminal.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(terminal.acceleration(i), 0.0);
    }

    // Having consumed the batch, the session is now cleanly drained.
    std::vector<struct trajectory::sample> tail;
    BOOST_CHECK_NO_THROW(tail = sess.sample_next(1));
    BOOST_CHECK(tail.empty());
}

BOOST_AUTO_TEST_SUITE_END()  // pivot

BOOST_AUTO_TEST_SUITE(stage_and_rebase)

BOOST_AUTO_TEST_CASE(extend_with_branch_behind_watermark_stages) {
    // Sampling all the way to the active trajectory's terminal pushes the watermark
    // past where the divergence between the initial and merged trajectories sits, so
    // the second extend must stage rather than pivot.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    const auto* initial_active = sess.active_trajectory();
    BOOST_REQUIRE(initial_active != nullptr);
    sess.sample_at_least(initial_active->duration());

    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // Stage: no new trajectory became active, so the generation count is unchanged.
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), 0.0);
}

BOOST_AUTO_TEST_CASE(staged_batch_rebases_when_sampling_past_terminal) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    const auto initial_duration = sess.active_trajectory()->duration();

    sess.sample_at_least(initial_duration);  // exhaust the active before extending

    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // After extend: stage. Generation count is still 1.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    // Sampling further triggers the rebase from the original trajectory's terminal pose.
    sess.sample_next(1);

    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK(sess.active_trajectory() != nullptr);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), initial_duration.count());
}

BOOST_AUTO_TEST_CASE(rebase_seam_configuration_is_continuous) {
    // The last sample of the original chain and the first sample of the rebased chain should
    // report the same joint configuration, since both correspond to the original trajectory's
    // terminal pose by the rest-to-rest invariant.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    const auto initial_duration = sess.active_trajectory()->duration();
    // Capture the terminal pose directly from the trajectory the session is about to leave.
    const auto terminal_sample = sess.active_trajectory()->sample(initial_duration);

    sess.sample_at_least(initial_duration);

    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    const auto post_rebase_samples = sess.sample_next(1);
    BOOST_REQUIRE_EQUAL(post_rebase_samples.size(), 1U);
    // Confirm rebase actually happened.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);

    // The first post-rebase sample lives exactly one sample period into the new trajectory
    // by construction of quantized_starting_at(new_active, rate, sample_period_). So its
    // configuration differs from the terminal pose by the motion the trajectory plans over
    // one sample period starting from rest: bounded above by 0.5 * max_accel * sample_period_^2
    // = 0.5 * 5.0 * 0.01^2 = 2.5e-4 rad per joint, with blend curvature potentially adding
    // a bit. 1e-3 leaves an order of magnitude of margin; tighter would be brittle.
    BOOST_CHECK(configs_match(post_rebase_samples.front().configuration, terminal_sample.configuration, 1e-3));
}

BOOST_AUTO_TEST_CASE(rebase_seam_time_keeps_flowing_forward) {
    // Across a rebase the epoch advances and the new active trajectory has a fresh
    // local-time origin. The session must add the new epoch when reporting sample
    // times so the global clock keeps moving forward rather than restarting near zero.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    const auto initial_duration = sess.active_trajectory()->duration();
    sess.sample_at_least(initial_duration);
    const auto pre_rebase_time = sess.current_time();
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);
    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());
    const auto post_rebase_samples = sess.sample_next(1);
    BOOST_REQUIRE_EQUAL(post_rebase_samples.size(), 1U);
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_GT(post_rebase_samples.front().time.count(), pre_rebase_time.count());
}

BOOST_AUTO_TEST_CASE(staged_batch_that_fails_to_build_surfaces_as_error_at_rebase) {
    // A staged batch whose geometry cannot build a path must surface as a hard error when the
    // rebase fires: this is robot motion, so a bad batch is never silently dropped nor masked
    // as a clean drain. Duplicate consecutive waypoints yield a zero-length linear segment,
    // which path::create rejects (path.cpp:77). We disable linear coalescing
    // (max_linear_deviation = 0) so the duplicate is not quietly removed before it can throw.
    //
    // To make the failure land at the rebase rather than at the extend, the bad batch must
    // arrive while the session is already locked out (staging non-empty): a locked-out extend
    // only records the batch, deferring the build to rebase_.
    path::options popt = default_path_options();
    popt.set_max_linear_deviation(0.0);
    streaming::session sess{popt, default_trajectory_options(), default_sample_rate()};

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    // Drain to the terminal so the next extend stages (branch behind the watermark), locking out.
    sess.sample_at_least(sess.active_trajectory()->duration());

    const pinned_waypoints good(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}});
    sess.extend(good.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // staged, locked out

    // Locked out: this batch is recorded, not built. Its tail {2,1} duplicates the prior staged
    // waypoint, so the eventual rebuild {1,1},{2,1},{2,1} carries a zero-length segment.
    const pinned_waypoints bad(xt::xarray<double>{{2.0, 1.0}, {2.0, 1.0}});
    BOOST_REQUIRE_NO_THROW(sess.extend(bad.accumulator()));  // locked-out extend defers the build

    // Draining fires the rebase, whose build fails. It must surface as an exception (the C ABI
    // maps this to an error return), not silently vanish.
    BOOST_CHECK_THROW(sess.sample_next(4), std::invalid_argument);

    // State is intact: the failed rebase installed no new trajectory (rebase_ mutates only after
    // the throwing build). And the batch is retained, not dropped, so a subsequent pull
    // re-attempts and reports the same error again. That is correct: a bad batch is a persistent,
    // honestly reported failure, never a silently dropped one.
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);
    BOOST_CHECK_THROW(sess.sample_next(1), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()  // stage_and_rebase

BOOST_AUTO_TEST_SUITE(multi_extend)

BOOST_AUTO_TEST_CASE(repeated_admissible_extends_compose_into_long_trajectory) {
    // Three extends in a row, each issued while the watermark is at zero (so each pivots).
    // The resulting sample stream should agree with a direct trajectory built over the
    // fully merged waypoint set.
    auto sess = fresh_session();

    const pinned_waypoints batch_1(xt::xarray<double>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}});
    sess.extend(batch_1.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    const pinned_waypoints batch_2(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(batch_2.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);

    const pinned_waypoints batch_3(xt::xarray<double>{{2.0, 2.0}, {3.0, 2.0}});
    sess.extend(batch_3.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 3U);

    const xt::xarray<double> merged{
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {2.0, 1.0},
        {2.0, 2.0},
        {3.0, 2.0},
    };
    const auto reference = reference_trajectory(merged);

    const auto samples = sess.sample_at_least(reference.duration());
    BOOST_REQUIRE(!samples.empty());

    check_samples_match_reference(samples, reference);
}

BOOST_AUTO_TEST_CASE(mixed_pivot_and_stage_eventually_drains_all_input) {
    // Pivot once, then sample to terminal to force the next extend to stage, then sample
    // past terminal to rebase. Generation count progression: 1 (initial), 2 (pivot),
    // 2 (stage, no new active), 3 (rebase).
    auto sess = fresh_session();

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    // Sample one tick, then extend (a pivot).
    sess.sample_next(1);
    const pinned_waypoints pivot_batch(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}});
    sess.extend(pivot_batch.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);

    // Sample past where the next extend's branch will lie, forcing it to stage.
    const auto duration_before_stage = sess.active_trajectory()->duration();
    sess.sample_at_least(duration_before_stage);
    const pinned_waypoints stage_batch(xt::xarray<double>{{2.0, 1.0}, {2.0, 2.0}});
    sess.extend(stage_batch.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);  // staged, not pivoted

    // Sample further to trigger the rebase.
    sess.sample_next(1);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 3U);
    BOOST_CHECK(sess.active_trajectory() != nullptr);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), duration_before_stage.count());
}

BOOST_AUTO_TEST_CASE(multi_batch_staging_accumulates_into_single_rebase) {
    // While locked out, several extends accumulate into staged_batches_ and a single rebase
    // folds them all into one rebuilt chain. Every other rebase test stages exactly one batch,
    // which leaves the multi-element accumulation path unexercised (stack_anchor_and_staged_
    // over more than one block, and the locked-out append firing repeatedly). It also backs the
    // reasoning that more input arriving before a drain keeps us on the normal, non-short rebase
    // path, which only holds if multiple staged batches merge correctly.
    auto sess = fresh_session();

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    const auto initial_duration = sess.active_trajectory()->duration();

    // Drain to the terminal so the next extend stages (branch behind the watermark), locking out.
    sess.sample_at_least(initial_duration);

    const pinned_waypoints batch_a(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(batch_a.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // staged, locked out

    // Second extend arrives while locked out: it accumulates onto staging rather than rebasing.
    const pinned_waypoints batch_b(xt::xarray<double>{{2.0, 2.0}, {3.0, 2.0}});
    sess.extend(batch_b.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // still just accumulated

    // Draining fires a single rebase that folds both staged batches into one new chain.
    const auto drained = sess.sample_at_least(trajectory::seconds{1000.0});
    BOOST_REQUIRE(!drained.empty());
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);  // one rebase, not one-per-batch
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), initial_duration.count());

    // Both batches folded in: the terminal reaches batch_b's last waypoint {3,2} at rest. Had
    // accumulation dropped or misassembled batch_b, the terminal would be batch_a's last, {2,2}.
    const auto& terminal = drained.back();
    const xt::xarray<double> final_waypoint{3.0, 2.0};
    BOOST_CHECK(configs_match(terminal.configuration, final_waypoint, 1e-3));
    BOOST_REQUIRE_EQUAL(terminal.velocity.shape(0), 2U);
    BOOST_REQUIRE_EQUAL(terminal.acceleration.shape(0), 2U);
    for (std::size_t i = 0; i < terminal.velocity.shape(0); ++i) {
        BOOST_CHECK_EQUAL(terminal.velocity(i), 0.0);
    }
    for (std::size_t i = 0; i < terminal.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(terminal.acceleration(i), 0.0);
    }

    // TODO: strengthen this to a full geometry-equivalence check that verifies the post-rebase
    // sample stream matches a trajectory built directly over the merged waypoint set (the
    // terminal pose followed by the staged tails), rather than only checking that it reaches the
    // endpoint. That is deferred because it needs an epoch-aware comparison. Post-rebase samples
    // carry global timestamps advanced by the prior chain's duration, and
    // check_samples_match_reference skips any sample past the reference's local duration, so
    // reusing it as-is would silently skip every post-rebase sample and pass without checking
    // anything. The full check would compare reference.sample(sample.time - epoch) through a
    // small epoch-shifted variant of that helper. This minimal test covers the property that
    // matters: both batches fold into one rebase, and the terminal reaches the last staged
    // waypoint at rest.
}

BOOST_AUTO_TEST_SUITE_END()  // multi_extend

BOOST_AUTO_TEST_SUITE(end_of_stream)

BOOST_AUTO_TEST_CASE(sample_next_after_exhaustion_returns_empty) {
    auto sess = fresh_session();
    const pinned_waypoints wp(three_waypoints());
    sess.extend(wp.accumulator());

    const auto* active = sess.active_trajectory();
    BOOST_REQUIRE(active != nullptr);
    sess.sample_at_least(active->duration() * 2.0);  // sample well past terminal

    // No staging exists, so further pulls drain to empty.
    const auto samples = sess.sample_next(5);
    BOOST_CHECK(samples.empty());
}

BOOST_AUTO_TEST_CASE(extend_after_exhaustion_eventually_starts_new_chain) {
    // When the active is exhausted and no staging exists, an arriving extend stages.
    // The next sample then triggers a rebase from the terminal pose, producing a new
    // active trajectory and advancing the epoch.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    const auto initial_duration = sess.active_trajectory()->duration();

    sess.sample_at_least(initial_duration * 2.0);  // drain to empty

    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // Extend on an exhausted session: stages, no new active built yet.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    sess.sample_next(1);

    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK(sess.active_trajectory() != nullptr);
    BOOST_CHECK_GE(sess.active_epoch().count(), initial_duration.count());
}

BOOST_AUTO_TEST_SUITE_END()  // end_of_stream

// These tests pin down a property that uniform_sampler's quantized-for-duration mode is
// supposed to give us at the trajectory level, and that the session needs to preserve
// across rebases: the last emitted sample lands exactly at the active trajectory's
// terminal, with zero joint velocity and zero joint acceleration by the rest-to-rest
// invariant.
BOOST_AUTO_TEST_SUITE(terminal_sampling)

BOOST_AUTO_TEST_CASE(final_emitted_sample_in_single_trajectory_lies_at_terminal_at_rest) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto duration = sess.active_trajectory()->duration();
    const auto samples = sess.sample_at_least(duration);
    BOOST_REQUIRE(!samples.empty());

    const auto& last = samples.back();

    BOOST_CHECK_EQUAL(last.time.count(), duration.count());
    BOOST_REQUIRE_EQUAL(last.velocity.shape(0), 2U);
    BOOST_REQUIRE_EQUAL(last.acceleration.shape(0), 2U);
    for (std::size_t i = 0; i < last.velocity.shape(0); ++i) {
        BOOST_CHECK_EQUAL(last.velocity(i), 0.0);
    }
    for (std::size_t i = 0; i < last.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(last.acceleration(i), 0.0);
    }
}

BOOST_AUTO_TEST_CASE(final_emitted_sample_after_rebase_lies_at_rebased_terminal_at_rest) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    const auto initial_duration = sess.active_trajectory()->duration();
    sess.sample_at_least(initial_duration);  // drain the initial chain through its terminal

    // Stage an extension by extending while the watermark sits at the terminal.
    const pinned_waypoints extension(xt::xarray<double>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    // Drain the rest with a generous horizon to fire the rebase and run out the new chain.
    const auto post_rebase_samples = sess.sample_at_least(trajectory::seconds{1000.0});
    BOOST_REQUIRE(!post_rebase_samples.empty());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);

    const auto& last = post_rebase_samples.back();
    const auto rebased_duration = sess.active_trajectory()->duration();
    const auto rebased_epoch = sess.active_epoch();

    BOOST_CHECK_EQUAL(last.time.count(), (rebased_epoch + rebased_duration).count());
    BOOST_REQUIRE_EQUAL(last.velocity.shape(0), 2U);
    BOOST_REQUIRE_EQUAL(last.acceleration.shape(0), 2U);
    for (std::size_t i = 0; i < last.velocity.shape(0); ++i) {
        BOOST_CHECK_EQUAL(last.velocity(i), 0.0);
    }
    for (std::size_t i = 0; i < last.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(last.acceleration(i), 0.0);
    }
}

BOOST_AUTO_TEST_SUITE_END()  // terminal_sampling

BOOST_AUTO_TEST_SUITE_END()  // streaming_session_tests
