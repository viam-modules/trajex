#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/hertz.hpp>

namespace viam::trajex::totg::streaming {

///
/// Streaming-input, streaming-output trajectory execution session.
///
/// Holds an active trajectory that grows as new waypoint batches arrive while
/// sampling proceeds. Each `extend()` call may either pivot the active trajectory
/// to a new one that incorporates the additional waypoints, or stage the batch
/// for later if the time of divergence between the old and new trajectories falls
/// behind the latest emitted sample. Staged batches are absorbed into a new
/// trajectory built from the current trajectory's terminal pose once that
/// trajectory has been sampled through.
///
/// Sampling is forward-only and stateful: each call to `sample_next()` or
/// `sample_at_least()` advances an internal cursor, and how far that cursor has
/// advanced determines whether a later extend can pivot or must stage. The session
/// assumes single-threaded ownership; sampling and extending from different threads
/// is unsupported.
///
class session {
   public:
    ///
    /// Constructs a session with the parameters used to build each trajectory and the
    /// sample rate at which samples will be emitted.
    ///
    /// No trajectory exists until the first call to `extend()`.
    ///
    /// @param path_options Path-construction options (used for every trajectory built by the session)
    /// @param trajectory_options Trajectory-construction options (used for every trajectory built by the session)
    /// @param sample_rate Nominal sample rate. Each underlying trajectory's sampler is
    ///                    quantized to land its last sample exactly on the trajectory's
    ///                    duration, so per-sample spacing approximates 1 / sample_rate
    ///                    with small per-trajectory drift. This parameter's shape may
    ///                    change if a sampler factory is added later.
    ///
    session(path::options path_options, trajectory::options trajectory_options, types::hertz sample_rate);

    ///
    /// Adds a batch of waypoints to the session.
    ///
    /// If no active trajectory exists, builds the initial one from `batch`.
    /// Otherwise, requires `batch`'s first waypoint to compare bit-exactly equal to the
    /// session's most recently stored waypoint, then absorbs the remainder of `batch` and
    /// attempts to build a trajectory incorporating it. The result is either swapped in
    /// (a pivot) or held aside for later (a stage); which one happens is not visible to
    /// the caller and does not need to be.
    ///
    /// Waypoints in `batch` are assumed to have been deduplicated by the caller. The
    /// bit-exact seam requirement means the merged sequence retains the dedup invariant
    /// after the seam point is dropped.
    ///
    /// @param batch Waypoints to append
    /// @throws std::invalid_argument if `batch`'s DOF disagrees with the session's existing
    ///         waypoint DOF, or if its first waypoint does not equal the session's last
    /// @throws Any exception raised by trajectory construction if computing the updated
    ///         trajectory fails. Session state is unchanged in that case.
    ///
    void extend(const waypoint_accumulator& batch);

    ///
    /// Returns the global time of the most recently emitted sample, or zero if no samples
    /// have been emitted yet.
    ///
    /// "Global time" is measured from the start of the session and runs continuously across
    /// pivots and rebases.
    ///
    /// @return Time of the most recently emitted sample
    ///
    trajectory::seconds current_time() const noexcept;

    ///
    /// Pulls the next `n` samples from the session, advancing the sampling cursor.
    ///
    /// What "next" means is sampler-defined; for the current uniform sampler, samples are
    /// spaced according to the session's sample rate. Returns fewer than `n` samples if the
    /// session is exhausted (active trajectory ran out and no staged batches were available
    /// to rebase onto).
    ///
    /// @param n Number of samples to attempt to produce. Defaults to 1.
    /// @return Vector of up to `n` samples
    ///
    std::vector<struct trajectory::sample> sample_next(std::size_t n = 1);

    ///
    /// Pulls samples until the most recent sample's time is at least
    /// `current_time() + horizon`, advancing the sampling cursor accordingly.
    ///
    /// Returns fewer (possibly zero) samples than that target if the session is exhausted.
    /// The name says `at_least` because a non-uniform sampler may overshoot the requested
    /// horizon by a bounded amount; the session does not split a sample period.
    ///
    /// @param horizon Minimum amount of time to advance before stopping
    /// @return Vector of samples covering at least `horizon`, or fewer on exhaustion
    ///
    std::vector<struct trajectory::sample> sample_at_least(trajectory::seconds horizon);

    ///
    /// Returns a pointer to the active trajectory, or null if none has been built yet.
    ///
    /// @note This is an internal implementation detail exposed for testing. Production
    ///       callers should drive the session through `extend()` and the sampling
    ///       methods; reaching past those to the underlying trajectory is not part of
    ///       the supported usage pattern.
    /// @warning The returned pointer is invalidated by any mutating call on the session,
    ///          including `extend()`, `sample_next()`, and `sample_at_least()`, because
    ///          any of those may pivot or rebase the active trajectory. Do not hold the
    ///          pointer across any such call.
    /// @return Pointer to the active trajectory, or null if no trajectory has been built
    ///
    const trajectory* active_trajectory() const noexcept;

    ///
    /// Returns the global time at which the active trajectory's local t=0 sits.
    ///
    /// Pivots preserve the epoch; rebases advance it by the prior active trajectory's
    /// duration. Returns zero when no active trajectory exists.
    ///
    /// @note This is an internal implementation detail exposed for testing. Production
    ///       callers should not need to translate between local and global time;
    ///       sampling methods deliver samples in global time directly.
    /// @warning The returned value is invalidated by any mutating call on the session
    ///          (see `active_trajectory()`).
    /// @return Global time corresponding to the active trajectory's local origin
    ///
    trajectory::seconds active_epoch() const noexcept;

    ///
    /// Returns the cumulative number of trajectories the session has produced.
    ///
    /// Increments by one each time a new trajectory becomes active: at the first
    /// successful `extend()`, on each pivot, and on each rebase. Stays unchanged on
    /// stage (no new active is produced), on failed extends, and on sampling calls
    /// that do not cross a chain boundary. Returns zero for a fresh session.
    ///
    /// @note This is an internal implementation detail exposed for testing. The
    ///       counter exists so tests can witness pivot and rebase transitions
    ///       without relying on object-address comparisons of `active_trajectory()`,
    ///       which need not change across a transition.
    /// @return Number of trajectories the session has built
    ///
    std::size_t trajectory_generation_count() const noexcept;

   private:
    // Builds a trajectory from the given waypoints xarray, threading through path::options
    // and trajectory::options. Throws on validation failure inside path::create or
    // trajectory::create; the session's state is unaffected because this is called before
    // any member is mutated.
    trajectory build_trajectory_from_(const xt::xarray<double>& waypoints) const;

    // Emits a single sample, advancing the cursor. Triggers a rebase if the active is
    // exhausted at the next-sample index and staging is non-empty. Returns nullopt when
    // the session is fully drained.
    std::optional<struct trajectory::sample> sample_one_();

    // Rebuilds the active trajectory from {terminal_pose, ...staged_batches}, advances
    // the epoch by the prior active's duration, clears staging, and increments the
    // generation count. Preconditions: active_ holds a value, staged_batches_ is non-empty.
    void rebase_();

    // Construction-time configuration. Reused for every trajectory the session builds.
    path::options path_options_;
    trajectory::options trajectory_options_;
    types::hertz sample_rate_;

    // Nominal sample period, derived once from sample_rate_. Used to compute the
    // per-trajectory starting offset at pivot and rebase transitions.
    trajectory::seconds sample_period_;

    // The waypoint set that built `active_`. Stable owned storage required because
    // waypoint_accumulator stores row-views into the array it was constructed over.
    // Empty (shape (0,)) until the first successful extend; thereafter shape (N, dof).
    xt::xarray<double> active_waypoints_;

    // The currently active trajectory, or nullopt before the first successful extend.
    // Storage in std::optional is in-place, so `&*active_` is a stable address across
    // pivot and rebase (which both proceed by move-assigning a freshly-built trajectory
    // into this optional). Tests must use trajectory_generation_count() to witness
    // transitions instead of comparing pointers.
    std::optional<trajectory> active_;

    // Per-trajectory uniform sampler and cursor. Reconstructed at every transition
    // (first build, pivot, rebase) so each new active is sampled on a fresh grid
    // aligned to its own duration. Both reference active_; reconstruction order is
    // always (assign active_) -> (emplace sampler_/cursor_) so the cursor points at
    // the freshly-installed trajectory.
    std::optional<uniform_sampler> sampler_;
    std::optional<trajectory::cursor> cursor_;

    // Global time at which active_'s local t=0 sits. Pivots leave this unchanged; rebases
    // advance it by the prior active's duration.
    trajectory::seconds epoch_{0.0};

    // Global time of the most recently emitted sample, or zero if no sample has been
    // emitted yet. Cached for the current_time() accessor.
    trajectory::seconds current_time_{0.0};

    // Cumulative count of samples emitted. Used at pivot time to distinguish "no
    // samples yet" (start new sampler at offset 0) from "samples emitted" (start at
    // current local time + one sample period).
    std::size_t emitted_sample_count_{0};

    // Batches received while locked-out, each pre-stripped of its seam point. Drained
    // into the new active during the next rebase.
    std::vector<xt::xarray<double>> staged_batches_;

    // The most recently received waypoint, against which the next extend's seam is
    // bit-exactly validated. Empty (shape (0,)) before the first extend.
    xt::xarray<double> last_waypoint_;

    // Cumulative count of trajectories the session has installed as active. Increments
    // on first build, on each pivot, and on each rebase.
    std::size_t generation_count_{0};
};

}  // namespace viam::trajex::totg::streaming
