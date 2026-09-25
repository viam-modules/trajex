#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/streaming/private/waypoint_store.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/hertz.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg::streaming {

///
/// Streaming-input, streaming-output trajectory execution session.
///
/// Holds an active trajectory that grows as new waypoint batches arrive while
/// sampling proceeds. Each `extend()` call may either pivot the active trajectory
/// to a new one that incorporates the additional waypoints, or stage the batch
/// for later if the branch between the old and new trajectories lies at or behind
/// the latest emitted sample. Staged batches are absorbed into a new trajectory
/// built from the current trajectory's terminal pose once that trajectory has been
/// sampled through.
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
    /// What one `extend()` call did with the batch it was given, and the timing it computed
    /// along the way.
    ///
    /// Both times are optional because they only exist on some paths through `extend()`. A
    /// branch slack needs a candidate trajectory to compare against the active one, which a
    /// seam-only call, or one arriving when batches are already staged, never builds. A
    /// duration delta needs a trajectory to have been installed, which staging by definition
    /// does not do. A pivot is the only kind that reports both.
    ///
    /// `branch_slack` is measured from the most recently emitted sample to the branch: the
    /// point at which the candidate first stops agreeing with the active trajectory, both
    /// expressed in global time. Positive means the branch was still ahead of everything
    /// handed out, and the call beat the deadline by that much. Negative means it sat in the
    /// already-emitted past, which is what forces a stage, and the magnitude is how much
    /// earlier the call needed to happen. The comparison is against what the session has
    /// emitted, not what the arm has executed, so a caller that pulls samples far ahead of
    /// execution spends its own slack doing so. It is present for `k_pivot`,
    /// `k_staged_branch_sampled` and `k_staged_unsamplable`.
    ///
    /// `delta_active_duration` compares the newly installed trajectory's duration against
    /// that of the trajectory it replaced. Weighed against the interval between calls, it
    /// says whether the caller is adding motion faster than sampling consumes it. It is
    /// signed rather than unsigned because the replacement no longer has to stop at the old
    /// terminal waypoint and so covers the shared part of the path faster than its
    /// predecessor did; that saving is normally smaller than the motion being added, but
    /// nothing guarantees it. It is present for `k_pivot`, and for `k_first_build`, where it
    /// is the whole of the new trajectory's duration.
    ///
    struct extend_result {
        ///
        /// How `extend()` handled a batch.
        ///
        /// A batch either builds the session's first trajectory, replaces the active
        /// trajectory with one that incorporates it, or waits in staging until the active
        /// trajectory has been sampled through. The difference matters to a caller pacing
        /// its own sends: a pivot is invisible to the arm, but every trajectory ends at
        /// rest, so a stage means the active trajectory will run to its end and bring the
        /// arm to a stop before the staged motion begins.
        ///
        /// The two values for a stage that followed a comparison distinguish the reasons a
        /// pivot was refused. One says the call arrived after the point it needed to change
        /// had already been handed out; the other says it arrived in time but carried less
        /// than one sample period of motion. The remedies differ, so the values do too.
        ///
        /// The integer values are pinned because the C ABI mirrors them.
        ///
        enum class kinds : std::uint8_t {
            k_first_build = 0,            ///< Built the session's first trajectory
            k_pivot = 1,                  ///< Replaced the active trajectory; sampling continues unbroken
            k_staged_branch_sampled = 2,  ///< Staged; sampling had already passed the branch
            k_staged_unsamplable = 3,     ///< Staged; less than one sample period of motion added
            k_staged_again = 4,           ///< Staged; batches were already staged, so nothing was compared
            k_noop = 5,                   ///< Nothing beyond the seam waypoint; session unchanged
        };

        kinds kind;                                                ///< How the batch was handled
        std::optional<trajectory::seconds> branch_slack;           ///< Time by which the branch beat the last sample
        std::optional<trajectory::seconds> delta_active_duration;  ///< Growth over the trajectory it replaced
    };

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
    /// (a pivot) or held aside for later (a stage). The returned `extend_result` says
    /// which, along with the timing a caller needs in order to pace its own sends; a
    /// caller with no interest in either may discard it.
    ///
    /// Waypoints in `batch` are assumed to have been deduplicated by the caller. The
    /// bit-exact seam requirement means the merged sequence retains the dedup invariant
    /// after the seam point is dropped.
    ///
    /// @param batch Waypoints to append
    /// @return How the batch was handled, and the timing that went with it
    /// @throws std::invalid_argument if `batch`'s DOF disagrees with the session's existing
    ///         waypoint DOF, or if its first waypoint does not equal the session's last
    /// @throws Any exception raised by trajectory construction if computing the updated
    ///         trajectory fails. Session state is unchanged in that case.
    ///
    extend_result extend(const waypoint_accumulator& batch);

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
    /// Returns how much of the active trajectory has not yet been sampled.
    ///
    /// This is the active trajectory's end in global time less the time of the most
    /// recently emitted sample, and it is clamped at zero rather than allowed to go
    /// slightly negative when the last sample lands on the trajectory's end.
    ///
    /// It counts only the active trajectory. Motion sitting in staged batches has no
    /// trajectory yet, and so has no duration to report, which means this value drains
    /// toward zero while batches are staged even though the session still has work
    /// queued, and then jumps back up when the rebase builds a trajectory for that work.
    /// A caller pacing itself against this number needs to know that. Reporting a true
    /// session-wide total has to wait until staged batches carry timing of their own.
    ///
    /// @return Unsampled time left in the active trajectory, or zero if there is none
    ///
    trajectory::seconds remaining_active_duration() const noexcept;

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
    // Builds a trajectory from the given waypoints, threading through path::options and
    // trajectory::options. Throws on validation failure inside path::create or
    // trajectory::create, leaving every member it does not touch alone; callers that have
    // already appended to `waypoints_` are responsible for winding that back.
    trajectory build_trajectory_from_(const waypoint_accumulator& waypoints) const;

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

    // The waypoint set that built `active_`, owned by the session because the accumulators
    // callers pass to `extend` view memory the session does not control. Empty until the
    // first successful extend.
    //
    // The store cannot be moved, which makes a session non-movable as well. That was
    // already true in substance: `cursor_` below holds a pointer into `active_`, so moving
    // a session would leave it pointing at the old location.
    waypoint_store waypoints_;

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

    // Batches received while staging, each pre-stripped of its seam point. Drained
    // into the new active during the next rebase.
    std::vector<xmatrix<>> staged_batches_;

    // The most recently received waypoint, against which the next extend's seam is
    // bit-exactly validated. Empty (shape (0,)) before the first extend.
    xvector<> last_waypoint_;

    // Cumulative count of trajectories the session has installed as active. Increments
    // on first build, on each pivot, and on each rebase.
    std::size_t generation_count_{0};
};

}  // namespace viam::trajex::totg::streaming
