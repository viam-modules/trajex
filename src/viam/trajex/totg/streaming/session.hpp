#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <viam/trajex/totg/export.hpp>
#include <viam/trajex/totg/path.hpp>
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
/// the latest emitted sample. Staged motion begins where the active trajectory ends,
/// and sampling continues into it once the active trajectory is exhausted. A caller
/// may also stop pivoting on its own initiative with `start_staging()`.
///
/// If staged motion is waiting when the active trajectory reaches its end, the sample
/// at that instant comes from the staged motion. It has zero velocity, as the end of
/// the active trajectory would, but carries the acceleration the staged motion starts
/// with rather than zero.
///
/// Sampling is forward-only and stateful: each call to `sample_next()` or
/// `sample_at_least()` advances an internal cursor, and how far that cursor has
/// advanced determines whether a later extend can pivot or must stage. The session
/// assumes single-threaded ownership; sampling and extending from different threads
/// is unsupported.
///
class VIAM_TRAJEX_TOTG_API session {
   public:
    ///
    /// What one `extend()` call did with the batch it was given, and the timing it computed
    /// along the way.
    ///
    /// The branch slack is optional because only the calls that decide between pivoting and
    /// staging produce it, and those report `k_pivot`, `k_staged_branch_sampled` or
    /// `k_staged_unsamplable`. Once the session is staging nothing is decided, since every
    /// batch joins the staged motion, so there is no deadline for a slack to measure.
    ///
    /// `branch_slack` is the time, in global time, between the most recently emitted sample
    /// and the branch, the point at which the trajectory that would incorporate the batch first
    /// differs from the active trajectory. A pivot is only possible while the branch is still
    /// ahead of sampling, so this is the margin the call had. Positive means it arrived that
    /// much ahead of the deadline; negative means sampling had already passed the branch,
    /// which forces a stage, and the magnitude is how much sooner the call needed to arrive.
    /// The comparison is against what the session has
    /// emitted, not what the arm has executed, so a caller that pulls samples far ahead of
    /// execution spends its own slack doing so.
    ///
    /// `delta_total_duration` is how much the call changed `remaining_total_duration()`, which
    /// is to say how much it added to the motion the session has yet to sample. For
    /// `k_first_build` that is the whole of the first trajectory, for `k_pivot` the difference
    /// between the replacement and the trajectory it replaced, for the staged kinds the growth
    /// of the staged motion, and for `k_noop` zero. Weighed against the interval between calls,
    /// it says whether the caller is adding motion faster than sampling consumes it. It is
    /// signed rather than unsigned because a pivot's replacement no longer has to stop at the
    /// old terminal waypoint and so covers the shared part of the path faster than its
    /// predecessor did; that saving is normally smaller than the motion being added, but
    /// nothing guarantees it.
    ///
    struct extend_result {
        ///
        /// How `extend()` handled a batch.
        ///
        /// A batch either builds the session's first trajectory, replaces the active
        /// trajectory with one that incorporates it, or joins staged motion that follows the
        /// active trajectory once it has been sampled through. The difference matters to a
        /// caller pacing its own sends: a pivot is invisible to the arm, but a stage means the
        /// active trajectory runs to its end, where the arm's velocity reaches zero before the
        /// staged motion begins.
        ///
        /// The two values for the batch that starts staging distinguish the reasons a pivot
        /// was refused. One says the call arrived after the point it needed to change had
        /// already been handed out; the other says it arrived in time but carried less than
        /// one sample period of motion. The remedies differ, so the values do too.
        ///
        /// The integer values are pinned because the C ABI mirrors them.
        ///
        enum class kinds : std::uint8_t {
            k_first_build = 0,            ///< Built the session's first trajectory
            k_pivot = 1,                  ///< Replaced the active trajectory; sampling continues unbroken
            k_staged_branch_sampled = 2,  ///< Staged; sampling had already passed the branch
            k_staged_unsamplable = 3,     ///< Staged; less than one sample period of motion added
            k_staged_again = 4,           ///< Staged; the session was already staging
            k_noop = 5,                   ///< Nothing beyond the seam waypoint; session unchanged
        };

        kinds kind;                                       ///< How the batch was handled
        std::optional<trajectory::seconds> branch_slack;  ///< Time by which the branch beat the last sample
        trajectory::seconds delta_total_duration;         ///< Change the call made to remaining_total_duration()
    };

    ///
    /// Constructs a session with the parameters used to build each trajectory and the
    /// sample rate at which samples will be emitted.
    ///
    /// No trajectory exists until the first call to `extend()`.
    ///
    /// @param path_options Path-construction options (used for every trajectory built by the session)
    /// @param trajectory_options Trajectory-construction options (used for every trajectory built by the session)
    /// @param sample_rate Nominal sample rate. Samples are spaced approximately
    ///                    1 / sample_rate apart, and the exact spacing may differ slightly
    ///                    from one trajectory to the next. This parameter's shape may
    ///                    change if a sampler factory is added later.
    ///
    session(path::options path_options, trajectory::options trajectory_options, types::hertz sample_rate);

    ///
    /// Sessions are neither copyable nor movable, because the sampling state points into the
    /// active trajectory, which lives inside the session.
    ///
    session(const session&) = delete;
    session& operator=(const session&) = delete;
    session(session&&) = delete;
    session& operator=(session&&) = delete;

    ///
    /// Destroys a session.
    ///
    ~session();

    ///
    /// Adds a batch of waypoints to the session.
    ///
    /// If no active trajectory exists, builds the initial one from `batch`.
    /// Otherwise, requires `batch`'s first waypoint to compare bit-exactly equal to the
    /// session's most recently stored waypoint, then either pivots the active trajectory
    /// onto one incorporating the remainder of `batch`, or stages that remainder to follow
    /// the active trajectory. The returned `extend_result` says which, along with the
    /// timing a caller needs in order to pace its own sends; a caller with no interest in
    /// either may discard it.
    ///
    /// Waypoints in `batch` are assumed to have been deduplicated by the caller. The
    /// bit-exact seam requirement means the merged sequence retains the dedup invariant
    /// after the seam point is dropped.
    ///
    /// @param batch Waypoints to append
    /// @return How the batch was handled, and the timing that went with it
    /// @throws std::invalid_argument if `batch`'s DOF disagrees with the session's existing
    ///         waypoint DOF, or if its first waypoint does not equal the session's last
    /// @throws Any exception raised by trajectory construction if `batch` cannot be built
    ///         into the active trajectory or the staged motion. Session state is unchanged
    ///         in that case.
    ///
    extend_result extend(const waypoint_accumulator& batch);

    ///
    /// Stops the session pivoting, so that every subsequent `extend()` stages its batch.
    ///
    /// The cost of an `extend()` grows with the motion added since staging last began, or
    /// since the session started if it never has. Calling this starts that growth over, at the
    /// price of the arm's velocity reaching zero at the end of the active trajectory before
    /// the staged motion begins.
    ///
    /// Staging lasts until the next rebase, after which `extend()` pivots again. Calling
    /// this while already staging has no effect, and neither does calling it before the
    /// first `extend()`, since the first batch always builds the first trajectory.
    ///
    void start_staging() noexcept;

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
    /// It counts only the active trajectory, not staged motion, so it drains toward zero
    /// while batches are staged even though the session still has work queued, and jumps
    /// back up when that work becomes active at the rebase. A caller pacing itself against
    /// this number needs to know that. `remaining_total_duration()` counts the staged motion
    /// too.
    ///
    /// @return Unsampled time left in the active trajectory, or zero if there is none
    ///
    trajectory::seconds remaining_active_duration() const noexcept;

    ///
    /// Returns how much motion the session has yet to sample, staged motion included.
    ///
    /// This is the end of the staged motion in global time, or of the active trajectory if
    /// nothing is staged, less the time of the most recently emitted sample. Like
    /// `remaining_active_duration()`, it is clamped at zero.
    ///
    /// @return Unsampled time left in the active trajectory and any staged motion, or zero
    ///         if there is none
    ///
    trajectory::seconds remaining_total_duration() const noexcept;

    ///
    /// Pulls the next `n` samples from the session, advancing the sampling cursor.
    ///
    /// Samples are spaced approximately one sample period apart, per the sample rate given at
    /// construction. Returns fewer than `n` samples if the session is exhausted, which happens
    /// when the active trajectory has run out and nothing is staged to follow it.
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
    /// The last sample may land somewhat past the horizon, since the session never splits a
    /// sample period to hit it exactly.
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
    // Waypoints the session owns. It's defined in the source file, so that the private
    // waypoint store type stays out of this public header.
    struct VIAM_TRAJEX_TOTG_HIDDEN waypoint_store_impl;

    // While staging, the waypoints of the staged motion and the trajectory built from them,
    // which follows the active trajectory once it has been sampled through. The waypoints start
    // with the active trajectory's last waypoint, so the staged trajectory begins where the
    // active one ends. Both stay empty until the first batch is staged.
    struct staging_state {
        std::unique_ptr<waypoint_store_impl> waypoints;
        std::optional<trajectory> next;
    };

    // Builds a trajectory from the given waypoints, threading through path::options and
    // trajectory::options. Throws on validation failure inside path::create or
    // trajectory::create, leaving every member it does not touch alone. A caller that has
    // already appended to a waypoint store is responsible for winding that back.
    VIAM_TRAJEX_TOTG_HIDDEN trajectory build_trajectory_from_(const waypoint_accumulator& waypoints) const;

    // Appends `batch` past its seam to the staged waypoints and rebuilds the staged trajectory
    // from them. Returns the growth in duration over the staged trajectory it replaced, or the
    // whole of the new duration if there was none. If the build throws, `staging` is left as
    // it was.
    VIAM_TRAJEX_TOTG_HIDDEN trajectory::seconds stage_(staging_state& staging, const waypoint_accumulator& batch);

    // Emits a single sample, advancing the cursor. Installs the staged trajectory when the
    // active trajectory is about to emit its terminal, or already has. Returns nullopt when
    // the session is fully drained.
    VIAM_TRAJEX_TOTG_HIDDEN std::optional<struct trajectory::sample> sample_one_();

    // Makes the staged trajectory active, starts sampling it `start` into its own time,
    // advances the epoch by the prior active's duration, ends staging, and increments the
    // generation count. Preconditions: active_ holds a value, and a staged trajectory exists.
    VIAM_TRAJEX_TOTG_HIDDEN void install_staged_(trajectory::seconds start);

    // The staged trajectory, or null if nothing has been staged.
    VIAM_TRAJEX_TOTG_HIDDEN const trajectory* staged_trajectory_() const noexcept;

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
    // A store cannot be moved, because its accumulator views its own storage, so it is held by
    // pointer. That lets the staged waypoints take over by handing over the pointer when the
    // staged trajectory becomes active, rather than by copying them across.
    std::unique_ptr<waypoint_store_impl> waypoints_;

    // The currently active trajectory, or nullopt before the first successful extend.
    // Storage in std::optional is in-place, so `&*active_` is a stable address across
    // pivot and rebase (which both proceed by move-assigning a freshly-built trajectory
    // into this optional). Tests must use trajectory_generation_count() to witness
    // transitions instead of comparing pointers.
    std::optional<trajectory> active_;

    // The sampler and cursor for the active trajectory. Both are rebuilt whenever a new
    // trajectory becomes active, so that each one is sampled on a fresh grid aligned to its
    // own duration. Both refer to active_, so they are emplaced only after the new trajectory
    // has been assigned to it.
    std::optional<uniform_sampler> sampler_;
    std::optional<trajectory::cursor> cursor_;

    // Global time at which active_'s local t=0 sits. Pivots leave this unchanged; rebases
    // advance it by the prior active's duration.
    trajectory::seconds epoch_{0.0};

    // Global time of the most recently emitted sample, or zero if no sample has been
    // emitted yet. Cached for the current_time() accessor.
    trajectory::seconds current_time_{0.0};

    // Cumulative count of samples emitted. A pivot uses it to decide where the new sampler
    // starts, which is at zero if nothing has been emitted, and otherwise one sample period
    // past the current local time.
    std::size_t emitted_sample_count_{0};

    // Engaged exactly when the session is staging, which begins when a batch stages or the
    // caller calls start_staging(), and ends at the rebase that makes the staged trajectory
    // active. Engaged with no staged trajectory means the caller started staging before any
    // batch arrived, so extend() must stage but sampling has nothing to rebase onto.
    std::optional<staging_state> staging_;

    // The most recently received waypoint, against which the next extend's seam is
    // bit-exactly validated. Empty (shape (0,)) before the first extend.
    xvector<> last_waypoint_;

    // Cumulative count of trajectories the session has installed as active. Increments
    // on first build, on each pivot, and on each rebase.
    std::size_t generation_count_{0};
};

}  // namespace viam::trajex::totg::streaming
