#include <viam/trajex/totg/streaming/session.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <viam/trajex/totg/streaming/private/waypoint_store.hpp>

namespace viam::trajex::totg::streaming {

// Exists only so that the header can hold waypoint stores without naming the private type.
struct session::waypoint_store_impl : waypoint_store {};

namespace {

// Compared bitwise rather than within a tolerance, because the seam waypoint is one the caller
// was handed back and is expected to return unmodified. Anything else is a protocol error on
// their side rather than drift worth accommodating.
bool rows_bit_exact(const waypoint_accumulator::value_type& a, const xvector<>& b) {
    return std::ranges::equal(a, b);
}

// Returns the local time of the branch, found by walking the integration points of `existing`
// and `candidate` in lockstep until they first disagree. If all of `existing`'s integration
// points are a prefix of `candidate`'s, the branch sits at the end of `existing`, so its
// duration is returned.
trajectory::seconds find_branch_local_time(const trajectory& existing, const trajectory& candidate) {
    const auto& existing_pts = existing.get_integration_points();
    const auto& candidate_pts = candidate.get_integration_points();
    const auto result = std::ranges::mismatch(existing_pts, candidate_pts);
    if (result.in1 == existing_pts.end()) {
        return existing.duration();
    }
    return result.in1->time;
}

trajectory::seconds validate_sample_rate_and_compute_period(types::hertz sample_rate) {
    if (!std::isfinite(sample_rate.value) || sample_rate.value <= 0.0) {
        throw std::invalid_argument("streaming::session: sample_rate must be positive and finite");
    }
    return trajectory::seconds{1.0 / sample_rate.value};
}

}  // namespace

session::session(path::options path_options, trajectory::options trajectory_options, types::hertz sample_rate)
    : path_options_(std::move(path_options)),
      trajectory_options_(std::move(trajectory_options)),
      sample_rate_(sample_rate),
      sample_period_(validate_sample_rate_and_compute_period(sample_rate)),
      waypoints_(std::make_unique<waypoint_store_impl>()) {}

session::~session() = default;

session::extend_result session::extend(const waypoint_accumulator& batch) {
    using kinds = extend_result::kinds;

    if (batch.empty()) {
        throw std::invalid_argument("streaming::session::extend: batch is empty");
    }

    // The first extend has nothing to pivot from, so it builds the initial trajectory directly
    // from the batch.
    if (!active_) {
        // The store has to be populated before the trajectory can be built from it, so a failed
        // build leaves waypoints behind that no trajectory corresponds to. Empty it before
        // rethrowing, so a caller retrying with a corrected batch starts where it was.
        waypoints_->append(batch, 0);
        auto new_active = [&] {
            try {
                return build_trajectory_from_(waypoints_->waypoints());  // throws on validation failure
            } catch (...) {
                waypoints_->truncate(0);
                throw;
            }
        }();

        // Build the sampler for the new active before committing any moves so the throw
        // contract (state unchanged on failure) is preserved.
        uniform_sampler new_sampler = uniform_sampler::quantized_for_trajectory(new_active, sample_rate_, trajectory::seconds{0.0});

        last_waypoint_ = waypoints_->last();
        active_ = std::move(new_active);
        cursor_.emplace(active_->create_cursor());
        sampler_.emplace(std::move(new_sampler));
        generation_count_ = 1;

        // Nothing preceded this trajectory, so there is no branch to measure against, and the
        // whole of what we just built counts as growth.
        return {kinds::k_first_build, std::nullopt, active_->duration()};
    }

    // Every later batch must agree with the session on DOF and on the seam waypoint, and both are
    // checked before any state is touched.
    if (batch.dof() != waypoints_->dof()) {
        throw std::invalid_argument("streaming::session::extend: DOF mismatch");
    }
    if (!rows_bit_exact(batch.at(0), last_waypoint_)) {
        throw std::invalid_argument("streaming::session::extend: seam mismatch");
    }

    // A batch that carries nothing past the seam leaves the session unchanged.
    if (batch.size() == 1) {
        return {kinds::k_noop, std::nullopt, trajectory::seconds{0.0}};
    }

    // A session that is already staging takes the batch into the staged motion, which nothing
    // has sampled yet, so there is no question of pivoting the active trajectory and no
    // deadline the batch could have missed. That is why there is no slack to report.
    if (staging_) {
        const auto growth = stage_(*staging_, batch);
        last_waypoint_ = batch.at(batch.size() - 1);
        return {kinds::k_staged_again, std::nullopt, growth};
    }

    // Build a candidate trajectory from the active waypoints plus the batch's new waypoints,
    // then find the branch, the earliest point at which the candidate stops agreeing with the
    // active trajectory. Where the branch falls decides whether we can pivot.
    //
    // Appending is provisional: the candidate may lose to staging below, and building it may
    // fail outright, so the store is wound back to `committed_waypoints` on either path. Only
    // a pivot keeps the appended waypoints, because only then do they describe `active_`.
    const auto committed_waypoints = waypoints_->size();
    waypoints_->append(batch, 1);
    auto candidate = [&] {
        try {
            return build_trajectory_from_(waypoints_->waypoints());  // throws on validation failure
        } catch (...) {
            waypoints_->truncate(committed_waypoints);
            throw;
        }
    }();

    const auto branch_local = find_branch_local_time(*active_, candidate);
    const auto branch_global = epoch_ + branch_local;

    // A pivot is admissible only when two conditions hold. The branch must lie ahead of the
    // latest emitted sample, unless nothing has been emitted yet, so that the new trajectory
    // differs from the old one only where we have not sampled. And the new sampler must have
    // something to sample. It resumes one sample period past the last emitted sample, which
    // keeps the spacing roughly uniform across the pivot, and if that lands at or past the
    // candidate's end, the candidate has less than one sample period left after the branch. A
    // pivot would then produce no new samples, and quantized_for_trajectory would reject the
    // start anyway, so the batch stages instead and becomes active at the next rebase.
    const auto starting_local_time = (emitted_sample_count_ == 0) ? trajectory::seconds{0.0} : (current_time_ - epoch_) + sample_period_;
    const bool branch_ahead = (emitted_sample_count_ == 0) || (branch_global > current_time_);
    const bool has_samplable_material = starting_local_time < candidate.duration();

    const auto branch_slack = branch_global - current_time_;

    if (branch_ahead && has_samplable_material) {
        // Both durations have to be read before the moves below: afterwards `candidate` is
        // gutted and `active_` names the new trajectory, so the difference would come out zero.
        const auto delta_total_duration = candidate.duration() - active_->duration();

        uniform_sampler new_sampler = uniform_sampler::quantized_for_trajectory(candidate, sample_rate_, starting_local_time);

        last_waypoint_ = waypoints_->last();
        active_ = std::move(candidate);
        cursor_.emplace(active_->create_cursor());
        sampler_.emplace(std::move(new_sampler));
        ++generation_count_;
        return {kinds::k_pivot, branch_slack, delta_total_duration};
    }

    // Staging instead of pivoting, so the candidate is discarded and its waypoints along with
    // it. The batch starts the staged motion instead. That is built aside and only kept once it
    // succeeds, so a batch the staged motion cannot take leaves the session as it was. Slack
    // against the active trajectory is what explains the stage, so it is the slack reported.
    waypoints_->truncate(committed_waypoints);
    staging_state staging;
    const auto growth = stage_(staging, batch);
    staging_ = std::move(staging);
    last_waypoint_ = batch.at(batch.size() - 1);

    // Both stage conditions can hold at once. Report lateness in that case, because it is the
    // one the caller can do something about: sending sooner fixes a branch that has already
    // been sampled, whereas an unsamplable candidate needs a larger batch instead.
    const auto kind = branch_ahead ? kinds::k_staged_unsamplable : kinds::k_staged_branch_sampled;
    return {kind, branch_slack, growth};
}

trajectory::seconds session::stage_(staging_state& staging, const waypoint_accumulator& batch) {
    // The staged motion starts where the active trajectory ends, at its last waypoint rather
    // than at its sampled terminal pose, for the reason given at install_staged_. The store is
    // allocated here rather than when staging begins, so that start_staging() need not allocate.
    if (!staging.waypoints) {
        staging.waypoints = std::make_unique<waypoint_store_impl>();
    }
    auto& store = *staging.waypoints;
    const auto committed_waypoints = store.size();
    if (store.empty()) {
        store.append(waypoints_->waypoints(), waypoints_->size() - 1);
    }

    // As with a pivot, appending is provisional until the trajectory builds.
    store.append(batch, 1);
    auto candidate = [&] {
        try {
            return build_trajectory_from_(store.waypoints());  // throws on validation failure
        } catch (...) {
            store.truncate(committed_waypoints);
            throw;
        }
    }();

    // With no staged trajectory yet, the whole of the new one counts as growth.
    const auto growth = staging.next ? candidate.duration() - staging.next->duration() : candidate.duration();
    staging.next = std::move(candidate);
    return growth;
}

void session::start_staging() noexcept {
    // Before the first trajectory exists there is nothing to stage behind, and remembering the
    // request would make the session stage the second batch instead, long after the caller
    // asked. If the session is already staging, emplacing again would discard the staged
    // motion, so it is left alone.
    if (active_ && !staging_) {
        staging_.emplace();
    }
}

trajectory::seconds session::current_time() const noexcept {
    return current_time_;
}

trajectory::seconds session::remaining_active_duration() const noexcept {
    if (!active_) {
        return trajectory::seconds{0.0};
    }

    // The active's end has to be lifted into global time before the subtraction: current_time_
    // is global, and after a rebase the epoch is non-zero, so differencing against the
    // trajectory's own local duration would run negative and stay there.
    const auto active_end = epoch_ + active_->duration();
    if (active_end <= current_time_) {
        return trajectory::seconds{0.0};
    }
    return active_end - current_time_;
}

trajectory::seconds session::remaining_total_duration() const noexcept {
    if (!active_) {
        return trajectory::seconds{0.0};
    }

    // Staged motion starts where the active trajectory ends, so its end in global time is the
    // active trajectory's end plus its own duration.
    auto end = epoch_ + active_->duration();
    if (const auto* staged = staged_trajectory_()) {
        end = end + staged->duration();
    }
    if (end <= current_time_) {
        return trajectory::seconds{0.0};
    }
    return end - current_time_;
}

std::vector<struct trajectory::sample> session::sample_next(std::size_t n) {
    std::vector<struct trajectory::sample> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto opt = sample_one_();
        if (!opt) {
            break;
        }
        result.push_back(std::move(*opt));
    }
    return result;
}

std::vector<struct trajectory::sample> session::sample_at_least(trajectory::seconds horizon) {
    const auto target = current_time_ + horizon;
    std::vector<struct trajectory::sample> result;
    while (true) {
        auto opt = sample_one_();
        if (!opt) {
            break;
        }
        result.push_back(std::move(*opt));
        if (current_time_ >= target) {
            break;
        }
    }
    return result;
}

const trajectory* session::active_trajectory() const noexcept {
    return active_ ? &(*active_) : nullptr;
}

trajectory::seconds session::active_epoch() const noexcept {
    return epoch_;
}

std::size_t session::trajectory_generation_count() const noexcept {
    return generation_count_;
}

trajectory session::build_trajectory_from_(const waypoint_accumulator& waypoints) const {
    path p = path::create(waypoints, path_options_);
    return trajectory::create(std::move(p), trajectory_options_);
}

std::optional<struct trajectory::sample> session::sample_one_() {
    if (!sampler_ || !cursor_) {
        return std::nullopt;
    }

    // If the next sample would be the active trajectory's terminal and staged motion is waiting
    // to follow it, rebase first and emit the staged trajectory's first sample in its place. The
    // two describe the same instant, at the same pose and at zero velocity, but the terminal
    // reports zero acceleration where the staged trajectory's first sample reports the
    // acceleration it starts with, so the acceleration goes straight from one trajectory's final
    // deceleration to the next one's initial acceleration.
    if (sampler_->remaining() == 1 && staged_trajectory_()) {
        install_staged_(trajectory::seconds{0.0});
    }

    auto local_sample = sampler_->next(*cursor_);
    if (!local_sample) {
        // The terminal has already been emitted, because nothing was staged when it came up. A
        // session that is staging but still has nothing staged has nothing to rebase onto, so it
        // reports itself drained. It stays staging, and the next batch to arrive follows on from
        // the end of the active trajectory.
        if (!staged_trajectory_()) {
            return std::nullopt;
        }
        install_staged_(sample_period_);
        local_sample = sampler_->next(*cursor_);
        if (!local_sample) {
            // The freshly built sampler should always have at least one sample to emit. If a
            // degenerate trajectory somehow has none, the session reports itself drained rather
            // than looping forever.
            return std::nullopt;
        }
    }

    auto sample = std::move(*local_sample);
    sample.time = sample.time + epoch_;
    ++emitted_sample_count_;
    current_time_ = sample.time;
    return sample;
}

const trajectory* session::staged_trajectory_() const noexcept {
    return (staging_ && staging_->next) ? &*staging_->next : nullptr;
}

void session::install_staged_(trajectory::seconds start) {
    // Preconditions: active_ holds, and a staged trajectory exists.
    //
    // The staged trajectory and its waypoints were built as batches arrived, so nothing here
    // builds a trajectory or copies waypoints. The staged waypoints begin with the active
    // trajectory's last waypoint, not its sampled terminal pose. Sampling the trajectory at its
    // duration would give a value that is mathematically equal to the last waypoint for a
    // rest-to-rest trajectory but can differ by a little floating-point drift, and trajex's
    // path-coalescing tolerances can react badly to that difference. Keep the streaming layer in
    // the waypoint domain.
    const auto old_duration = active_->duration();
    auto& staged = *staging_->next;

    // The new sampler starts `start` into the staged trajectory. At zero, the staged trajectory's
    // first sample takes the place of the previous terminal, which has not been emitted.
    // Otherwise the previous terminal has already gone out at global time epoch_ + old_duration,
    // and the caller starts one sample period past it, so that the seam carries no duplicate
    // sample.
    //
    // A staged trajectory shorter than that one sample period would put the start at or past
    // its end, where quantized_for_trajectory refuses to start. This is the same case extend()
    // guards against on the pivot side. The staged motion is still valid and reachable, so we
    // must deliver it rather than drop it, but the whole move fits inside one sample period, so
    // the only sample worth emitting is the terminal, where the arm has completed the move and
    // come to rest at the destination. Build a one-sample grid that lands on the trajectory's
    // end. Emitting only the terminal also avoids repeating the seam sample, which a sampler that
    // started at zero would do.
    uniform_sampler new_sampler = (start < staged.duration()) ? uniform_sampler::quantized_for_trajectory(staged, sample_rate_, start)
                                                              : uniform_sampler{std::size_t{1}};

    active_ = std::move(staged);
    waypoints_ = std::move(staging_->waypoints);
    cursor_.emplace(active_->create_cursor());
    sampler_.emplace(std::move(new_sampler));
    epoch_ = epoch_ + old_duration;
    staging_.reset();
    ++generation_count_;
}

}  // namespace viam::trajex::totg::streaming
