#include <viam/trajex/totg/streaming/session.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xview.hpp>
#endif

namespace viam::trajex::totg::streaming {

namespace {

// Staging keeps batches aside until the next rebase folds them in, and it cannot hold onto
// the caller's accumulator to do so, because that accumulator views memory the caller owns.
// These copy the rows it needs into arrays the session owns.

// Compared bitwise rather than within a tolerance, because the seam waypoint is one the
// caller was handed back and is expected to return unmodified; anything else is a protocol
// error on their side rather than drift worth accommodating.
bool rows_bit_exact(const waypoint_accumulator::value_type& a, const xvector<>& b) {
    return std::ranges::equal(a, b);
}

// Caller must ensure batch.size() > from.
xmatrix<> accumulator_tail_to_matrix(const waypoint_accumulator& batch, std::size_t from) {
    // Allocated without initialising, since every element is written below.
    auto result = xmatrix<>::from_shape(std::vector<std::size_t>{batch.size() - from, batch.dof()});

    std::size_t row = 0;
    for (const auto& waypoint : batch | std::views::drop(from)) {
        xt::view(result, row++, xt::all()) = waypoint;
    }
    return result;
}

xmatrix<> stack_anchor_and_staged(const xvector<>& anchor, const std::vector<xmatrix<>>& staged) {
    const auto staged_rows =
        std::transform_reduce(staged.begin(), staged.end(), std::size_t{0}, std::plus{}, [](const auto& batch) { return batch.shape(0); });

    auto result = xmatrix<>::from_shape(std::vector<std::size_t>{staged_rows + 1, anchor.shape(0)});
    xt::view(result, 0, xt::all()) = anchor;

    // Each staged batch is already a contiguous block of rows, so it lands in one assignment
    // rather than a row at a time.
    std::size_t row = 1;
    for (const auto& batch : staged) {
        const auto rows = batch.shape(0);
        xt::view(result, xt::range(row, row + rows), xt::all()) = batch;
        row += rows;
    }
    return result;
}

// Returns the local time of the first divergence between `active`'s integration points
// and `candidate`'s integration points, walking them in lockstep. If `active`'s entire
// integration-point sequence is a prefix of `candidate`'s, returns the active's duration
// (the branch effectively sits at the end of active).
trajectory::seconds find_branch_local_time(const trajectory& active, const trajectory& candidate) {
    const auto& active_pts = active.get_integration_points();
    const auto& candidate_pts = candidate.get_integration_points();
    const auto result = std::ranges::mismatch(active_pts, candidate_pts);
    if (result.in1 == active_pts.end()) {
        return active.duration();
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
      sample_period_(validate_sample_rate_and_compute_period(sample_rate)) {}

session::extend_result session::extend(const waypoint_accumulator& batch) {
    using kinds = extend_result::kinds;

    if (batch.empty()) {
        throw std::invalid_argument("streaming::session::extend: batch is empty");
    }

    // First extend: build the initial trajectory directly from the batch.
    if (!active_) {
        // The store has to be populated before the trajectory can be built from it, so a
        // failed build leaves waypoints behind that no trajectory corresponds to. Empty it
        // again before rethrowing, so a caller that retries with a corrected batch starts
        // from the same state it had before.
        waypoints_.append(batch, 0);
        auto new_active = [&] {
            try {
                return build_trajectory_from_(waypoints_.waypoints());  // throws on validation failure
            } catch (...) {
                waypoints_.truncate(0);
                throw;
            }
        }();

        // Build the sampler for the new active before committing any moves so the throw
        // contract (state unchanged on failure) is preserved.
        uniform_sampler new_sampler = uniform_sampler::quantized_for_trajectory(new_active, sample_rate_, trajectory::seconds{0.0});

        last_waypoint_ = waypoints_.last();
        active_ = std::move(new_active);
        cursor_.emplace(active_->create_cursor());
        sampler_.emplace(std::move(new_sampler));
        generation_count_ = 1;

        // Nothing preceded this trajectory, so there is no branch to measure against, and the
        // whole of what we just built counts as growth.
        return {kinds::k_first_build, std::nullopt, active_->duration()};
    }

    // Subsequent extends: validate DOF and seam before touching any state.
    if (batch.dof() != waypoints_.dof()) {
        throw std::invalid_argument("streaming::session::extend: DOF mismatch");
    }
    if (!rows_bit_exact(batch.at(0), last_waypoint_)) {
        throw std::invalid_argument("streaming::session::extend: seam mismatch");
    }

    const std::size_t post_seam_count = batch.size() - 1;

    // Already staging: skip the candidate build, just record the new waypoints in staging.
    // Nothing is compared here, so neither time can be reported.
    if (!staged_batches_.empty()) {
        if (post_seam_count == 0) {
            return {kinds::k_noop, std::nullopt, std::nullopt};
        }
        staged_batches_.push_back(accumulator_tail_to_matrix(batch, 1));
        last_waypoint_ = batch.at(batch.size() - 1);
        return {kinds::k_staged_again, std::nullopt, std::nullopt};
    }

    // Seam-only batch with no new waypoints: nothing to do.
    if (post_seam_count == 0) {
        return {kinds::k_noop, std::nullopt, std::nullopt};
    }

    // Build a candidate trajectory from the active waypoints plus the batch's new waypoints,
    // then find the branch: the earliest point where the candidate diverges from the current
    // active. Where that branch falls decides whether we can pivot.
    //
    // Appending is provisional: the candidate may lose to staging below, and building it may
    // fail outright, so the store is wound back to `committed_waypoints` on either path. Only
    // a pivot keeps the appended waypoints, because only then do they describe `active_`.
    const auto committed_waypoints = waypoints_.size();
    waypoints_.append(batch, 1);
    auto candidate = [&] {
        try {
            return build_trajectory_from_(waypoints_.waypoints());  // throws on validation failure
        } catch (...) {
            waypoints_.truncate(committed_waypoints);
            throw;
        }
    }();

    const auto branch_local = find_branch_local_time(*active_, candidate);
    const auto branch_global = epoch_ + branch_local;

    // Decide between pivot and stage. A pivot is admissible only when two conditions hold.
    // First, the branch must lie ahead of the latest emitted sample (or nothing has been
    // emitted yet), so that the new trajectory differs from the old one only where we have not
    // sampled yet. Second, the new sampler's resume offset must still leave some trajectory to
    // sample before the candidate ends. That offset is one sample period past the last emitted
    // sample, which keeps the sample spacing roughly uniform across the pivot; if it lands at
    // or past the candidate's duration, the candidate has less than one sample period left
    // after the branch, so a pivot would produce no new samples (and quantized_for_trajectory
    // would reject a start at or beyond the duration). In that case stage the batch and let it
    // fold in at the next rebase.
    const auto starting_local_time = (emitted_sample_count_ == 0) ? trajectory::seconds{0.0} : (current_time_ - epoch_) + sample_period_;
    const bool branch_ahead = (emitted_sample_count_ == 0) || (branch_global > current_time_);
    const bool has_samplable_material = starting_local_time < candidate.duration();

    const auto branch_slack = branch_global - current_time_;

    if (branch_ahead && has_samplable_material) {
        // Both durations have to be read before the moves below: afterwards `candidate` is
        // gutted and `active_` names the new trajectory, so the difference would come out zero.
        const auto delta_active_duration = candidate.duration() - active_->duration();

        uniform_sampler new_sampler = uniform_sampler::quantized_for_trajectory(candidate, sample_rate_, starting_local_time);

        last_waypoint_ = waypoints_.last();
        active_ = std::move(candidate);
        cursor_.emplace(active_->create_cursor());
        sampler_.emplace(std::move(new_sampler));
        ++generation_count_;
        return {kinds::k_pivot, branch_slack, delta_active_duration};
    }

    // Staging instead of pivoting, so the candidate is discarded and its waypoints along
    // with it; they will arrive again by way of `staged_batches_` at the next rebase.
    waypoints_.truncate(committed_waypoints);
    staged_batches_.push_back(accumulator_tail_to_matrix(batch, 1));
    last_waypoint_ = batch.at(batch.size() - 1);

    // Both stage conditions can hold at once. Report lateness in that case, because it is the
    // one the caller can do something about: sending sooner fixes a branch that has already
    // been sampled, whereas an unsamplable candidate needs a larger batch instead.
    const auto kind = branch_ahead ? kinds::k_staged_unsamplable : kinds::k_staged_branch_sampled;
    return {kind, branch_slack, std::nullopt};
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

    auto local_sample = sampler_->next(*cursor_);
    if (!local_sample) {
        if (staged_batches_.empty()) {
            return std::nullopt;
        }
        rebase_();
        local_sample = sampler_->next(*cursor_);
        if (!local_sample) {
            // Defensive: the freshly-built sampler should always have at least one sample
            // to emit, but if a degenerate trajectory somehow has none, treat as drained
            // rather than infinite-looping.
            return std::nullopt;
        }
    }

    auto sample = std::move(*local_sample);
    sample.time = sample.time + epoch_;
    ++emitted_sample_count_;
    current_time_ = sample.time;
    return sample;
}

void session::rebase_() {
    // Preconditions: active_ holds, staged_batches_ non-empty.
    //
    // The new chain's first waypoint is the active's last waypoint (the literal end of the
    // prior chain's waypoint sequence), not the sampled terminal pose. Sampling the trajectory
    // at its duration would give a value that is mathematically equal to the last waypoint for
    // a rest-to-rest trajectory but can differ by a little floating-point drift, and trajex's
    // path-coalescing tolerances can react badly to that difference. Keep the streaming layer
    // in the waypoint domain.
    const auto old_duration = active_->duration();
    auto anchor = waypoints_.last();

    // Assembled into a flat array first, and loaded into the store only once the trajectory
    // has been built from it, so that a failed build leaves the session's waypoints as they
    // were rather than half-replaced.
    auto new_waypoints = stack_anchor_and_staged(anchor, staged_batches_);
    const waypoint_accumulator replacement{new_waypoints};
    auto new_active = build_trajectory_from_(replacement);

    // The previous chain's terminal was emitted as its last sample at global time
    // (epoch_ + old_duration). Start the new sampler one nominal sample period past that, so
    // the seam carries no duplicate sample and the gap between the two trajectories is exactly
    // sample_period_.
    //
    // If the rebuilt trajectory is shorter than one sample period, that resume offset lands at
    // or past its end, and quantized_for_trajectory rejects a start at or beyond the duration.
    // This is the same case extend() guards against on the pivot side. The staged motion is
    // still valid and reachable, so we must deliver it rather than drop it, but the whole move
    // fits inside one sample period, so the only sample worth emitting is the terminal, where
    // the arm has completed the move and come to rest at the destination. Build a one-sample
    // grid that lands on the trajectory's end. Emitting only the terminal also avoids repeating
    // the seam sample, which a sampler that started at zero would do.
    uniform_sampler new_sampler = (sample_period_ < new_active.duration())
                                      ? uniform_sampler::quantized_for_trajectory(new_active, sample_rate_, sample_period_)
                                      : uniform_sampler{std::size_t{1}};

    // The anchor is already the store's last waypoint and also row zero of `new_waypoints`,
    // so reducing the store to it and appending the remainder replaces the contents without
    // storing that waypoint twice.
    waypoints_.reset_to_last();
    waypoints_.append(replacement, 1);

    active_ = std::move(new_active);
    cursor_.emplace(active_->create_cursor());
    sampler_.emplace(std::move(new_sampler));
    epoch_ = epoch_ + old_duration;
    staged_batches_.clear();
    ++generation_count_;
}

}  // namespace viam::trajex::totg::streaming
