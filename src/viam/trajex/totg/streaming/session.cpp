#include <viam/trajex/totg/streaming/session.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
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

// These helpers materialize accumulator row-views and 2D-xarray slices into owned xarrays.
// The session needs stable storage independent of caller-provided accumulators because
// waypoint_accumulator holds row-views into source arrays it does not own.

xt::xarray<double> view_to_xarray(const waypoint_accumulator::value_type& row) {
    const std::size_t dof = row.shape(0);
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{dof});
    for (std::size_t j = 0; j < dof; ++j) {
        result(j) = row(j);
    }
    return result;
}

xt::xarray<double> row_to_xarray(const xt::xarray<double>& arr, std::size_t row) {
    const std::size_t dof = arr.shape(1);
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{dof});
    for (std::size_t j = 0; j < dof; ++j) {
        result(j) = arr(row, j);
    }
    return result;
}

bool rows_bit_exact(const waypoint_accumulator::value_type& a, const xt::xarray<double>& b) {
    if (a.shape(0) != b.shape(0)) {
        return false;
    }
    for (std::size_t i = 0; i < b.shape(0); ++i) {
        if (a(i) != b(i)) {
            return false;
        }
    }
    return true;
}

xt::xarray<double> accumulator_to_xarray(const waypoint_accumulator& batch) {
    const std::size_t count = batch.size();
    const std::size_t dof = batch.dof();
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{count, dof});
    for (std::size_t i = 0; i < count; ++i) {
        const auto& row = batch.at(i);
        for (std::size_t j = 0; j < dof; ++j) {
            result(i, j) = row(j);
        }
    }
    return result;
}

// Caller must ensure batch.size() > from.
xt::xarray<double> accumulator_tail_to_xarray(const waypoint_accumulator& batch, std::size_t from) {
    const std::size_t count = batch.size() - from;
    const std::size_t dof = batch.dof();
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{count, dof});
    for (std::size_t i = 0; i < count; ++i) {
        const auto& row = batch.at(from + i);
        for (std::size_t j = 0; j < dof; ++j) {
            result(i, j) = row(j);
        }
    }
    return result;
}

xt::xarray<double> concat_active_with_batch_tail(const xt::xarray<double>& base,
                                                 const waypoint_accumulator& batch,
                                                 std::size_t batch_from) {
    const std::size_t n_base = base.shape(0);
    const std::size_t n_add = batch.size() - batch_from;
    const std::size_t dof = base.shape(1);
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{n_base + n_add, dof});
    for (std::size_t i = 0; i < n_base; ++i) {
        for (std::size_t j = 0; j < dof; ++j) {
            result(i, j) = base(i, j);
        }
    }
    for (std::size_t i = 0; i < n_add; ++i) {
        const auto& row = batch.at(batch_from + i);
        for (std::size_t j = 0; j < dof; ++j) {
            result(n_base + i, j) = row(j);
        }
    }
    return result;
}

xt::xarray<double> stack_anchor_and_staged(const xt::xarray<double>& anchor, const std::vector<xt::xarray<double>>& staged) {
    const std::size_t dof = anchor.shape(0);
    std::size_t total_rows = 1;
    for (const auto& s : staged) {
        total_rows += s.shape(0);
    }
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{total_rows, dof});
    for (std::size_t j = 0; j < dof; ++j) {
        result(0, j) = anchor(j);
    }
    std::size_t row = 1;
    for (const auto& s : staged) {
        for (std::size_t i = 0; i < s.shape(0); ++i) {
            for (std::size_t j = 0; j < dof; ++j) {
                result(row, j) = s(i, j);
            }
            ++row;
        }
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

void session::extend(const waypoint_accumulator& batch) {
    if (batch.empty()) {
        throw std::invalid_argument("streaming::session::extend: batch is empty");
    }

    // First extend: build the initial trajectory directly from the batch.
    if (!active_) {
        auto new_waypoints = accumulator_to_xarray(batch);
        auto new_active = build_trajectory_from_(new_waypoints);  // throws on validation failure

        // Build the sampler for the new active before committing any moves so the throw
        // contract (state unchanged on failure) is preserved.
        uniform_sampler new_sampler = uniform_sampler::quantized_for_trajectory(new_active, sample_rate_, trajectory::seconds{0.0});

        last_waypoint_ = row_to_xarray(new_waypoints, new_waypoints.shape(0) - 1);
        active_waypoints_ = std::move(new_waypoints);
        active_ = std::move(new_active);
        cursor_.emplace(active_->create_cursor());
        sampler_.emplace(std::move(new_sampler));
        generation_count_ = 1;
        return;
    }

    // Subsequent extends: validate DOF and seam before touching any state.
    if (batch.dof() != active_waypoints_.shape(1)) {
        throw std::invalid_argument("streaming::session::extend: DOF mismatch");
    }
    if (!rows_bit_exact(batch.at(0), last_waypoint_)) {
        throw std::invalid_argument("streaming::session::extend: seam mismatch");
    }

    const std::size_t post_seam_count = batch.size() - 1;

    // Already locked-out: skip the candidate build, just record the new waypoints in staging.
    if (!staged_batches_.empty()) {
        if (post_seam_count > 0) {
            staged_batches_.push_back(accumulator_tail_to_xarray(batch, 1));
        }
        last_waypoint_ = view_to_xarray(batch.at(batch.size() - 1));
        return;
    }

    // Seam-only batch with no new waypoints: nothing to do.
    if (post_seam_count == 0) {
        return;
    }

    // Build a candidate trajectory from the active waypoints plus the batch's new waypoints,
    // then find the branch: the earliest point where the candidate diverges from the current
    // active. Where that branch falls decides whether we can pivot.
    auto new_waypoints = concat_active_with_batch_tail(active_waypoints_, batch, 1);
    auto candidate = build_trajectory_from_(new_waypoints);  // throws on validation failure

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

    if (branch_ahead && has_samplable_material) {
        uniform_sampler new_sampler = uniform_sampler::quantized_for_trajectory(candidate, sample_rate_, starting_local_time);

        last_waypoint_ = row_to_xarray(new_waypoints, new_waypoints.shape(0) - 1);
        active_waypoints_ = std::move(new_waypoints);
        active_ = std::move(candidate);
        cursor_.emplace(active_->create_cursor());
        sampler_.emplace(std::move(new_sampler));
        ++generation_count_;
    } else {
        staged_batches_.push_back(accumulator_tail_to_xarray(batch, 1));
        last_waypoint_ = view_to_xarray(batch.at(batch.size() - 1));
    }
}

trajectory::seconds session::current_time() const noexcept {
    return current_time_;
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

trajectory session::build_trajectory_from_(const xt::xarray<double>& waypoints) const {
    const waypoint_accumulator acc(waypoints);
    path p = path::create(acc, path_options_);
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
    auto anchor = row_to_xarray(active_waypoints_, active_waypoints_.shape(0) - 1);

    auto new_waypoints = stack_anchor_and_staged(anchor, staged_batches_);
    auto new_active = build_trajectory_from_(new_waypoints);

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

    active_waypoints_ = std::move(new_waypoints);
    active_ = std::move(new_active);
    cursor_.emplace(active_->create_cursor());
    sampler_.emplace(std::move(new_sampler));
    epoch_ = epoch_ + old_duration;
    staged_batches_.clear();
    ++generation_count_;
}

}  // namespace viam::trajex::totg::streaming
