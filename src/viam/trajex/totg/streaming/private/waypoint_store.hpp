#pragma once

// Persistent waypoint storage for a streaming session.
//
// A session outlives the batches handed to it, so it cannot retain the caller's
// `waypoint_accumulator`: that accumulator holds row-views into memory the caller owns and
// may reuse or destroy as soon as `extend` returns. The store keeps its own copy and
// presents it back as an accumulator, which is the form `path::create` consumes.
//
// Waypoints are kept in fixed-size chunks held in a list, which is what makes appending
// cheap. A session appends to the same set on every extend, so storage that had to be
// reallocated and recopied to grow would cost O(N) per extend and O(N^2) across a session.
// Here a chunk is allocated once and filled in place, a new one is linked on when it fills,
// and neither the list nodes nor the arrays inside them ever move -- which they must not,
// because the accumulator below holds views referencing those arrays by address.
//
// This is a private header: header-only, no library backing, and not installed. It exists
// so the session and the pipeline benchmarks share one definition of what accumulating
// waypoints across a session costs, which lets a change to the storage strategy be
// measured rather than guessed at.

#include <cstddef>
#include <iterator>
#include <list>
#include <optional>
#include <stdexcept>
#include <vector>

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xview.hpp>
#endif

#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg::streaming {

class waypoint_store {
   public:
    waypoint_store() = default;

    // Neither copyable nor movable, because the accumulator returned by `waypoints()` holds
    // views referencing the chunk arrays by address. Moving the store would move those
    // arrays and leave the views dangling while still appearing valid.
    waypoint_store(const waypoint_store&) = delete;
    waypoint_store& operator=(const waypoint_store&) = delete;
    waypoint_store(waypoint_store&&) = delete;
    waypoint_store& operator=(waypoint_store&&) = delete;

    std::size_t size() const noexcept {
        return size_;
    }

    std::size_t dof() const noexcept {
        return dof_;
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

    // Appends rows `[from, batch.size())`, copying them out of the batch.
    //
    // A batch arrives carrying the previous batch's final waypoint so the session can verify
    // the seam, so callers pass `from = 1` for every batch after the first to avoid storing
    // that waypoint twice.
    void append(const waypoint_accumulator& batch, std::size_t from) {
        if (from > batch.size()) {
            throw std::out_of_range("waypoint_store::append: `from` exceeds batch size");
        }
        if (from == batch.size()) {
            return;
        }
        if (dof_ != 0 && batch.dof() != dof_) {
            throw std::invalid_argument("waypoint_store::append: DOF mismatch");
        }
        dof_ = batch.dof();

        auto chunk = chunk_for_write_(size_ / k_chunk_rows);
        auto offset = size_ % k_chunk_rows;

        for (std::size_t i = from; i != batch.size(); ++i) {
            if (offset == k_chunk_rows) {
                ++chunk;
                offset = 0;
                if (chunk == chunks_.end()) {
                    chunk = allocate_chunk_();
                }
            }

            // Copied element by element rather than as `xt::view(...) = batch.at(i)`, which
            // reads better but measured 27 to 40 percent slower across every benchmark cell.
            // At six degrees of freedom, building two views and evaluating an xtensor
            // assignment through them costs more than the six stores it performs.
            const auto& row = batch.at(i);
            for (std::size_t joint = 0; joint != dof_; ++joint) {
                (*chunk)(offset, joint) = row(joint);
            }
            extend_accumulator_(*chunk, offset);

            ++offset;
            ++size_;
        }
    }

    // Shrinks the store to its first `count` rows.
    //
    // The session builds a candidate trajectory before it knows whether that candidate can
    // be pivoted onto, so an append is provisional. Recording `size()` beforehand and
    // truncating back to it abandons the candidate without disturbing what came before.
    //
    // Chunks left with nothing in them are kept rather than released, because appending and
    // truncating is the session's ordinary rhythm and freeing a chunk the moment it empties
    // would reallocate it on the next batch.
    void truncate(std::size_t count) {
        if (count > size_) {
            throw std::out_of_range("waypoint_store::truncate: `count` exceeds stored size");
        }

        while (size_ != count) {
            accumulator_->pop_back();
            --size_;
        }
        if (size_ == 0) {
            accumulator_.reset();
        }
    }

    // Discards every row but the last, which remains as the seam that following batches are
    // checked and joined against.
    //
    // A rebase abandons the trajectory the session has finished emitting and starts a new
    // one from where that trajectory ended, so nothing before its final waypoint can
    // influence what comes next.
    void reset_to_last() {
        if (size_ <= 1) {
            return;
        }

        // Copied out before anything is overwritten, since the surviving waypoint is about
        // to be written over row zero and may currently live there.
        const auto surviving = last();

        accumulator_.reset();
        size_ = 0;

        auto& first = chunks_.front();
        xt::view(first, 0, xt::all()) = surviving;
        size_ = 1;
        extend_accumulator_(first, 0);
    }

    // The stored waypoints in the form `path::create` accepts.
    //
    // The returned reference is invalidated by any subsequent mutation of the store, which
    // may append to or pop from the accumulator it refers to.
    const waypoint_accumulator& waypoints() const {
        if (!accumulator_) {
            throw std::out_of_range("waypoint_store::waypoints: store is empty");
        }
        return *accumulator_;
    }

    // The final stored row, copied out so it survives later mutation.
    xvector<> last() const {
        if (empty()) {
            throw std::out_of_range("waypoint_store::last: store is empty");
        }
        const auto index = size_ - 1;
        return xvector<>{xt::view(chunk_at_(index / k_chunk_rows), index % k_chunk_rows, xt::all())};
    }

   private:
    // Rows per chunk. At six degrees of freedom a chunk is roughly 48 KiB, so a session
    // holding tens of thousands of waypoints costs a few dozen allocations, and a short
    // move wastes at most one chunk's worth of unused rows.
    static constexpr std::size_t k_chunk_rows = 1024;

    using chunk_list = std::list<xmatrix<>>;

    // Every chunk but the one currently being filled is exactly full, so a row's position
    // follows from its index alone and chunks need carry no fill count of their own.
    chunk_list::iterator allocate_chunk_() {
        chunks_.emplace_back(xmatrix<>::from_shape(std::vector<std::size_t>{k_chunk_rows, dof_}));
        return std::prev(chunks_.end());
    }

    chunk_list::iterator chunk_for_write_(std::size_t index) {
        while (chunks_.size() <= index) {
            allocate_chunk_();
        }
        return std::next(chunks_.begin(), static_cast<chunk_list::difference_type>(index));
    }

    const xmatrix<>& chunk_at_(std::size_t index) const {
        return *std::next(chunks_.begin(), static_cast<chunk_list::difference_type>(index));
    }

    // Grown one waypoint at a time as rows are written, rather than rebuilt when read,
    // because rebuilding costs a view per stored waypoint and the session reads it on every
    // extend. The views stay valid because chunk arrays are never reallocated or moved.
    void extend_accumulator_(const xmatrix<>& chunk, std::size_t offset) {
        auto row = xt::view(chunk, offset, xt::all());
        if (accumulator_) {
            accumulator_->add_waypoint(row);
        } else {
            accumulator_.emplace(row);
        }
    }

    chunk_list chunks_;
    std::size_t size_ = 0;
    std::size_t dof_ = 0;
    std::optional<waypoint_accumulator> accumulator_;
};

}  // namespace viam::trajex::totg::streaming
