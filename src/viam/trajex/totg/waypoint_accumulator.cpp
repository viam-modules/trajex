#include <viam/trajex/totg/waypoint_accumulator.hpp>

#include <stdexcept>

namespace viam::trajex::totg {

waypoint_accumulator::waypoint_accumulator(const xmatrix<>& waypoints) {
    require_non_empty_(waypoints);

    dof_ = waypoints.shape()[1];

    // Rows adapt the caller's array rather than copying it, which is what makes path creation
    // cheap to feed. The caller must ensure the array outlives this accumulator.
    append_rows_(waypoints);
}

waypoint_accumulator::waypoint_accumulator(const waypoint_view& first_waypoint) {
    dof_ = first_waypoint.shape()[0];
    waypoints_.push_back(first_waypoint);
}

waypoint_accumulator::waypoint_accumulator(const waypoint_accumulator&) = default;
waypoint_accumulator::waypoint_accumulator(waypoint_accumulator&&) noexcept = default;

// The default copy assignment cannot work: std::vector assigns through the elements it already
// holds, and assigning to a row adaptor writes into whatever that row points at, which is
// const. Copy construction builds fresh adaptors instead, so defer to it.
waypoint_accumulator& waypoint_accumulator::operator=(const waypoint_accumulator& other) {
    if (this != &other) {
        auto copy = other;
        *this = std::move(copy);
    }
    return *this;
}

waypoint_accumulator& waypoint_accumulator::operator=(waypoint_accumulator&&) noexcept = default;

waypoint_accumulator& waypoint_accumulator::add_waypoints(const xmatrix<>& waypoints) {
    require_matching_dof_(waypoints);
    append_rows_(waypoints);
    return *this;
}

waypoint_accumulator& waypoint_accumulator::add_waypoint(const waypoint_view& waypoint) {
    if (waypoint.shape()[0] != dof_) {
        throw std::invalid_argument{"Waypoint DOF must match existing DOF"};
    }
    waypoints_.push_back(waypoint);
    return *this;
}

size_t waypoint_accumulator::dof() const noexcept {
    return dof_;
}

size_t waypoint_accumulator::size() const noexcept {
    return waypoints_.size();
}

bool waypoint_accumulator::empty() const noexcept {
    return waypoints_.empty();
}

waypoint_accumulator::const_iterator waypoint_accumulator::begin() const noexcept {
    return waypoints_.cbegin();
}

waypoint_accumulator::const_iterator waypoint_accumulator::end() const noexcept {
    return waypoints_.cend();
}

waypoint_accumulator::const_iterator waypoint_accumulator::cbegin() const noexcept {
    return waypoints_.cbegin();
}

waypoint_accumulator::const_iterator waypoint_accumulator::cend() const noexcept {
    return waypoints_.cend();
}

const waypoint_accumulator::waypoint_view& waypoint_accumulator::operator[](size_t i) const {
    return waypoints_[i];
}

const waypoint_accumulator::waypoint_view& waypoint_accumulator::at(size_t i) const {
    if (i >= waypoints_.size()) [[unlikely]] {
        throw std::out_of_range{"waypoint_accumulator::at: index out of range"};
    }
    return waypoints_[i];
}

const waypoint_accumulator::waypoint_view& waypoint_accumulator::back() const noexcept {
    return waypoints_.back();
}

void waypoint_accumulator::pop_back() noexcept {
    waypoints_.pop_back();
}

}  // namespace viam::trajex::totg
