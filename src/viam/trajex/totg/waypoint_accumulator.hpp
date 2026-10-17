#pragma once

#include <concepts>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xadapt.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>
#endif

#include <viam/trajex/totg/export.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg {

///
/// Implementation details for waypoint_accumulator.
///
namespace waypoint_accumulator_details {

///
/// Satisfied when each row of T can be adapted where it lies, rather than copied.
///
/// Requires contiguous row-major storage of the element type waypoint_view adapts. A lazy
/// expression has nothing to point at and a strided view exposes no data interface. This is
/// about layout, not ownership: a contiguous view qualifies.
///
template <typename T>
concept rows_adaptable = xexpression_like<T> && xt::has_data_interface<std::decay_t<T>>::value &&
                         (std::decay_t<T>::static_layout == xt::layout_type::row_major) &&
                         std::same_as<typename std::decay_t<T>::value_type, xmatrix<>::value_type>;

///
/// Satisfied by anything waypoint_accumulator will read a waypoint set out of.
///
/// Admits an adaptor over a caller's own buffer or a fixed-shape array as readily as an
/// xmatrix, none of which need copying to be read row by row.
///
template <typename T>
concept waypoints_like = xrank_same_as_or_dynamic<T, xmatrix<>> && rows_adaptable<T>;

}  // namespace waypoint_accumulator_details

///
/// Accumulator for building up a sequence of waypoints.
///
/// **Lifetime requirements**: All waypoint arrays passed to this object must outlive it.
///
/// **Move semantics**: This class is move-only (not copyable).
///
/// Usage:
/// @code
///   waypoint_accumulator waypoints(initial_waypoints);
///   waypoints.add_waypoints(more_waypoints);
/// @endcode
///
class VIAM_TRAJEX_TOTG_API waypoint_accumulator {
   public:
    ///
    /// View type for individual waypoints.
    ///
    /// An adaptor over the row rather than a view of the array it came from, because a view
    /// carries the container type and so admits rows from one container only. A row of any
    /// row-major rank-2 array is contiguous, so the adaptor spells every such row the same way.
    ///
    using waypoint_view = decltype(xt::adapt(std::declval<const xmatrix<>::value_type*>(), std::size_t{}, xt::no_ownership()));

    ///
    /// Constructs with initial waypoints.
    ///
    /// @param waypoints 2D array (num_waypoints, num_joints)
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    explicit waypoint_accumulator(const xmatrix<>& waypoints);
    explicit waypoint_accumulator(xmatrix<>&& waypoints) = delete;

    ///
    /// Constructs with initial waypoints held in some other rank-2 array.
    ///
    /// @param waypoints 2D array (num_waypoints, num_joints)
    /// @throws std::invalid_argument if the array turns out not to be 2-dimensional
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    /// The xmatrix overload above is the one to reach for, and the one a near miss is diagnosed
    /// against. This exists so a caller holding waypoints in an adaptor over their own memory,
    /// or in a fixed-shape array, need not copy them first.
    ///
    /// The rank check runs either way. A rank settled at runtime needs it, a fixed one folds it
    /// away, and splitting them would mean a second overload differing by one comparison.
    ///
    template <waypoint_accumulator_details::waypoints_like T>
    explicit waypoint_accumulator(const T& waypoints) {
        require_rank_two_(waypoints);
        require_non_empty_(waypoints);
        dof_ = waypoints.shape()[1];
        append_rows_(waypoints);
    }

    // The rows point into the argument, so a temporary must be refused here as it is for
    // xmatrix above. A const rvalue reference is not a forwarding reference, so this takes
    // rvalues only and leaves the lvalue overload alone.
    template <waypoint_accumulator_details::waypoints_like T>
    explicit waypoint_accumulator(const T&&) = delete;

    // What is left cannot be read as waypoints at all: wrong rank, non-contiguous rows, the
    // wrong element type, or an expression with no storage behind it.
    template <xexpression_like T>
        requires(!waypoint_accumulator_details::waypoints_like<T>)
    explicit waypoint_accumulator(const T&) = delete;

    ///
    /// Constructs with a single waypoint view.
    ///
    /// @param first_waypoint View to first waypoint (establishes DOF)
    /// @note The underlying array must outlive the waypoint_accumulator object
    ///
    explicit waypoint_accumulator(const waypoint_view& first_waypoint);

    ///
    /// Copy constructs a waypoint_accumulator.
    ///
    waypoint_accumulator(const waypoint_accumulator&);

    ///
    /// Move constructs a waypoint_accumulator.
    ///
    waypoint_accumulator(waypoint_accumulator&&) noexcept;

    ///
    /// Copy assigns a waypoint_accumulator.
    ///
    /// @return Reference to this
    ///
    waypoint_accumulator& operator=(const waypoint_accumulator&);

    ///
    /// Move assigns a waypoint_accumulator.
    ///
    /// @return Reference to this
    ///
    waypoint_accumulator& operator=(waypoint_accumulator&&) noexcept;

    ///
    /// Adds additional waypoints.
    ///
    /// @param waypoints 2D array where each row is a waypoint, shape (num_waypoints, num_joints)
    /// @return Reference to this for method chaining
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    waypoint_accumulator& add_waypoints(const xmatrix<>& waypoints);
    waypoint_accumulator& add_waypoints(xmatrix<>&& waypoints) = delete;

    ///
    /// Adds additional waypoints held in some other rank-2 array.
    ///
    /// @param waypoints 2D array where each row is a waypoint, shape (num_waypoints, num_joints)
    /// @return Reference to this for method chaining
    /// @throws std::invalid_argument if the array turns out not to be 2-dimensional, or the DOF differs
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    template <waypoint_accumulator_details::waypoints_like T>
    waypoint_accumulator& add_waypoints(const T& waypoints) {
        require_rank_two_(waypoints);
        require_matching_dof_(waypoints.shape()[1]);
        append_rows_(waypoints);
        return *this;
    }

    template <waypoint_accumulator_details::waypoints_like T>
    waypoint_accumulator& add_waypoints(const T&&) = delete;

    template <xexpression_like T>
        requires(!waypoint_accumulator_details::waypoints_like<T>)
    waypoint_accumulator& add_waypoints(const T&) = delete;

    ///
    /// Adds a single waypoint view.
    ///
    /// @param waypoint View to waypoint to add
    /// @return Reference to this for method chaining
    /// @throws std::invalid_argument if waypoint DOF doesn't match
    /// @note The underlying array must outlive the waypoint_accumulator object
    ///
    waypoint_accumulator& add_waypoint(const waypoint_view& waypoint);

    ///
    /// Gets the number of degrees of freedom.
    ///
    /// @return Number of DOF
    ///
    size_t dof() const noexcept;

    ///
    /// Gets the number of waypoints.
    ///
    /// @return Total number of waypoints
    ///
    size_t size() const noexcept;

    ///
    /// Checks if empty.
    ///
    /// @return True if no waypoints
    ///
    bool empty() const noexcept;

    ///
    /// Iterator type for waypoint views.
    ///
    using const_iterator = std::vector<waypoint_view>::const_iterator;

    ///
    /// Value type for waypoint views.
    ///
    using value_type = waypoint_view;

    ///
    /// Size type.
    ///
    using size_type = std::size_t;

    ///
    /// Gets begin iterator.
    ///
    /// @return Iterator to first waypoint
    ///
    const_iterator begin() const noexcept;

    ///
    /// Gets end iterator.
    ///
    /// @return Iterator past last waypoint
    ///
    const_iterator end() const noexcept;

    ///
    /// Gets const begin iterator.
    ///
    /// @return Const iterator to first waypoint (same as begin())
    ///
    const_iterator cbegin() const noexcept;

    ///
    /// Gets const end iterator.
    ///
    /// @return Const iterator past last waypoint (same as end())
    ///
    const_iterator cend() const noexcept;

    ///
    /// Accesses waypoint by index (no bounds checking).
    ///
    /// @param i Index of waypoint
    /// @return Reference to waypoint view
    ///
    const waypoint_view& operator[](size_t i) const;

    ///
    /// Accesses waypoint by index with bounds checking.
    ///
    /// @param i Index of waypoint
    /// @return Reference to waypoint view
    /// @throws std::out_of_range if i >= size()
    ///
    const waypoint_view& at(size_t i) const;

    ///
    /// Accesses the last waypoint.
    ///
    /// @return Reference to last waypoint view
    ///
    const waypoint_view& back() const noexcept;

    ///
    /// Removes the last waypoint.
    ///
    void pop_back() noexcept;

   private:
    // Called by the templated overloads above, so a waypoint set is admitted on the same terms
    // however its rank became known.
    static void require_rank_two_(const auto& waypoints) {
        if (!xrank_is_same_as<xmatrix<>>(waypoints)) {
            throw std::invalid_argument{"Waypoints must be 2-dimensional"};
        }
    }

    static void require_non_empty_(const auto& waypoints) {
        if (waypoints.shape()[0] == 0) {
            throw std::invalid_argument{"Waypoints cannot be empty"};
        }
    }

    void require_matching_dof_(size_t dof) const {
        if (dof != dof_) {
            throw std::invalid_argument{"Waypoints dimensions must match existing DOF"};
        }
    }

    // Each row is contiguous because the source is dense and row-major, which is what lets the
    // row be adapted where it lies instead of copied.
    void append_rows_(const auto& waypoints) {
        const size_t num_waypoints = waypoints.shape()[0];
        for (size_t i = 0; i < num_waypoints; ++i) {
            waypoints_.push_back(xt::adapt(&waypoints(i, 0), dof_, xt::no_ownership()));
        }
    }

    size_t dof_;
    std::vector<waypoint_view> waypoints_;
};

// Verify waypoint_accumulator satisfies C++20 range concepts
static_assert(std::ranges::range<waypoint_accumulator>);
static_assert(std::ranges::sized_range<waypoint_accumulator>);
static_assert(std::ranges::forward_range<waypoint_accumulator>);

}  // namespace viam::trajex::totg
