#pragma once

#include <bitset>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg {

///
/// Geometric path through configuration space with linear segments and circular blends.
///
/// Provides queries for segment lookup and arc length parameterization.
///
/// **Ownership**: Owns all segment data. Safe to use after source waypoints are destroyed.
///
/// **Thread safety**: All const methods are thread-safe for concurrent access.
///
/// Example usage:
/// @code
///   // Create path from waypoints
///   xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 0.0}};
///   path p = path::create(waypoints);
///
///   // Query path at specific arc length
///   arc_length s{0.5};
///   auto config = p.configuration(s);
///   auto tangent = p.tangent(s);
///   auto curvature = p.curvature(s);
///
///   // Iterate over segments
///   for (const auto& view : p) {
///       view.visit([](const auto& seg_data) {
///           // Process segment data
///       });
///   }
/// @endcode
///
class path {
   public:
    ///
    /// Geometric segment in configuration space.
    ///
    class segment {
       public:
        ///
        /// Linear segment in configuration space.
        ///
        struct linear {
            ///
            /// Constructs linear segment from start and end configurations.
            ///
            /// Computes unit_direction and length from endpoints.
            ///
            /// @param start Starting configuration
            /// @param end Ending configuration
            /// @throws std::invalid_argument if start == end
            ///
            linear(xvector<> start, const xvector<>& end);

            ///
            /// Constructs linear segment from precomputed components.
            ///
            /// Use when direction and length are already known to avoid recomputation.
            /// Caller is responsible for ensuring unit_direction is normalized and non-zero.
            ///
            /// @param start Starting configuration
            /// @param unit_direction Precomputed unit direction vector
            /// @param length Arc length (must be positive)
            /// @throws std::invalid_argument if length is not positive
            ///
            linear(xvector<> start, xvector<> unit_direction, arc_length length);

            xvector<> start;           ///< Starting configuration
            xvector<> unit_direction;  ///< Precomputed unit direction vector (normalized end-start)
            arc_length length;         ///< Precomputed length (norm of end-start)
        };

        ///
        /// Circular blend segment in configuration space.
        ///
        struct circular {
            ///
            /// Constructs circular arc with validation.
            ///
            /// @param center Center of arc in configuration space
            /// @param x First basis vector (must be unit length)
            /// @param y Second basis vector (must be unit length and perpendicular to x)
            /// @param radius Radius of the circular arc
            /// @param angle_rads Total angle swept by arc (radians)
            /// @throws std::invalid_argument if x,y are not orthonormal
            ///
            circular(xvector<> center, xvector<> x, xvector<> y, double radius, double angle_rads);

            xvector<> center;   ///< Center of arc in configuration space
            xvector<> x;        ///< First basis vector (defines rotation plane)
            xvector<> y;        ///< Second basis vector (perpendicular to x)
            double radius;      ///< Radius of the circular arc (units: configuration space distance)
            double angle_rads;  ///< Total angle swept by arc (units: radians)
        };

        ///
        /// View of a segment's position within the path.
        ///
        /// Provides the primary query interface for segments, mapping global path
        /// arc lengths to local segment parameters.
        ///
        class view {
           public:
            ///
            /// Constructs view from segment reference and arc length bounds.
            ///
            /// @param seg Segment to view
            /// @param start Starting arc length on path
            /// @param end Ending arc length on path
            ///
            view(const class segment& seg, arc_length start, arc_length end) noexcept;

            ///
            /// Gets the underlying segment.
            ///
            /// @return Reference to the segment
            ///
            const class segment& segment() const noexcept;

            ///
            /// Gets the starting arc length on path.
            ///
            /// @return Starting arc length
            ///
            arc_length start() const noexcept;

            ///
            /// Gets the ending arc length on path.
            ///
            /// @return Ending arc length
            ///
            arc_length end() const noexcept;

            ///
            /// Gets the length of this view.
            ///
            /// @return Length of the view (end - start)
            ///
            arc_length length() const noexcept;

            ///
            /// Checks if segment holds a specific type.
            ///
            /// @tparam T Type to check (segment::linear or segment::circular)
            /// @return True if segment holds type T
            ///
            template <typename T>
            bool is() const noexcept {
                return std::holds_alternative<T>(seg_.get().data_);
            }

            ///
            /// Visits the underlying segment variant.
            ///
            /// @tparam Visitor Callable object type
            /// @param v Visitor to apply to the segment variant
            /// @return Result of visiting the variant
            ///
            template <typename Visitor>
            decltype(auto) visit(Visitor&& v) const {
                return std::visit(std::forward<Visitor>(v), seg_.get().data_);
            }

            ///
            /// Gets configuration at global arc length.
            ///
            /// @param s Global arc length on path
            /// @return Configuration vector at arc length s
            ///
            xvector<> configuration(arc_length s) const;

            ///
            /// Gets tangent vector at global arc length.
            ///
            /// @param s Global arc length on path
            /// @return Unit tangent vector at arc length s
            ///
            xvector<> tangent(arc_length s) const;

            ///
            /// Gets curvature vector at global arc length.
            ///
            /// @param s Global arc length on path
            /// @return Curvature vector at arc length s
            ///
            xvector<> curvature(arc_length s) const;

            ///
            /// Writes configuration at global arc length into caller-provided storage.
            ///
            /// The value-returning overload allocates a fresh array per call, which the
            /// integrator cannot afford at the rate it queries geometry. This overload
            /// performs the same computation writing into storage the caller already owns.
            ///
            /// @param s Global arc length on path
            /// @param out Destination, sized to the path's degrees of freedom
            /// @throws std::out_of_range if s lies outside this view's bounds
            /// @throws std::invalid_argument if out is not sized to the degrees of freedom
            ///
            void configuration(arc_length s, std::span<double> out) const;

            ///
            /// Writes tangent vector at global arc length into caller-provided storage.
            ///
            /// @param s Global arc length on path
            /// @param out Destination, sized to the path's degrees of freedom
            /// @throws std::out_of_range if s lies outside this view's bounds
            /// @throws std::invalid_argument if out is not sized to the degrees of freedom
            ///
            void tangent(arc_length s, std::span<double> out) const;

            ///
            /// Writes curvature vector at global arc length into caller-provided storage.
            ///
            /// @param s Global arc length on path
            /// @param out Destination, sized to the path's degrees of freedom
            /// @throws std::out_of_range if s lies outside this view's bounds
            /// @throws std::invalid_argument if out is not sized to the degrees of freedom
            ///
            void curvature(arc_length s, std::span<double> out) const;

           private:
            // Degrees of freedom of the viewed segment, recovered from its stored vectors.
            // Only the value-returning accessors need this; the filling ones take the size
            // from the caller's span and validate it against the segment in place.
            std::size_t dof_() const;

            // Reference to the segment
            std::reference_wrapper<const class segment> seg_;

            // Arc length bounds on path
            arc_length start_;
            arc_length end_;
        };

        ///
        /// Constructs segment from linear data.
        ///
        /// @param data Linear segment data
        ///
        explicit segment(linear data);

        ///
        /// Constructs segment from circular data.
        ///
        /// @param data Circular segment data
        ///
        explicit segment(circular data);

       private:
        using detail = std::variant<linear, circular>;

        // Variant holding either linear or circular segment data
        detail data_;
    };

    ///
    /// Options for path creation with fluent interface.
    ///
    class options {
       public:
        ///
        /// Default maximum deviation.
        ///
        static constexpr double k_default_max_deviation = 0.0;

        ///
        /// Default minimum blend curvature (1/radius in configuration space).
        ///
        static constexpr double k_default_min_blend_curvature = 1e-5;

        ///
        /// Default maximum blend curvature (1/radius in configuration space).
        ///
        static constexpr double k_default_max_blend_curvature = 1e5;

        ///
        /// Constructs options with default values.
        ///
        options();

        ///
        /// Sets maximum blend deviation.
        ///
        /// @param deviation Maximum distance blend arc can deviate from corner
        /// @return Reference to this for method chaining
        ///
        options& set_max_deviation(double deviation);

        ///
        /// Sets maximum blend deviation.
        ///
        /// @param deviation Maximum distance blend arc can deviate from corner
        /// @return Reference to this for method chaining
        /// @note Intended primarily for testing
        ///
        options& set_max_blend_deviation(double deviation);

        ///
        /// Sets maximum linear deviation for coalescing.
        ///
        /// This is `Divergent Behavior 1`: the Kunz & Stilman paper does not include this pass.
        ///
        /// @param deviation Maximum deviation for waypoint coalescing
        /// @return Reference to this for method chaining
        /// @note Intended primarily for testing
        ///
        options& set_max_linear_deviation(double deviation);

        ///
        /// Sets minimum blend curvature.
        ///
        /// Blend arcs whose natural radius would exceed 1/curvature are capped at that
        /// radius. This prevents numerically fragile enormous-radius arcs at near-collinear
        /// waypoints while retaining exact C1 continuity at every segment boundary.
        ///
        /// This is `Divergent Behavior 3`: the Kunz & Stilman paper uses no bounds on blend curvature.
        ///
        /// @param curvature Minimum acceptable blend curvature (1/radius)
        /// @return Reference to this for method chaining
        ///
        options& set_min_blend_curvature(double curvature);

        ///
        /// Sets maximum blend curvature.
        ///
        /// Waypoints whose blend arc would exceed this curvature (1/radius) are treated
        /// as unblended corners. This prevents near-degenerate tiny arcs at near-reversal
        /// waypoints.
        ///
        /// This is `Divergent Behavior 3`: the Kunz & Stilman paper uses no bounds on blend curvature.
        ///
        /// @param curvature Maximum acceptable blend curvature (1/radius)
        /// @return Reference to this for method chaining
        ///
        options& set_max_blend_curvature(double curvature);

        ///
        /// Gets maximum blend deviation.
        ///
        /// @return Maximum distance blend arc can deviate from corner
        ///
        double max_blend_deviation() const noexcept;

        ///
        /// Gets maximum linear deviation.
        ///
        /// @return Maximum deviation for waypoint coalescing
        ///
        double max_linear_deviation() const noexcept;

        ///
        /// Gets minimum blend curvature.
        ///
        /// @return Minimum acceptable blend curvature (1/radius)
        ///
        double min_blend_curvature() const noexcept;

        ///
        /// Gets maximum blend curvature.
        ///
        /// @return Maximum acceptable blend curvature (1/radius)
        ///
        double max_blend_curvature() const noexcept;

       private:
        double max_blend_deviation_;
        double max_linear_deviation_;
        double min_blend_curvature_;
        double max_blend_curvature_;
    };

    ///
    /// Creates a path from a waypoints accumulator.
    ///
    /// @param waypoints Waypoint sequence to follow
    /// @param opts Path creation options (coalescing and blending parameters)
    /// @return Constructed path with segments
    ///
    [[nodiscard]] static path create(const waypoint_accumulator& waypoints, const options& opts = options{});

    ///
    /// Creates path directly from waypoint array.
    ///
    /// Convenience overload that constructs waypoint_accumulator internally.
    ///
    /// @param waypoints 2D array (num_waypoints, dof) of waypoints
    /// @param opts Path creation options (coalescing and blending parameters)
    /// @return Constructed path with segments
    ///
    [[nodiscard]] static path create(const xmatrix<>& waypoints, const options& opts = options{});

    ///
    /// Gets total arc length of path.
    ///
    /// @return Total length in configuration space
    ///
    arc_length length() const noexcept;

    ///
    /// Gets number of segments.
    ///
    /// @return Number of linear and circular segments
    ///
    size_t size() const noexcept;

    ///
    /// Checks if path is empty.
    ///
    /// @return True if path has no segments
    ///
    bool empty() const noexcept;

    ///
    /// Forward iterator over segment views.
    ///
    class const_iterator;

    ///
    /// Gets begin iterator for segment views.
    ///
    /// @return Iterator to first segment
    ///
    const_iterator begin() const noexcept;

    ///
    /// Gets end iterator for segment views.
    ///
    /// @return Iterator past last segment
    ///
    const_iterator end() const noexcept;

    ///
    /// Gets const begin iterator for segment views.
    ///
    /// @return Const iterator to first segment (same as begin())
    ///
    const_iterator cbegin() const noexcept;

    ///
    /// Gets const end iterator for segment views.
    ///
    /// @return Const iterator past last segment (same as end())
    ///
    const_iterator cend() const noexcept;

    ///
    /// Gets number of degrees of freedom.
    ///
    /// @return Number of DOF
    ///
    size_t dof() const noexcept;

    ///
    /// Evaluates path at given arc length.
    ///
    /// @param s Arc length along path
    /// @return View of the segment containing s, with segment's arc length bounds
    ///
    segment::view operator()(arc_length s) const;

    ///
    /// Gets configuration at arc length.
    ///
    /// @param s Arc length along path
    /// @return Configuration vector at s
    ///
    xvector<> configuration(arc_length s) const;

    ///
    /// Gets tangent at arc length.
    ///
    /// @param s Arc length along path
    /// @return Unit tangent vector at s
    ///
    xvector<> tangent(arc_length s) const;

    ///
    /// Gets curvature at arc length.
    ///
    /// @param s Arc length along path
    /// @return Curvature vector at s
    ///
    xvector<> curvature(arc_length s) const;

    ///
    /// Cursor for efficient sequential traversal.
    ///
    class cursor;

    ///
    /// Creates a cursor at specified arc length position.
    ///
    /// Cursors provide efficient sequential traversal with O(1) amortized access via hints.
    /// Multiple cursors can exist on the same path (e.g., for forward/backward integration).
    ///
    /// @param s Starting position (default: 0, path start). Clamped to [0, length].
    /// @return Cursor positioned at s
    /// @throws std::invalid_argument if path is empty
    ///
    [[nodiscard]] cursor create_cursor(arc_length s = arc_length{0.0}) const;

   private:
    // Internal storage: segment with its starting position on the path
    struct positioned_segment {
        segment seg;
        arc_length start;
    };

    path(std::vector<positioned_segment> segments, size_t dof, arc_length length);

    std::vector<positioned_segment> segments_;
    size_t dof_{0};
    arc_length length_{0.0};

    friend class const_iterator;
    friend class cursor;
};

///
/// Forward iterator for path segments, yielding segment::view on dereference.
///
/// Dereferencing returns segment::view by value, not by reference.
/// Each dereference constructs a new view object. This is intentional - views are
/// lightweight (reference + 2 arc_lengths) and computed on-the-fly from storage.
///
/// Usage:
/// @code
///   auto it = path.begin();
///   auto view = *it;        // Copy view by value (recommended)
///   auto& ref = *it;        // Binds to temporary - avoid!
///
///   // Multiple dereferences create distinct objects:
///   &(*it) != &(*it)        // true - different addresses
/// @endcode
///
class path::const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = segment::view;
    using difference_type = std::ptrdiff_t;
    using pointer = const segment::view*;
    using reference = segment::view;

    ///
    /// Default constructs singular iterator.
    ///
    const_iterator() noexcept;

    ///
    /// Dereferences to get segment view.
    ///
    /// @return Segment view for current segment
    ///
    segment::view operator*() const;

    ///
    /// Pre-increments iterator.
    ///
    /// @return Reference to this
    ///
    const_iterator& operator++() noexcept;

    ///
    /// Post-increments iterator.
    ///
    /// @return Iterator to previous position
    ///
    const_iterator operator++(int) noexcept;

    ///
    /// Pre-decrements iterator.
    ///
    /// @return Reference to this
    ///
    const_iterator& operator--() noexcept;

    ///
    /// Post-decrements iterator.
    ///
    /// @return Iterator to previous position
    ///
    const_iterator operator--(int) noexcept;

    ///
    /// Compares iterators for equality.
    ///
    /// @param other Iterator to compare with
    /// @return True if iterators point to same position
    ///
    bool operator==(const const_iterator& other) const noexcept;

   private:
    friend class path;

    const_iterator(const path* p, std::vector<positioned_segment>::const_iterator it) noexcept : path_{p}, it_{it} {}

    const path* path_;
    std::vector<positioned_segment>::const_iterator it_;
};

///
/// Cursor for efficient sequential traversal of path with hint optimization.
///
/// Maintains current arc length position and segment hint for O(1) amortized
/// access during sequential traversal. Supports bidirectional traversal for
/// forward and backward integration in TOTG algorithm.
///
/// **Hint optimization**: Tracks last-accessed segment to avoid binary search
/// on sequential access. Provides O(1) amortized lookup vs O(log n) per query.
///
/// **Thread safety**: Not thread-safe. Each integration pass should use its own cursor.
///
/// **Semantics**: Cursor is a view-like object with internal state. Cursors are copyable
/// to support snapshots (e.g., for forward/backward integration passes).
///
/// **Geometry queries allocate**: configuration(), tangent() and curvature() each return a
/// freshly allocated array. Code that queries geometry repeatedly should either fill storage
/// it owns through the span overloads, or hold a cursor::rich obtained from enrich(), which
/// owns that storage and reuses it.
///
/// Example usage:
/// @code
///   path p = path::create(waypoints);
///
///   // Forward integration
///   for (auto c = p.create_cursor(); c != c.end(); c.seek_by(arc_length{0.01})) {
///       auto config = c.configuration();
///       auto tangent = c.tangent();
///       // ... process ...
///   }
///
///   // Backward integration
///   auto c = p.create_cursor(p.length());  // Start at end
///   while (c.position() > arc_length{0.0}) {
///       auto config = c.configuration();
///       c.seek_by(arc_length{-0.01});  // negative delta
///   }
/// @endcode
///
class path::cursor {
   public:
    ///
    /// Gets the path being traversed.
    ///
    /// @return Reference to the path
    ///
    const class path& path() const noexcept;

    ///
    /// Gets current arc length position.
    ///
    /// @return Current position along path
    ///
    arc_length position() const noexcept;

    ///
    /// Dereferences cursor to get segment view at current position.
    ///
    /// Returns a view of the segment containing the current cursor position,
    /// including the segment's arc length bounds on the path.
    ///
    /// @return View of segment at current position
    /// @note O(1) - uses cached hint
    ///
    segment::view operator*() const;

    ///
    /// Seeks to specific arc length position (absolute positioning).
    ///
    /// Sets cursor position to target arc length. Clamps to [0, infinity).
    /// If target exceeds path length, cursor is set to infinity (sentinel position).
    ///
    /// @param s Target arc length position
    /// @return Reference to this cursor for method chaining
    ///
    cursor& seek(arc_length s) noexcept;

    ///
    /// Seeks along path by arc length delta (relative positioning).
    ///
    /// Advances cursor by arc length offset. Clamps to [0, infinity).
    /// If result exceeds path length, cursor is set to infinity (sentinel position).
    /// Supports bidirectional traversal (positive = forward, negative = backward).
    ///
    /// @param delta Arc length offset
    /// @return Reference to this cursor for method chaining
    ///
    cursor& seek_by(arc_length delta) noexcept;

    ///
    /// Gets configuration at current position.
    ///
    /// @return Configuration vector
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    ///
    xvector<> configuration() const;

    ///
    /// Gets tangent at current position.
    ///
    /// @return Unit tangent vector
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    ///
    xvector<> tangent() const;

    ///
    /// Gets curvature at current position.
    ///
    /// @return Curvature vector
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    ///
    xvector<> curvature() const;

    ///
    /// Writes configuration at current position into caller-provided storage.
    ///
    /// @param out Destination, sized to the path's degrees of freedom
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    /// @throws std::invalid_argument if out is not sized to the degrees of freedom
    ///
    void configuration(std::span<double> out) const;

    ///
    /// Writes tangent at current position into caller-provided storage.
    ///
    /// @param out Destination, sized to the path's degrees of freedom
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    /// @throws std::invalid_argument if out is not sized to the degrees of freedom
    ///
    void tangent(std::span<double> out) const;

    ///
    /// Writes curvature at current position into caller-provided storage.
    ///
    /// @param out Destination, sized to the path's degrees of freedom
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    /// @throws std::invalid_argument if out is not sized to the degrees of freedom
    ///
    void curvature(std::span<double> out) const;

    ///
    /// Cursor that owns and reuses storage for the geometry at its current position.
    ///
    class rich;

    ///
    /// Promotes a copy of this cursor to one that caches geometry.
    ///
    /// The returned cursor starts at this cursor's position with an empty cache; this
    /// cursor is unaffected. Use it where the same position is queried more than once, or
    /// where geometry is queried often enough that per-call allocation matters.
    ///
    /// @return Rich cursor at this cursor's position
    ///
    [[nodiscard]] rich enrich() const;

    ///
    /// Gets sentinel for end-of-path comparison.
    ///
    /// Returns a sentinel value that compares equal to cursors positioned
    /// past the end of the path. Useful for detecting when iteration should stop.
    ///
    /// @return Sentinel value for comparison
    ///
    std::default_sentinel_t end() const noexcept;

    ///
    /// Returns an iterator to the segment containing the current cursor position.
    ///
    /// By analogy with std::reverse_iterator::base(), unwraps the cursor to its
    /// underlying iterator primitive. The caller may decrement or increment the
    /// returned iterator to inspect adjacent segments without moving the cursor.
    ///
    /// If the cursor is singular (past end or before start), returns path().end() —
    /// the only singular iterator on path.
    ///
    /// @return Iterator to the segment containing the current position, or path().end() if singular
    ///
    path::const_iterator base() const noexcept;

    ///
    /// Compares cursor with end sentinel.
    ///
    /// @param c Cursor to compare
    /// @return True if cursor is at sentinel position (past end or invalid)
    ///
    friend bool operator==(const cursor& c, std::default_sentinel_t) noexcept;

    ///
    /// Compares end sentinel with cursor (reversed order).
    ///
    /// @param c Cursor to compare
    /// @return True if cursor is at sentinel position (past end or invalid)
    ///
    friend bool operator==(std::default_sentinel_t, const cursor& c) noexcept;

   private:
    friend class path;

    // Construct cursor at given position
    // @param p Path to traverse (must outlive cursor)
    // @param s Initial position
    explicit cursor(const class path* p, arc_length s);

    // Update hint to match current position
    // Called after position changes to maintain O(1) amortized access
    void update_hint_() noexcept;

    // Path being traversed
    const class path* path_;

    // Current position along path
    arc_length position_;

    // Hint: iterator to segment containing current position
    // Maintained by update_hint_() for O(1) amortized lookups
    // Invariant: If position_ is within [hint_->start, hint_->end), then hint_ points to correct segment
    path::const_iterator hint_;
};

///
/// Cursor that owns and reuses storage for the geometry at its current position.
///
/// A plain cursor allocates a fresh array on every geometry query. This one allocates its
/// storage once, fills each component on first use after a move, and hands back a reference
/// to it. Where the integrator queries geometry a million times per trajectory, that is the
/// difference between a million allocations and three.
///
/// **The returned references are windows, not snapshots.** Storage is allocated once and
/// lives as long as the cursor, so a reference obtained from any accessor stays valid. Its
/// *contents* track the cursor: after a seek the stale values remain readable until someone
/// asks for that component again, at which point the reference begins reporting the new
/// position. Nothing announces the change, and it can be triggered by a query made anywhere
/// else holding the same cursor. Code that needs a value outliving the next seek must copy it.
///
/// **Relationship to path::cursor**: inherited privately, so a rich cursor cannot be handed
/// out as a plain one. That is deliberate. Relocating through a base reference would move the
/// position without clearing the cache, leaving geometry that reads as valid but belongs to
/// the position before last. Use plain() for an explicit, independent copy of the position.
///
/// **Thread safety**: Not thread-safe, and less so than a plain cursor: the accessors fill
/// storage and are const, so concurrent reads of the same rich cursor race.
///
/// Copying copies the cache along with the position.
///
class path::cursor::rich : private path::cursor {
   public:
    ///
    /// Constructs a rich cursor at the position of an existing cursor, with an empty cache.
    ///
    /// @param c Cursor whose position to adopt
    ///
    explicit rich(cursor c);

    using cursor::base;
    using cursor::end;
    using cursor::path;
    using cursor::position;
    using cursor::operator*;

    ///
    /// Gets an independent plain cursor at this cursor's position.
    ///
    /// The result carries no cache and moves independently of this one.
    ///
    /// @return Copy of the underlying cursor
    ///
    [[nodiscard]] cursor plain() const;

    ///
    /// Seeks to specific arc length position, discarding cached geometry.
    ///
    /// @param s Target arc length position
    /// @return Reference to this cursor for method chaining
    ///
    rich& seek(arc_length s) noexcept;

    ///
    /// Seeks along path by arc length delta, discarding cached geometry.
    ///
    /// @param delta Arc length offset
    /// @return Reference to this cursor for method chaining
    ///
    rich& seek_by(arc_length delta) noexcept;

    ///
    /// Gets configuration at current position, computing it if not already cached.
    ///
    /// @return Reference to storage owned by this cursor; see the class note on windows
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    ///
    const xvector<>& configuration() const;

    ///
    /// Gets tangent at current position, computing it if not already cached.
    ///
    /// @return Reference to storage owned by this cursor; see the class note on windows
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    ///
    const xvector<>& tangent() const;

    ///
    /// Gets curvature at current position, computing it if not already cached.
    ///
    /// @return Reference to storage owned by this cursor; see the class note on windows
    /// @throws std::out_of_range if cursor is at sentinel position or before start
    ///
    const xvector<>& curvature() const;

    ///
    /// Compares cursor with end sentinel.
    ///
    /// @param c Cursor to compare
    /// @return True if cursor is at sentinel position (past end or invalid)
    ///
    friend bool operator==(const rich& c, std::default_sentinel_t) noexcept {
        return static_cast<const cursor&>(c) == std::default_sentinel;
    }

    ///
    /// Compares end sentinel with cursor (reversed order).
    ///
    /// @param c Cursor to compare
    /// @return True if cursor is at sentinel position (past end or invalid)
    ///
    friend bool operator==(std::default_sentinel_t, const rich& c) noexcept {
        return static_cast<const cursor&>(c) == std::default_sentinel;
    }

   private:
    // Position of each component within cached_bits_. Keeping them in one bitset means
    // invalidation clears every component by construction, rather than by remembering to
    // clear each one.
    static constexpr std::size_t k_configuration_bit_ = 0;
    static constexpr std::size_t k_tangent_bit_ = 1;
    static constexpr std::size_t k_curvature_bit_ = 2;
    static constexpr std::size_t k_cached_bit_count_ = 3;

    // Discards every cached component. Called on any move, since all three belong to the
    // position the cursor just left.
    void invalidate_() noexcept;

    // Fills `storage` if `bit` is not already set, then sets it. A throwing fill leaves the
    // bit clear, so a query that failed is not later mistaken for one that succeeded.
    template <typename Fill>
    const xvector<>& cached_(xvector<>& storage, std::size_t bit, Fill&& fill) const {
        if (!cached_bits_.test(bit)) {
            std::forward<Fill>(fill)(std::span<double>{storage.data(), storage.size()});
            cached_bits_.set(bit);
        }

        return storage;
    }

    // Storage is filled by const accessors, so it and the bits are mutable. The storage is
    // never released while the cursor lives, which is what lets the accessors return
    // references at all -- an optional<> here would free and reallocate on every
    // invalidation, reintroducing exactly the cost this type exists to remove.
    mutable xvector<> configuration_;
    mutable xvector<> tangent_;
    mutable xvector<> curvature_;
    mutable std::bitset<k_cached_bit_count_> cached_bits_;
};

///
/// Requirements shared by path::cursor and path::cursor::rich.
///
/// The two are interchangeable in generic code, with one caveat that the type system cannot
/// express: a plain cursor returns geometry by value, while a rich cursor returns a reference
/// into storage the next seek overwrites. Binding with `const auto&` is correct for both, but
/// generic code must hold such a reference no longer than the rich cursor's rule allows.
///
template <typename C>
concept cursor_like = requires(C& c, const C& cc, arc_length s) {
    { cc.path() } -> std::same_as<const path&>;
    { cc.position() } -> std::same_as<arc_length>;
    { *cc } -> std::same_as<path::segment::view>;
    { cc.base() } -> std::same_as<path::const_iterator>;
    { cc.end() } -> std::same_as<std::default_sentinel_t>;
    { c.seek(s) } -> std::same_as<C&>;
    { c.seek_by(s) } -> std::same_as<C&>;
    { cc.configuration() } -> std::convertible_to<const xvector<>&>;
    { cc.tangent() } -> std::convertible_to<const xvector<>&>;
    { cc.curvature() } -> std::convertible_to<const xvector<>&>;
    { cc == std::default_sentinel } -> std::same_as<bool>;
};

static_assert(cursor_like<path::cursor>);
static_assert(cursor_like<path::cursor::rich>);

///
/// ADL-findable end sentinel for cursors.
///
/// Returns a sentinel value that can be compared with cursors to detect
/// end-of-path. This free function enables ADL and provides an alternative
/// to the member function cursor.end(). It is constrained rather than taking
/// path::cursor directly because a rich cursor does not convert to one.
///
/// @return Sentinel value for comparison
///
template <cursor_like C>
constexpr std::default_sentinel_t end(const C&) noexcept {
    return std::default_sentinel;
}

// Verify that path satisfies range concepts
static_assert(std::ranges::range<path>);
static_assert(std::ranges::sized_range<path>);
static_assert(std::ranges::bidirectional_range<path>);

}  // namespace viam::trajex::totg
