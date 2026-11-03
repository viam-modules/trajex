#pragma once

// This header, and the rest of the public compat interface, is C++03 so that code written
// against the reference implementation can include it unchanged. Only the implementation
// requires a modern compiler.

#include <list>
#include <utility>

#include <Eigen/Core>

#include <viam/trajex/totg/compat/export.hpp>

// std::exception_ptr is older than feature-test macros, so we test the language version instead.
// MSVC reports a C++98 __cplusplus unless built with /Zc:__cplusplus, which is why we also look at
// _MSVC_LANG.
#if __cplusplus >= 201103L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
#include <exception>
#endif

namespace viam {
namespace trajex {
namespace totg {
namespace compat {

class Trajectory;

///
/// The `Path` interface of the Kunz & Stilman reference implementation, backed by trajex.
///
/// Code written against the original `Path` compiles against this one unchanged. It behaves
/// differently in these ways:
///
/// - Construction never throws for bad input. A path that cannot be built reports a length of
///   zero, returns empty vectors from the geometry queries, and makes any `Trajectory` built
///   from it invalid. `getError` reports why.
/// - `maxDeviation` controls corner blending only, as in the original. trajex's merging of
///   near-collinear waypoints is not applied.
/// - Arc length arguments are clamped to `[0, getLength()]`. The original extrapolates past the
///   end and reads out of bounds before the start.
/// - Switching points are not reported, so `getSwitchingPoints` returns an empty list and
///   `getNextSwitchingPoint` reports the end of the path.
/// - `PathSegment` is not provided. The original never hands one out.
/// - Copy assignment is a deep copy. The original's generated assignment shares segments
///   between the two paths and deletes them twice.
///
class VIAM_TRAJEX_TOTG_COMPAT_API Path {
   public:
    ///
    /// Builds a path through waypoints, blending corners with circular arcs.
    ///
    /// @param path Waypoints, all of the same dimension
    /// @param maxDeviation Maximum distance a blend may deviate from its corner. Zero disables
    ///        blending.
    ///
    Path(const std::list<Eigen::VectorXd>& path, double maxDeviation = 0.0);

    ///
    /// Copy constructs a path.
    ///
    Path(const Path& path);

    ///
    /// Copy assigns a path.
    ///
    Path& operator=(const Path& path);

    ///
    /// Destroys a path.
    ///
    ~Path();

    ///
    /// Gets the arc length of the path.
    ///
    /// @return Arc length, or zero if construction failed
    ///
    double getLength() const;

    ///
    /// Gets the configuration at an arc length.
    ///
    /// @param s Arc length, clamped to `[0, getLength()]`
    /// @return Configuration, or an empty vector if construction failed or `s` is NaN
    ///
    Eigen::VectorXd getConfig(double s) const;

    ///
    /// Gets the unit tangent at an arc length.
    ///
    /// @param s Arc length, clamped to `[0, getLength()]`
    /// @return Tangent, or an empty vector if construction failed or `s` is NaN
    ///
    Eigen::VectorXd getTangent(double s) const;

    ///
    /// Gets the curvature vector at an arc length.
    ///
    /// @param s Arc length, clamped to `[0, getLength()]`
    /// @return Curvature, or an empty vector if construction failed or `s` is NaN
    ///
    Eigen::VectorXd getCurvature(double s) const;

    ///
    /// Gets the next switching point after an arc length.
    ///
    /// @param s Arc length
    /// @param discontinuity Always set to true
    /// @return Always `getLength()`, since switching points are not reported
    ///
    double getNextSwitchingPoint(double s, bool& discontinuity) const;

    ///
    /// Gets the switching points of the path.
    ///
    /// @return Always an empty list, since switching points are not reported
    ///
    // C++03 needs the space between the closing brackets, which the formatter would remove.
    // clang-format off
    std::list<std::pair<double, bool> > getSwitchingPoints() const;
    // clang-format on

#if __cplusplus >= 201103L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
    ///
    /// Gets the reason construction failed. This extends the original interface, and needs C++11.
    ///
    /// @return The exception construction raised, or a null pointer if it succeeded
    ///
    std::exception_ptr getError() const;
#endif

   private:
    friend class Trajectory;

    struct VIAM_TRAJEX_TOTG_COMPAT_HIDDEN impl;
    impl* impl_;
};

}  // namespace compat
}  // namespace totg
}  // namespace trajex
}  // namespace viam
