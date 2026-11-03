#pragma once

#include <Eigen/Core>

#include <viam/trajex/totg/compat/export.hpp>
#include <viam/trajex/totg/compat/path.hpp>

namespace viam {
namespace trajex {
namespace totg {
namespace compat {

///
/// The `Trajectory` interface of the Kunz & Stilman reference implementation, backed by trajex.
///
/// Code written against the original `Trajectory` compiles against this one unchanged. The
/// implementation is trajex's, so results won't match the original's exactly, and the interface
/// behaves differently in these ways:
///
/// - Construction never throws. Failure is reported through `isValid`, as in the original, and
///   `getError` reports why. An invalid trajectory has a duration of zero and returns empty
///   vectors.
/// - Time arguments are clamped to `[0, getDuration()]`. The original extrapolates past the end
///   and reads out of bounds before the start.
/// - `outputPhasePlaneTrajectory` does nothing.
///
class VIAM_TRAJEX_TOTG_COMPAT_API Trajectory {
   public:
    ///
    /// Generates a time-optimal trajectory along a path.
    ///
    /// @param path Path to follow. The trajectory keeps its own copy, so the path need not outlive it.
    /// @param maxVelocity Maximum velocity per dimension
    /// @param maxAcceleration Maximum acceleration per dimension
    /// @param timeStep Integration time step
    ///
    Trajectory(const Path& path, const Eigen::VectorXd& maxVelocity, const Eigen::VectorXd& maxAcceleration, double timeStep = 0.001);

    ///
    /// Copy constructs a trajectory.
    ///
    Trajectory(const Trajectory& trajectory);

    ///
    /// Copy assigns a trajectory.
    ///
    Trajectory& operator=(const Trajectory& trajectory);

    ///
    /// Destroys a trajectory.
    ///
    ~Trajectory();

    ///
    /// Reports whether trajectory generation succeeded.
    ///
    /// @return true if the trajectory is usable
    ///
    bool isValid() const;

    ///
    /// Gets the duration of the trajectory.
    ///
    /// @return Duration, or zero if the trajectory is invalid
    ///
    double getDuration() const;

    ///
    /// Gets the configuration at a time.
    ///
    /// @param time Time, clamped to `[0, getDuration()]`
    /// @return Configuration, or an empty vector if the trajectory is invalid or `time` is NaN
    ///
    Eigen::VectorXd getPosition(double time) const;

    ///
    /// Gets the velocity at a time.
    ///
    /// @param time Time, clamped to `[0, getDuration()]`
    /// @return Velocity, or an empty vector if the trajectory is invalid or `time` is NaN
    ///
    Eigen::VectorXd getVelocity(double time) const;

    ///
    /// Does nothing. The original writes phase plane debugging files.
    ///
    void outputPhasePlaneTrajectory() const;

#if __cplusplus >= 201103L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
    ///
    /// Gets the reason trajectory generation failed. This extends the original interface, and
    /// needs C++11.
    ///
    /// @return The exception generation raised, the path's error if the path was invalid, or a
    ///         null pointer if generation succeeded
    ///
    std::exception_ptr getError() const;
#endif

   private:
    struct VIAM_TRAJEX_TOTG_COMPAT_HIDDEN impl;
    impl* impl_;
};

}  // namespace compat
}  // namespace totg
}  // namespace trajex
}  // namespace viam
