// Test utility functions shared across multiple test files
#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <exception>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include <viam/trajex/totg/observers.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg::test {

/// Check if two configurations are close within tolerance
bool configs_close(const xvector<>& a, const xvector<>& b, double tolerance = 1e-6);

/// Verify path visits all waypoints within max_deviation
void verify_path_visits_waypoints(const path& p, const xmatrix<>& waypoints, double max_deviation);

/// Build a segment-type string: 'L' per linear segment, 'C' per circular, left to right.
/// Example: a 3-waypoint path with a single circular blend yields "LCL".
std::string path_type_sequence(const path& p);

/// Returns the angle between two nonzero vectors, in [0, pi], accurate everywhere including near 0 and pi.
double angle_between(const xvector<>& a, const xvector<>& b);

/// Returns how far an arc's curvature may sit from the one path construction chose for it, as a
/// fraction of that value.
double curvature_refit_bound(const path::segment::view& arc);

/// Validates the geometry of a path against the waypoints and options it was built from.
///
/// The segments and their joins are checked directly rather than by sampling the path, since an arc
/// can be far shorter than any sampling interval and a sample would never land on it. Every
/// bound is derived from the error bounds path declares and from the rounding of this function's own
/// arithmetic, so a failure means the path broke its contract rather than that a tolerance was tight.
void validate_path_quality(const path& p, const waypoint_accumulator& waypoints, const path::options& opts);

/// The quaternion w, x, y, z for a URDF fixed-axis XYZ rpy, which composes as
/// Rz(yaw) * Ry(pitch) * Rx(roll).
std::array<double, 4> rpy_to_quaternion(double roll, double pitch, double yaw);

/// The Yaskawa GP12: six revolute joints and a fixed tool, transcribed from gp12.urdf. The tool's
/// origin keeps that file's truncated rpy rather than exact multiples of pi, as the model table it
/// replaced did, so the regression tests that use it still run the arm they were written against.
kinematics::chain gp12_chain();

/// 3xN linear-velocity Jacobian of a planar 2-link arm at joint angles q=[q1,q2].
/// Rows are [dx; dy; dz]; dz is always 0 (planar). Lengths l1,l2 in metres.
xmatrix<> planar_2link_jacobian(double l1, double l2, const xvector<>& q);

/// The TCP's linear speed per unit of arc velocity for the planar 2-link, and its derivative along
/// the path, for tests that build tcp_limits by hand. The derivative is a central difference of
/// ||planar_2link_jacobian(q) * q_prime|| along the path. Empty where that speed is exactly zero,
/// as the evaluator contract asks, since the derivative does not exist there.
std::optional<trajectory::tcp_limits::linear_speed_per_arc_velocity> planar_2link_linear_speed_per_arc_velocity(
    double l1, double l2, const xvector<>& q, const xvector<>& q_prime, const xvector<>& q_double_prime);

/// Builds tcp_limits for a planar 2-link (l1 = l2 = 1).
trajectory::tcp_limits planar_2link_tcp_limits(double max_linear_speed);

/// Maximum realized Cartesian TCP speed over a finished trajectory, for a planar 2-link arm.
/// At each sampled instant the realized TCP velocity is J(q)*q_dot; this returns the peak of
/// its Euclidean norm. Used to verify a TCP-limited trajectory respects its v_TCP cap.
double max_realized_tcp_speed(const trajectory& traj, double l1, double l2, int samples = 1000);

/// Observer that validates expectations as events occur.
class expectation_observer final : public trajectory::integration_observer {
   public:
    struct expected_forward_start {
        std::optional<arc_length> s;
        std::optional<arc_velocity> s_dot;
        std::optional<double> tolerance_percent;
    };

    struct expected_hit_limit {
        std::optional<arc_length> s;
        std::optional<arc_velocity> s_dot;
        std::optional<arc_velocity> s_dot_max_acc;
        std::optional<arc_velocity> s_dot_max_vel;
        std::optional<double> tolerance_percent;
    };

    struct expected_backward_start {
        std::optional<arc_length> s;
        std::optional<arc_velocity> s_dot;
        std::optional<trajectory::switching_point_kind> kind;
        std::optional<double> tolerance_percent;
    };

    struct expected_splice {
        std::optional<trajectory::seconds> duration;
        std::optional<std::size_t> num_pruned;
        std::optional<double> tolerance_percent;
    };

    using integration_expectation = std::variant<expected_forward_start, expected_hit_limit, expected_backward_start, expected_splice>;

    explicit expectation_observer(double default_tolerance_percent = 0.1);

    expectation_observer& expect_forward_start(std::optional<arc_length> s = std::nullopt,
                                               std::optional<arc_velocity> s_dot = std::nullopt,
                                               std::optional<double> tolerance_percent = std::nullopt);

    expectation_observer& expect_hit_limit(std::optional<arc_length> s = std::nullopt,
                                           std::optional<arc_velocity> s_dot = std::nullopt,
                                           std::optional<arc_velocity> acc_limit = std::nullopt,
                                           std::optional<arc_velocity> vel_limit = std::nullopt,
                                           std::optional<double> tolerance_percent = std::nullopt);

    expectation_observer& expect_backward_start(std::optional<arc_length> s = std::nullopt,
                                                std::optional<arc_velocity> s_dot = std::nullopt,
                                                std::optional<trajectory::switching_point_kind> kind = std::nullopt,
                                                std::optional<double> tolerance_percent = std::nullopt);

    expectation_observer& expect_splice(std::optional<trajectory::seconds> duration = std::nullopt,
                                        std::optional<std::size_t> num_pruned = std::nullopt,
                                        std::optional<double> tolerance_percent = std::nullopt);

    void verify_all_expectations_met() const;

    const std::deque<integration_expectation>& get_expectations() const;

    double get_default_tolerance_percent() const;

    void on_started_forward_integration(const trajectory&, started_forward_event event) override;

    void on_hit_limit_curve(const trajectory&, limit_hit_event event) override;

    void on_started_backward_integration(const trajectory&, started_backward_event event) override;

    void on_trajectory_extended(const trajectory& traj, splice_event event) override;

    void on_failed(std::exception_ptr, std::shared_ptr<const trajectory>) noexcept override;

   private:
    double default_tolerance_percent_;
    std::deque<integration_expectation> expectations_;
};

/// Validates fundamental trajectory invariants.
void validate_trajectory_invariants(const trajectory& traj, double tolerance_percent = 0.1);

/// Validates joint-space kinematic bounds across a dense sampling of the trajectory.
///
/// Checks:
///   - Boundary conditions: zero velocity at start/end, zero acceleration at end
///   - Velocity and acceleration bounds at every sample
///   - Velocity continuity via finite differences (catches instantaneous joint-velocity jumps
///     that the arc-length phase-plane check in validate_trajectory_invariants cannot detect)
///   - Position-velocity consistency: finite difference of position should match trapezoidal
///     velocity estimate; large discrepancy indicates the two fields are mutually inconsistent
void validate_joint_kinematics(const trajectory& traj,
                               const xvector<>& max_velocity,
                               const xvector<>& max_acceleration,
                               double tolerance_percent = 0.1);

/// Test fixture for trajectory integration tests.
struct trajectory_test_fixture {
    path::options path_opts;
    trajectory::options traj_opts;
    std::shared_ptr<expectation_observer> expectation_observer_;
    composite_integration_observer composite_observer_;
    double validation_tolerance_percent = 0.1;
    size_t dof_;
    xmatrix<> waypoints_;

    // Trajectory-wide expectations (optional)
    struct expected_duration {
        trajectory::seconds duration;
        std::optional<double> tolerance_percent;
    };
    struct expected_path_length {
        arc_length length;
        std::optional<double> tolerance_percent;
    };

    std::optional<expected_duration> expected_duration_;
    std::optional<size_t> expected_integration_point_count_;
    std::optional<expected_path_length> expected_path_length_;

    // Legacy comparison (A/B testing against reference implementation)
    bool compare_with_legacy_ = false;
    std::optional<double> legacy_duration_;
    std::optional<double> legacy_path_length_;
    std::optional<double> legacy_generation_time_ms_;
    double legacy_path_tolerance_percent = 5.0;
    double legacy_duration_tolerance_percent = 5.0;

    // Event validation control
    bool allow_any_events_ = false;

    // Joint-space kinematic validation
    bool validate_joint_kinematics_ = true;

    // JSON output for visualization
    std::optional<std::string> json_output_filename_;
    bool json_on_failure_only_ = false;
    bool assert_on_create_failure_ = true;
    std::shared_ptr<trajectory_integration_event_collector> collector_;

    explicit trajectory_test_fixture(size_t dof = 6, double expectation_tolerance_percent = 0.1);

    trajectory_test_fixture& set_waypoints_deg(const xmatrix<>& waypoints_deg);

    trajectory_test_fixture& set_waypoints_rad(const xmatrix<>& waypoints_rad);

    trajectory_test_fixture& set_max_velocity(const xvector<>& max_vel);

    trajectory_test_fixture& set_max_acceleration(const xvector<>& max_acc);

    trajectory_test_fixture& set_max_blend_deviation(double max_dev);

    trajectory_test_fixture& expect_duration(trajectory::seconds duration, std::optional<double> tolerance_percent = std::nullopt);

    trajectory_test_fixture& expect_integration_point_count(size_t count);

    trajectory_test_fixture& expect_path_length(arc_length length, std::optional<double> tolerance_percent = std::nullopt);

    trajectory_test_fixture& enable_legacy_comparison();

    trajectory_test_fixture& allow_any_events();

    trajectory_test_fixture& disable_joint_kinematics_validation();

    trajectory_test_fixture& set_legacy_path_tolerance(double tolerance_percent);

    trajectory_test_fixture& set_legacy_duration_tolerance(double tolerance_percent);

    trajectory_test_fixture& enable_json_output(std::string filename);

    trajectory_test_fixture& enable_json_output_on_failure(std::string filename);

    trajectory_test_fixture& enable_event_collection();

    trajectory_test_fixture& suppress_create_failure_assertions();

    void try_legacy_comparison(const xmatrix<>& waypoints);

    void validate_against_legacy(const trajectory& traj);

    trajectory create_and_validate();
};

/// Constraint profile for parameterized trajectory testing.
struct constraint_profile {
    std::string name;
    xvector<> max_velocity;
    xvector<> max_acceleration;
};

/// Print operator for Boost.Test data framework.
std::ostream& operator<<(std::ostream& os, const constraint_profile& profile);

}  // namespace viam::trajex::totg::test

// Equality operators for trajectory event types. Placed in viam::trajex::totg so
// ADL finds them for types nested in trajectory, which lives in that namespace.
// They live in test_utils.hpp rather than the library headers to avoid committing
// to them as part of the public ABI.

namespace viam::trajex::totg {

bool operator==(const trajectory::integration_observer::started_forward_event& a,
                const trajectory::integration_observer::started_forward_event& b) noexcept;

bool operator==(const trajectory::integration_observer::limit_hit_event& a,
                const trajectory::integration_observer::limit_hit_event& b) noexcept;

bool operator==(const trajectory::integration_observer::started_backward_event& a,
                const trajectory::integration_observer::started_backward_event& b) noexcept;

bool operator==(const trajectory::integration_observer::splice_event& a, const trajectory::integration_observer::splice_event& b) noexcept;

}  // namespace viam::trajex::totg
