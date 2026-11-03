// Test utility functions shared across multiple test files
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>

#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg::test {

/// Check if two configurations are close within tolerance
bool configs_close(const xvector<>& a, const xvector<>& b, double tolerance = 1e-6);

/// Verify path visits all waypoints within max_deviation
void verify_path_visits_waypoints(const path& p, const xmatrix<>& waypoints, double max_deviation);

/// Build a segment-type string: 'L' per linear segment, 'C' per circular, left to right.
/// Example: a 3-waypoint path with a single circular blend yields "LCL".
std::string path_type_sequence(const path& p);

/// The quaternion w, x, y, z for a URDF fixed-axis XYZ rpy, which composes as
/// Rz(yaw) * Ry(pitch) * Rx(roll).
inline std::array<double, 4> rpy_to_quaternion(double roll, double pitch, double yaw) {
    const double cr = std::cos(roll / 2.0);
    const double sr = std::sin(roll / 2.0);
    const double cp = std::cos(pitch / 2.0);
    const double sp = std::sin(pitch / 2.0);
    const double cy = std::cos(yaw / 2.0);
    const double sy = std::sin(yaw / 2.0);
    return {
        (cr * cp * cy) + (sr * sp * sy), (sr * cp * cy) - (cr * sp * sy), (cr * sp * cy) + (sr * cp * sy), (cr * cp * sy) - (sr * sp * cy)};
}

/// The Yaskawa GP12: six revolute joints and a fixed tool, transcribed from gp12.urdf. The tool's
/// origin keeps that file's truncated rpy rather than exact multiples of pi, as the model table it
/// replaced did, so the regression tests that use it still run the arm they were written against.
inline kinematics::chain gp12_chain() {
    using joint_origin = kinematics::chain::joint_origin;
    using rotation_axis = kinematics::chain::rotation_axis;
    return kinematics::chain::builder{}
        .extend(joint_origin{.translation = {0, 0, 0.450}}, rotation_axis{{0, 0, 1}})
        .extend(joint_origin{.translation = {0.155, 0, 0}}, rotation_axis{{0, 1, 0}})
        .extend(joint_origin{.translation = {0, 0, 0.614}}, rotation_axis{{0, -1, 0}})
        .extend(joint_origin{.translation = {0.640, 0, 0.200}}, rotation_axis{{-1, 0, 0}})
        .extend(joint_origin{}, rotation_axis{{0, -1, 0}})
        .extend(joint_origin{}, rotation_axis{{-1, 0, 0}})
        .extend(joint_origin{.translation = {0.100, 0, 0}, .rotation = rpy_to_quaternion(3.14159265, -1.570796, 0)})
        .build();
}

/// 3xN linear-velocity Jacobian of a planar 2-link arm at joint angles q=[q1,q2].
/// Rows are [dx; dy; dz]; dz is always 0 (planar). Lengths l1,l2 in metres.
inline xmatrix<> planar_2link_jacobian(double l1, double l2, const xvector<>& q) {
    const double q1 = q(0);
    const double q2 = q(1);
    const double s1 = std::sin(q1);
    const double c1 = std::cos(q1);
    const double s12 = std::sin(q1 + q2);
    const double c12 = std::cos(q1 + q2);
    xmatrix<> J = xt::zeros<double>({std::size_t{3}, std::size_t{2}});
    J(0, 0) = (-l1 * s1) - (l2 * s12);
    J(0, 1) = -l2 * s12;
    J(1, 0) = (l1 * c1) + (l2 * c12);
    J(1, 1) = l2 * c12;
    return J;
}

/// The TCP's linear speed per unit of arc velocity for the planar 2-link, and its derivative along
/// the path, for tests that build tcp_limits by hand. The derivative is a central difference of
/// ||planar_2link_jacobian(q) * q_prime|| along the path. Empty where that speed is exactly zero,
/// as the evaluator contract asks, since the derivative does not exist there.
inline std::optional<trajectory::tcp_limits::linear_speed_per_arc_velocity> planar_2link_linear_speed_per_arc_velocity(
    double l1, double l2, const xvector<>& q, const xvector<>& q_prime, const xvector<>& q_double_prime) {
    const auto speed_ratio_at = [&](const xvector<>& qq, const xvector<>& qp) {
        const auto J = planar_2link_jacobian(l1, l2, qq);
        double vx = 0.0;
        double vy = 0.0;
        double vz = 0.0;
        for (std::size_t j = 0; j < qp.size(); ++j) {
            vx += J(0, j) * qp(j);
            vy += J(1, j) * qp(j);
            vz += J(2, j) * qp(j);
        }
        return std::sqrt((vx * vx) + (vy * vy) + (vz * vz));
    };
    const double value = speed_ratio_at(q, q_prime);
    if (value == 0.0) {
        return std::nullopt;
    }
    const double h = 1e-6;
    const double derivative =
        (speed_ratio_at(q + h * q_prime, q_prime + h * q_double_prime) - speed_ratio_at(q - h * q_prime, q_prime - h * q_double_prime)) /
        (2.0 * h);
    return trajectory::tcp_limits::linear_speed_per_arc_velocity{.value = value, .arc_length_derivative = derivative};
}

/// Builds tcp_limits for a planar 2-link (l1 = l2 = 1).
inline trajectory::tcp_limits planar_2link_tcp_limits(double max_linear_speed) {
    return trajectory::tcp_limits{
        .max_linear_speed = max_linear_speed,
        .evaluator =
            [](const xvector<>& q, const xvector<>& q_prime, const xvector<>& q_double_prime) {
                return planar_2link_linear_speed_per_arc_velocity(1.0, 1.0, q, q_prime, q_double_prime);
            },
    };
}

/// Maximum realized Cartesian TCP speed over a finished trajectory, for a planar 2-link arm.
/// At each sampled instant the realized TCP velocity is J(q)*q_dot; this returns the peak of
/// its Euclidean norm. Used to verify a TCP-limited trajectory respects its v_TCP cap.
inline double max_realized_tcp_speed(const trajectory& traj, double l1, double l2, int samples = 1000) {
    double peak = 0.0;
    // End-of-path stamping artifacts surface at the terminal instant, so the exact end of the
    // trajectory must be checked against the cap too. The sampler spaces its samples evenly over
    // [0, duration] and lands the last one exactly on the duration, never past it.
    uniform_sampler sampler{static_cast<std::size_t>(samples) + 1};
    auto cursor = traj.create_cursor();
    while (const auto smp = sampler.next(cursor)) {
        const auto J = planar_2link_jacobian(l1, l2, smp->configuration);
        double v[3] = {0.0, 0.0, 0.0};
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t c = 0; c < smp->velocity.size(); ++c) {
                v[r] += J(r, c) * smp->velocity(c);
            }
        }
        peak = std::max(peak, std::sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2])));
    }
    return peak;
}

}  // namespace viam::trajex::totg::test

// Equality operators for trajectory event types. Placed in viam::trajex::totg so
// ADL finds them for types nested in trajectory, which lives in that namespace.
// They live in test_utils.hpp rather than the library headers to avoid committing
// to them as part of the public ABI.

namespace viam::trajex::totg {

inline bool operator==(const trajectory::integration_observer::started_forward_event& a,
                       const trajectory::integration_observer::started_forward_event& b) noexcept {
    return a.start == b.start;
}

inline bool operator==(const trajectory::integration_observer::limit_hit_event& a,
                       const trajectory::integration_observer::limit_hit_event& b) noexcept {
    return a.breach == b.breach && a.s_dot_max_acc == b.s_dot_max_acc && a.s_dot_max_vel == b.s_dot_max_vel;
}

inline bool operator==(const trajectory::integration_observer::started_backward_event& a,
                       const trajectory::integration_observer::started_backward_event& b) noexcept {
    return a.start == b.start && a.kind == b.kind;
}

inline bool operator==(const trajectory::integration_observer::splice_event& a,
                       const trajectory::integration_observer::splice_event& b) noexcept {
    // We need a custom comparator here, because for splice events, we can have s_ddot = NaN for the last element.
    return std::ranges::equal(a.pruned, b.pruned, [](const auto& lhs, const auto& rhs) {
        if (lhs.time != rhs.time) {
            return false;
        }
        if (lhs.s != rhs.s) {
            return false;
        }
        if (lhs.s_dot != rhs.s_dot) {
            return false;
        }
        if (!std::isnan(static_cast<double>(lhs.s_ddot)) && !std::isnan(static_cast<double>(lhs.s_ddot)) && (lhs.s_ddot != rhs.s_ddot)) {
            return false;
        }
        return true;
    });
}

}  // namespace viam::trajex::totg
