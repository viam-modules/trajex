// Test utility functions shared across multiple test files
#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/totg/trajectory.hpp>

namespace viam::trajex::totg::test {

/// Check if two configurations are close within tolerance
bool configs_close(const xt::xarray<double>& a, const xt::xarray<double>& b, double tolerance = 1e-6);

/// Verify path visits all waypoints within max_deviation
void verify_path_visits_waypoints(const path& p, const xt::xarray<double>& waypoints, double max_deviation);

/// Build a segment-type string: 'L' per linear segment, 'C' per circular, left to right.
/// Example: a 3-waypoint path with a single circular blend yields "LCL".
std::string path_type_sequence(const path& p);

/// Yaskawa GP12 model table (viam::sdk::ModelTable tensor format): six revolute joints
/// followed by a fixed tool row, transcribed from gp12.urdf.
inline xt::xarray<double> gp12_model_table() {
    return xt::xarray<double>{
        {0, 0, 0.450, 0, 0, 0, 0, 0, 1, 0},
        {0.155, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0.614, 0, 0, 0, 0, -1, 0, 0},
        {0.640, 0, 0.200, 0, 0, 0, -1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, -1, 0, 0},
        {0, 0, 0, 0, 0, 0, -1, 0, 0, 0},
        {0.100, 0, 0, 3.14159265, -1.570796, 0, 0, 0, 0, 3},
    };
}

/// 3xN linear-velocity Jacobian of a planar 2-link arm at joint angles q=[q1,q2].
/// Rows are [dx; dy; dz]; dz is always 0 (planar). Lengths l1,l2 in metres.
inline xt::xarray<double> planar_2link_jacobian(double l1, double l2, const xt::xarray<double>& q) {
    const double q1 = q(0);
    const double q2 = q(1);
    const double s1 = std::sin(q1);
    const double c1 = std::cos(q1);
    const double s12 = std::sin(q1 + q2);
    const double c12 = std::cos(q1 + q2);
    xt::xarray<double> J = xt::zeros<double>({std::size_t{3}, std::size_t{2}});
    J(0, 0) = -l1 * s1 - l2 * s12;
    J(0, 1) = -l2 * s12;
    J(1, 0) = l1 * c1 + l2 * c12;
    J(1, 1) = l2 * c12;
    return J;
}

/// Numerical linear velocity gain for the planar 2-link, for tests that build tcp_limits by hand.
/// Central-differences ||planar_2link_jacobian(q) * q_prime|| along the path.
inline jacobian::kinematic_chain::linear_velocity_gain planar_2link_linear_velocity_gain(
    double l1, double l2, const xt::xarray<double>& q, const xt::xarray<double>& q_prime, const xt::xarray<double>& q_double_prime) {
    const auto gain_at = [&](const xt::xarray<double>& qq, const xt::xarray<double>& qp) {
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
    const double g = gain_at(q, q_prime);
    const double h = 1e-6;
    const double dg =
        (gain_at(q + h * q_prime, q_prime + h * q_double_prime) - gain_at(q - h * q_prime, q_prime - h * q_double_prime)) / (2.0 * h);
    return {.gain_per_arc_unit = g, .d_gain_ds = dg};
}

/// Builds tcp_limits for a planar 2-link (l1 = l2 = 1) with both callbacks set.
inline trajectory::tcp_limits planar_2link_tcp_limits(double max_linear_velocity) {
    return trajectory::tcp_limits{
        .max_linear_velocity = max_linear_velocity,
        .linear_jacobian = [](const xt::xarray<double>& q) { return planar_2link_jacobian(1.0, 1.0, q); },
        .linear_velocity_gain =
            [](const xt::xarray<double>& q, const xt::xarray<double>& q_prime, const xt::xarray<double>& q_double_prime) {
                return planar_2link_linear_velocity_gain(1.0, 1.0, q, q_prime, q_double_prime);
            },
    };
}

/// Maximum realized Cartesian TCP speed over a finished trajectory, for a planar 2-link arm.
/// At each sampled instant the realized TCP velocity is J(q)*q_dot; this returns the peak of
/// its Euclidean norm. Used to verify a TCP-limited trajectory respects its v_TCP cap.
inline double max_realized_tcp_speed(const trajectory& traj, double l1, double l2, int samples = 1000) {
    const double duration = traj.duration().count();
    double peak = 0.0;
    // Sample over [0, duration] inclusive: end-of-path stamping artifacts surface at the terminal
    // instant, so the exact end of the trajectory must be checked against the cap too. The
    // clamp matters: the duration * i / samples round trip can land one ulp above the duration on
    // the last iteration (duration-value dependent, observed on arm64), and sampling past the
    // duration throws.
    for (int i = 0; i <= samples; ++i) {
        const double t = std::min(duration, duration * static_cast<double>(i) / static_cast<double>(samples));
        const auto smp = traj.sample(trajectory::seconds{t});
        const auto J = planar_2link_jacobian(l1, l2, smp.configuration);
        double v[3] = {0.0, 0.0, 0.0};
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t c = 0; c < smp.velocity.size(); ++c) {
                v[r] += J(r, c) * smp.velocity(c);
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
