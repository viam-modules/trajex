// Test utility function implementations
#include <viam/trajex/totg/test/test_utils.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/tools/json_serialization.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/angles.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/arc_operations.hpp>
#include <viam/trajex/types/hertz.hpp>

#if __has_include(<xtensor/reducers/xnorm.hpp>)
#include <xtensor/reducers/xnorm.hpp>
#else
#include <xtensor/xnorm.hpp>
#endif

#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
#include <list>

#include <Eigen/Core>

#include <third_party/trajectories/Path.h>
#include <third_party/trajectories/Trajectory.h>
#endif

namespace viam::trajex::totg::test {

bool configs_close(const xvector<>& a, const xvector<>& b, double tolerance) {
    if (a.shape(0) != b.shape(0)) {
        return false;
    }
    for (size_t i = 0; i < a.shape(0); ++i) {
        if (std::abs(a(i) - b(i)) > tolerance) {
            return false;
        }
    }
    return true;
}

void verify_path_visits_waypoints(const path& p, const xmatrix<>& waypoints, double max_deviation) {
    using namespace viam::trajex;

    for (size_t i = 0; i < waypoints.shape(0); ++i) {
        const xvector<> waypoint = xt::view(waypoints, i, xt::all());

        // Find minimum distance from waypoint to any point on path
        double min_distance = std::numeric_limits<double>::max();
        // Use dense sampling: 1000 samples to ensure we don't miss waypoints
        const size_t num_samples = 1000;

        for (size_t j = 0; j <= num_samples; ++j) {
            const double fraction = static_cast<double>(j) / static_cast<double>(num_samples);
            const arc_length s{static_cast<double>(p.length()) * fraction};
            const auto config = p.configuration(s);

            // Compute distance
            const auto diff = waypoint - config;
            const double distance = xt::norm_l2(diff)();
            min_distance = std::min(min_distance, distance);
        }

        // Waypoint should be within max_deviation of some point on path
        // Allow small numerical tolerance for floating point and sampling error
        BOOST_CHECK_LE(min_distance, max_deviation + 1e-4);
    }
}

std::string path_type_sequence(const path& p) {
    std::string result;
    result.reserve(p.size());
    for (const auto& seg : p) {
        result += seg.is<path::segment::linear>() ? 'L' : 'C';
    }
    return result;
}

double angle_between(const xvector<>& a, const xvector<>& b) {
    // For unit vectors u and v at angle theta, |v - u| = 2 sin(theta / 2) and |v + u| = 2 cos(theta / 2),
    // so theta is twice the arctangent of their ratio. Unlike the arccosine of u . v, which loses its
    // digits where the cosine is flat near 0 and pi, both lengths are well determined there. Normalizing
    // first means the vectors need not be unit already.
    const xvector<> u = a / xt::norm_l2(a)();
    const xvector<> v = b / xt::norm_l2(b)();
    return 2.0 * std::atan2(xt::norm_l2(v - u)(), xt::norm_l2(v + u)());
}

double curvature_refit_bound(const path::segment::view& arc) {
    // Path construction refits an arc's radius to the span the arc is given on the path, which is
    // radius * angle rounded into the path's arc length. Rounding the arc's end moves that span by at
    // most an ULP of the end, and so moves the radius, and the curvature with it, by at most that much
    // relative to the arc's length. The division that refits the radius rounds once more.
    const auto end = static_cast<double>(arc.end());
    const auto end_ulp = std::nextafter(end, std::numeric_limits<double>::infinity()) - end;
    return (end_ulp / static_cast<double>(arc.length())) + std::numeric_limits<double>::epsilon();
}

void validate_path_quality(const path& p, const waypoint_accumulator& waypoints, const path::options& opts) {
    constexpr auto eps = std::numeric_limits<double>::epsilon();
    const auto dof = static_cast<double>(p.dof());

    const auto magnitude = [](const xvector<>& q) { return xt::amax(xt::abs(q))(); };
    const auto ulp_of = [](arc_length s) {
        const auto value = static_cast<double>(s);
        return std::nextafter(value, std::numeric_limits<double>::infinity()) - value;
    };

    const std::vector<path::segment::view> segments(p.begin(), p.end());

    // This is the size of the coordinates where each segment's positions are computed. A position on a
    // segment sums its start or apex with an offset. Neither term is larger than the segment's ends
    // plus its length, so that is the scale. We scale per segment rather than by the whole path so
    // that one far waypoint doesn't loosen the bound everywhere else.
    std::vector<double> scales;
    scales.reserve(segments.size());
    for (const auto& segment : segments) {
        scales.push_back(std::max(magnitude(segment.configuration(segment.start())), magnitude(segment.configuration(segment.end()))) +
                         static_cast<double>(segment.length()));
    }

    BOOST_TEST_CONTEXT("Validating path quality") {
        BOOST_REQUIRE(!segments.empty());

        // The exact path starts and ends on the first and last waypoints.
        BOOST_TEST_CONTEXT("Endpoints") {
            BOOST_CHECK(configs_close(
                p.configuration(arc_length{0.0}), xvector<>{waypoints[0]}, path::k_position_error_bound_ulps * eps * scales.front()));
            BOOST_CHECK(configs_close(p.configuration(p.length()),
                                      xvector<>{waypoints[waypoints.size() - 1]},
                                      (path::k_position_error_bound_ulps * eps * scales.back()) + ulp_of(p.length())));
        }

        for (std::size_t k = 0; k < segments.size(); ++k) {
            const auto& segment = segments[k];
            BOOST_TEST_CONTEXT("Segment " << k << " at s = " << segment.start()) {
                BOOST_CHECK(segment.has_interior());

                // A unit tangent within k_tangent_error_bound_ulps of one has a norm within that many
                // ULPs of 1. Taking the norm sums dof squares, which rounds by at most (dof / 2 + 1) eps
                // relative.
                const auto tangent_norm_bound = (path::k_tangent_error_bound_ulps + (dof / 2.0) + 1.0) * eps;
                BOOST_CHECK_SMALL(xt::norm_l2(segment.tangent(segment.start()))() - 1.0, tangent_norm_bound);
                BOOST_CHECK_SMALL(xt::norm_l2(segment.tangent(segment.end()))() - 1.0, tangent_norm_bound);

                if (segment.is<path::segment::circular>()) {
                    // An arc traversed at unit speed turns its tangent by its length times its curvature,
                    // which is less than pi. The angle between two directions moves by no more than
                    // either direction does, so each end's tangent contributes at most
                    // k_tangent_error_bound_ulps. Computing the angle normalizes two vectors and takes
                    // the norms of their difference and sum, which rounds by under (dof + 4) eps in all.
                    // The curvature's relative error and the rounding of its product with the length
                    // both scale with the angle, which is at most pi.
                    const auto turning_bound = ((2.0 * path::k_tangent_error_bound_ulps) + dof + 4.0 +
                                                (std::numbers::pi * (path::k_curvature_error_bound_ulps + 1.0))) *
                                               eps;
                    const auto curvature = xt::norm_l2(segment.curvature(segment.start()))();
                    const auto turned = angle_between(segment.tangent(segment.start()), segment.tangent(segment.end()));
                    BOOST_CHECK_SMALL(turned - (static_cast<double>(segment.length()) * curvature), turning_bound);

                    // The exact curvature lies within the limits path construction was given, up to the refit
                    // of the radius to the arc's span.
                    const auto curvature_bound = curvature_refit_bound(segment) + (path::k_curvature_error_bound_ulps * eps);
                    BOOST_CHECK_GE(curvature, opts.min_blend_curvature() * (1.0 - curvature_bound));
                    BOOST_CHECK_LE(curvature, opts.max_blend_curvature() * (1.0 + curvature_bound));
                }
            }
        }

        // Neighbours meet in position, and in tangent wherever an arc is involved. A join of two lines
        // is a corner left unblended, which turns by design. Each side of a join is within
        // k_position_error_bound_ulps of the same exact point, scaled by the larger of the two
        // segments' coordinates, plus an ULP of arc length for addressing the end of the earlier
        // segment. Refitting an arc's radius moves its ends by at most the rounding of its span,
        // another ULP of arc length. Each side's tangent, a line's as much as an arc's, is within
        // k_tangent_error_bound_ulps of the exact tangent they share.
        for (std::size_t k = 1; k < segments.size(); ++k) {
            const auto& before = segments[k - 1];
            const auto& after = segments[k];
            BOOST_TEST_CONTEXT("Join " << k << " at s = " << after.start()) {
                const auto scale = std::max(scales[k - 1], scales[k]);
                const auto position_bound = (2.0 * path::k_position_error_bound_ulps * eps * scale) + (2.0 * ulp_of(after.start()));
                BOOST_CHECK(configs_close(before.configuration(before.end()), after.configuration(after.start()), position_bound));

                if (!(before.is<path::segment::linear>() && after.is<path::segment::linear>())) {
                    const auto tangent_bound = 2.0 * path::k_tangent_error_bound_ulps * eps;
                    BOOST_CHECK(configs_close(before.tangent(before.end()), after.tangent(after.start()), tangent_bound));
                }
            }
        }

        // The path visits every waypoint, in order, within the deviation budget. We walk the segments
        // forward from where the previous waypoint was visited and take the first point within budget
        // of this one. We resume on the previous waypoint's segment, since colinearized waypoints share
        // one, but never behind the previous visit, because on a path that reverses the segment we
        // resume on may have crossed this waypoint's position before it reached the previous one. The
        // candidate on each segment is its nearest point, held back to the previous visit.
        //
        // The exact path passes within the larger deviation budget of each waypoint. A computed
        // position on it is within k_position_error_bound_ulps of that, plus an ULP of arc length for
        // addressing it and another for the refit of an arc.
        const auto deviation_budget = std::max(opts.max_blend_deviation(), opts.max_linear_deviation());
        std::size_t match = 0;
        auto previous_visit = arc_length{0.0};
        for (std::size_t i = 0; i < waypoints.size(); ++i) {
            const xvector<> waypoint{waypoints[i]};
            std::optional<arc_length> visit;
            for (; match < segments.size(); ++match) {
                const auto& segment = segments[match];
                const auto deviation_bound =
                    deviation_budget + (path::k_position_error_bound_ulps * eps * scales[match]) + (2.0 * ulp_of(segment.end()));
                const auto nearest = std::max(segment.nearest(std::span<const double>{waypoint.data(), waypoint.size()}), previous_visit);
                if (nearest <= segment.end() && xt::norm_l2(segment.configuration(nearest) - waypoint)() <= deviation_bound) {
                    visit = nearest;
                    break;
                }
            }
            if (!visit) {
                BOOST_ERROR("waypoint " << i << " is not within " << deviation_budget << " of the path after waypoint " << (i - 1));
                break;
            }
            previous_visit = *visit;
        }
    }
}

std::array<double, 4> rpy_to_quaternion(double roll, double pitch, double yaw) {
    const double cr = std::cos(roll / 2.0);
    const double sr = std::sin(roll / 2.0);
    const double cp = std::cos(pitch / 2.0);
    const double sp = std::sin(pitch / 2.0);
    const double cy = std::cos(yaw / 2.0);
    const double sy = std::sin(yaw / 2.0);
    return {
        (cr * cp * cy) + (sr * sp * sy), (sr * cp * cy) - (cr * sp * sy), (cr * sp * cy) + (sr * cp * sy), (cr * cp * sy) - (sr * sp * cy)};
}

kinematics::chain gp12_chain() {
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

xmatrix<> planar_2link_jacobian(double l1, double l2, const xvector<>& q) {
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

std::optional<trajectory::tcp_limits::linear_speed_per_arc_velocity> planar_2link_linear_speed_per_arc_velocity(
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

trajectory::tcp_limits planar_2link_tcp_limits(double max_linear_speed) {
    return trajectory::tcp_limits{
        .max_linear_speed = max_linear_speed,
        .evaluator =
            [](const xvector<>& q, const xvector<>& q_prime, const xvector<>& q_double_prime) {
                return planar_2link_linear_speed_per_arc_velocity(1.0, 1.0, q, q_prime, q_double_prime);
            },
    };
}

double max_realized_tcp_speed(const trajectory& traj, double l1, double l2, int samples) {
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

namespace {

constexpr bool k_log_met_expectations = false;

}  // namespace

expectation_observer::expectation_observer(double default_tolerance_percent) : default_tolerance_percent_(default_tolerance_percent) {}

expectation_observer& expectation_observer::expect_forward_start(std::optional<arc_length> s,
                                                                 std::optional<arc_velocity> s_dot,
                                                                 std::optional<double> tolerance_percent) {
    expectations_.push_back(expected_forward_start{s, s_dot, tolerance_percent});
    return *this;
}

expectation_observer& expectation_observer::expect_hit_limit(std::optional<arc_length> s,
                                                             std::optional<arc_velocity> s_dot,
                                                             std::optional<arc_velocity> acc_limit,
                                                             std::optional<arc_velocity> vel_limit,
                                                             std::optional<double> tolerance_percent) {
    expectations_.push_back(expected_hit_limit{s, s_dot, acc_limit, vel_limit, tolerance_percent});
    return *this;
}

expectation_observer& expectation_observer::expect_backward_start(std::optional<arc_length> s,
                                                                  std::optional<arc_velocity> s_dot,
                                                                  std::optional<trajectory::switching_point_kind> kind,
                                                                  std::optional<double> tolerance_percent) {
    expectations_.push_back(expected_backward_start{s, s_dot, kind, tolerance_percent});
    return *this;
}

expectation_observer& expectation_observer::expect_splice(std::optional<trajectory::seconds> duration,
                                                          std::optional<std::size_t> num_pruned,
                                                          std::optional<double> tolerance_percent) {
    expectations_.push_back(expected_splice{duration, num_pruned, tolerance_percent});
    return *this;
}

void expectation_observer::verify_all_expectations_met() const {
    BOOST_TEST_CONTEXT("Checking all expectations were met") {
        BOOST_CHECK_MESSAGE(expectations_.empty(), "Not all expected events occurred. Remaining: " << expectations_.size());
    }
}

const std::deque<expectation_observer::integration_expectation>& expectation_observer::get_expectations() const {
    return expectations_;
}

double expectation_observer::get_default_tolerance_percent() const {
    return default_tolerance_percent_;
}

void expectation_observer::on_failed(std::exception_ptr, std::shared_ptr<const trajectory>) noexcept {}

void expectation_observer::on_started_forward_integration(const trajectory&, started_forward_event event) {
    BOOST_TEST_CONTEXT("on_started_forward_integration event") {
        BOOST_REQUIRE_MESSAGE(!expectations_.empty(),
                              "Unexpected forward_start event at s=" << event.start.s << " s_dot=" << event.start.s_dot);

        const auto* expected = std::get_if<expected_forward_start>(&expectations_.front());
        BOOST_REQUIRE_MESSAGE(expected != nullptr, "Expected different event type, got forward_start at s=" << event.start.s);

        const double tol = expected->tolerance_percent.value_or(default_tolerance_percent_);

        if (expected->s.has_value()) {
            BOOST_CHECK_CLOSE(static_cast<double>(event.start.s), static_cast<double>(*expected->s), tol);
        }
        if (expected->s_dot.has_value()) {
            BOOST_CHECK_CLOSE(static_cast<double>(event.start.s_dot), static_cast<double>(*expected->s_dot), tol);
        }

        if constexpr (k_log_met_expectations) {
            std::cout << "Met expectation: forward_start at s=" << event.start.s << " s_dot=" << event.start.s_dot << "\n";
        }

        expectations_.pop_front();
    }
}

void expectation_observer::on_hit_limit_curve(const trajectory&, limit_hit_event event) {
    BOOST_TEST_CONTEXT("on_hit_limit_curve event") {
        BOOST_REQUIRE_MESSAGE(!expectations_.empty(),
                              "Unexpected hit_limit event at s=" << event.breach.s << " s_dot=" << event.breach.s_dot);

        const auto* expected = std::get_if<expected_hit_limit>(&expectations_.front());
        BOOST_REQUIRE_MESSAGE(expected != nullptr, "Expected different event type, got hit_limit at s=" << event.breach.s);

        const double tol = expected->tolerance_percent.value_or(default_tolerance_percent_);

        if (expected->s.has_value()) {
            BOOST_CHECK_CLOSE(static_cast<double>(event.breach.s), static_cast<double>(*expected->s), tol);
        }
        if (expected->s_dot.has_value()) {
            BOOST_CHECK_CLOSE(static_cast<double>(event.breach.s_dot), static_cast<double>(*expected->s_dot), tol);
        }

        if (expected->s_dot_max_acc.has_value()) {
            const double actual = static_cast<double>(event.s_dot_max_acc);
            const double expected_val = static_cast<double>(*expected->s_dot_max_acc);
            if (std::isinf(expected_val)) {
                BOOST_CHECK_EQUAL(actual, expected_val);
            } else {
                BOOST_CHECK_CLOSE(actual, expected_val, tol);
            }
        }

        if (expected->s_dot_max_vel.has_value()) {
            BOOST_CHECK_CLOSE(static_cast<double>(event.s_dot_max_vel), static_cast<double>(*expected->s_dot_max_vel), tol);
        }

        if constexpr (k_log_met_expectations) {
            std::cout << "Met expectation: hit_limit at s=" << event.breach.s << " s_dot=" << event.breach.s_dot
                      << " acc_limit=" << event.s_dot_max_acc << " vel_limit=" << event.s_dot_max_vel << "\n";
        }

        expectations_.pop_front();
    }
}

void expectation_observer::on_started_backward_integration(const trajectory&, started_backward_event event) {
    BOOST_TEST_CONTEXT("on_started_backward_integration event") {
        BOOST_REQUIRE_MESSAGE(!expectations_.empty(),
                              "Unexpected backward_start event at s=" << event.start.s << " s_dot=" << event.start.s_dot);

        const auto* expected = std::get_if<expected_backward_start>(&expectations_.front());
        BOOST_REQUIRE_MESSAGE(expected != nullptr, "Expected different event type, got backward_start at s=" << event.start.s);

        const double tol = expected->tolerance_percent.value_or(default_tolerance_percent_);

        if (expected->s.has_value()) {
            BOOST_CHECK_CLOSE(static_cast<double>(event.start.s), static_cast<double>(*expected->s), tol);
        }
        if (expected->s_dot.has_value()) {
            BOOST_CHECK_CLOSE(static_cast<double>(event.start.s_dot), static_cast<double>(*expected->s_dot), tol);
        }

        // Check switching point kind if expectation includes it
        if (expected->kind.has_value()) {
            BOOST_CHECK_EQUAL(static_cast<int>(event.kind), static_cast<int>(*expected->kind));
        }

        if constexpr (k_log_met_expectations) {
            std::cout << "Met expectation: backward_start at s=" << event.start.s << " s_dot=" << event.start.s_dot
                      << " kind=" << static_cast<int>(event.kind) << "\n";
        }

        expectations_.pop_front();
    }
}

void expectation_observer::on_trajectory_extended(const trajectory& traj, splice_event event) {
    BOOST_TEST_CONTEXT("on_trajectory_extended event") {
        BOOST_REQUIRE_MESSAGE(!expectations_.empty(), "Unexpected splice event at duration=" << traj.duration().count());

        const auto* expected = std::get_if<expected_splice>(&expectations_.front());
        BOOST_REQUIRE_MESSAGE(expected != nullptr, "Expected different event type, got splice at duration=" << traj.duration().count());

        const double tol = expected->tolerance_percent.value_or(default_tolerance_percent_);

        if (expected->duration.has_value()) {
            BOOST_CHECK_CLOSE(traj.duration().count(), expected->duration->count(), tol);
        }

        if (expected->num_pruned.has_value()) {
            BOOST_CHECK_EQUAL(event.pruned.size(), *expected->num_pruned);
        }

        if constexpr (k_log_met_expectations) {
            std::cout << "Met expectation: splice at duration=" << traj.duration().count() << " pruned=" << event.pruned.size() << "\n";
        }

        expectations_.pop_front();
    }
}

void validate_trajectory_invariants(const trajectory& traj, double tolerance_percent) {
    const auto& points = traj.get_integration_points();
    const auto& p = traj.path();

    BOOST_TEST_CONTEXT("Validating trajectory invariants") {
        // Boundary conditions
        BOOST_TEST_CONTEXT("Boundary conditions") {
            BOOST_REQUIRE(!points.empty());

            const auto& first = points.front();
            BOOST_CHECK_EQUAL(static_cast<double>(first.s), 0.0);
            BOOST_CHECK_EQUAL(static_cast<double>(first.s_dot), 0.0);
            BOOST_CHECK_EQUAL(static_cast<double>(first.time.count()), 0.0);

            const auto& last = points.back();
            BOOST_CHECK_EQUAL(static_cast<double>(last.s), static_cast<double>(p.length()));
            BOOST_CHECK_EQUAL(static_cast<double>(last.s_dot), 0.0);
            BOOST_CHECK_EQUAL(static_cast<double>(last.s_ddot), 0.0);
        }

        // Boundedness
        BOOST_TEST_CONTEXT("Boundedness") {
            for (size_t i = 0; i < points.size(); ++i) {
                BOOST_CHECK(std::isfinite(static_cast<double>(points[i].s)));
                BOOST_CHECK(std::isfinite(static_cast<double>(points[i].s_dot)));
                BOOST_CHECK(std::isfinite(static_cast<double>(points[i].s_ddot)));
                BOOST_CHECK(std::isfinite(points[i].time.count()));
                BOOST_CHECK_GE(static_cast<double>(points[i].s), 0.0);
                BOOST_CHECK_LE(static_cast<double>(points[i].s), static_cast<double>(p.length()));
                BOOST_CHECK_GE(static_cast<double>(points[i].s_dot), 0.0);
            }
        }

        // Monotonicity
        BOOST_TEST_CONTEXT("Monotonicity") {
            for (size_t i = 1; i < points.size(); ++i) {
                BOOST_CHECK_LT(points[i - 1].time.count(), points[i].time.count());
                BOOST_CHECK_LT(static_cast<double>(points[i - 1].s), static_cast<double>(points[i].s));
            }
        }

        // Phase plane constraints (velocity limits and acceleration bounds)
        BOOST_TEST_CONTEXT("Phase plane constraints") {
            auto cursor = p.create_cursor(arc_length{0.0});

            for (size_t i = 0; i < points.size(); ++i) {
                const auto& pt = points[i];
                cursor.seek(pt.s);

                // Validate velocity limits
                const auto limits = traj.get_velocity_limits(cursor);
                const double s_dot_max_vel = static_cast<double>(limits.s_dot_max_vel);
                const double s_dot_max_acc = static_cast<double>(limits.s_dot_max_acc);
                const double s_dot_limit = std::min(s_dot_max_vel, s_dot_max_acc);
                const double actual_s_dot = static_cast<double>(pt.s_dot);

                // Validate acceleration bounds
                const auto bounds = traj.get_acceleration_bounds(cursor, pt.s_dot);
                const double s_ddot_min = static_cast<double>(bounds.s_ddot_min);
                const double s_ddot_max = static_cast<double>(bounds.s_ddot_max);
                const double actual_s_ddot = static_cast<double>(pt.s_ddot);

                BOOST_TEST_CONTEXT("Integration point " << i << " at s=" << static_cast<double>(pt.s)) {
                    BOOST_CHECK(std::isfinite(actual_s_dot));
                    BOOST_CHECK(std::isfinite(actual_s_ddot));
                    BOOST_CHECK_GE(actual_s_dot, 0.0);

                    // Check velocity limits
                    const double allowed_vel_limit = s_dot_limit * (1.0 + tolerance_percent / 100.0);
                    BOOST_CHECK_LE(actual_s_dot, allowed_vel_limit);

                    if (actual_s_dot > allowed_vel_limit) {
                        BOOST_TEST_MESSAGE("s_dot=" << actual_s_dot << " exceeds limit=" << s_dot_limit << " (vel_limit=" << s_dot_max_vel
                                                    << ", acc_limit=" << s_dot_max_acc << ")");
                    }

                    // Check acceleration bounds
                    const double tolerance_margin = std::max(std::abs(s_ddot_max), std::abs(s_ddot_min)) * (tolerance_percent / 100.0);
                    const double allowed_min = s_ddot_min - tolerance_margin;
                    const double allowed_max = s_ddot_max + tolerance_margin;

                    BOOST_CHECK_GE(actual_s_ddot, allowed_min);
                    BOOST_CHECK_LE(actual_s_ddot, allowed_max);

                    if (actual_s_ddot < allowed_min || actual_s_ddot > allowed_max) {
                        BOOST_TEST_MESSAGE("s_ddot=" << actual_s_ddot << " outside bounds=[" << s_ddot_min << ", " << s_ddot_max << "]");
                    }
                }
            }
        }

        // Integration point kinematic consistency
        //
        // Validates that consecutive integration points form a consistent kinematic sequence.
        // Starting from point i with (s, s_dot, s_ddot), constant-acceleration integration
        // should approximately reach point i+1's (s, s_dot). This checks that the stored
        // acceleration values and time steps produce the claimed trajectory.
        BOOST_TEST_CONTEXT("Integration point kinematic consistency") {
            for (size_t i = 0; i + 1 < points.size(); ++i) {
                const auto& curr = points[i];
                const auto& next = points[i + 1];

                const auto dt = next.time - curr.time;

                // Constant-acceleration kinematics: integrate from current point
                const auto predicted_s_dot = curr.s_dot + (curr.s_ddot * dt);
                const auto predicted_s = curr.s + (curr.s_dot * dt) + (0.5 * (curr.s_ddot * dt) * dt);

                BOOST_TEST_CONTEXT("Point " << i << " -> " << (i + 1) << " (dt=" << dt.count() << "s, s=" << static_cast<double>(curr.s)
                                            << " -> " << static_cast<double>(next.s) << ")") {
                    // Position check (always use relative error)
                    BOOST_CHECK_CLOSE(static_cast<double>(predicted_s), static_cast<double>(next.s), tolerance_percent);

                    // Velocity check: use absolute error for near-zero, relative error otherwise
                    const double next_s_dot_val = static_cast<double>(next.s_dot);
                    const double predicted_s_dot_val = static_cast<double>(predicted_s_dot);

                    if (std::abs(next_s_dot_val) < 1e-9) {
                        // Near-zero velocity: check absolute error
                        // BOOST_CHECK_CLOSE would divide by ~zero and give meaningless result
                        const double abs_error = std::abs(predicted_s_dot_val - next_s_dot_val);
                        BOOST_TEST_CONTEXT("Near-zero velocity: checking absolute error") {
                            BOOST_CHECK_LT(abs_error, 1e-9);
                            if (abs_error >= 1e-9) {
                                BOOST_TEST_MESSAGE("Absolute error: " << abs_error << " (predicted=" << predicted_s_dot_val
                                                                      << ", actual=" << next_s_dot_val << ")");
                            }
                        }
                    } else {
                        // Normal case: check relative error
                        BOOST_CHECK_CLOSE(predicted_s_dot_val, next_s_dot_val, tolerance_percent);
                        if (std::abs((predicted_s_dot_val - next_s_dot_val) / next_s_dot_val) * 100.0 > tolerance_percent) {
                            BOOST_TEST_MESSAGE("Velocity mismatch: predicted=" << predicted_s_dot_val << ", actual=" << next_s_dot_val
                                                                               << ", curr.s_ddot=" << static_cast<double>(curr.s_ddot));
                        }
                    }
                }
            }
        }
    }
}

void validate_joint_kinematics(const trajectory& traj,
                               const xvector<>& max_velocity,
                               const xvector<>& max_acceleration,
                               double tolerance_percent) {
    const size_t dof = max_velocity.size();
    const double tol_factor = 1.0 + (tolerance_percent / 100.0);

    BOOST_TEST_CONTEXT("Joint-space kinematic validation") {
        // Boundary conditions: s_dot=0 at both endpoints forces all joint velocities to zero;
        // s_ddot=0 at the endpoint forces all joint accelerations to zero there as well.
        BOOST_TEST_CONTEXT("Boundary conditions") {
            const struct trajectory::sample first = traj.sample(trajectory::seconds{0.0});
            const struct trajectory::sample last = traj.sample(traj.duration());
            for (size_t i = 0; i < dof; ++i) {
                BOOST_TEST_CONTEXT("joint=" << i) {
                    BOOST_CHECK_EQUAL(first.velocity(i), 0.0);
                    BOOST_CHECK_EQUAL(last.velocity(i), 0.0);
                    BOOST_CHECK_EQUAL(last.acceleration(i), 0.0);
                }
            }
        }

        // Sample at 2.5x the integration frequency (above Nyquist with a safety margin) so
        // that any velocity discontinuity manifests as a finite-difference acceleration far
        // exceeding the joint limit.
        const double sample_hz = 2.5 / traj.get_options().delta.count();
        auto sampler = uniform_sampler::quantized_for_trajectory(traj, viam::trajex::types::hertz{sample_hz});

        std::optional<struct trajectory::sample> prev;
        for (const auto& s : traj.samples(sampler)) {
            BOOST_TEST_CONTEXT("t=" << s.time.count()) {
                for (size_t i = 0; i < dof; ++i) {
                    BOOST_TEST_CONTEXT("joint=" << i) {
                        BOOST_CHECK_LE(std::abs(s.velocity(i)), max_velocity(i) * tol_factor);
                        BOOST_CHECK_LE(std::abs(s.acceleration(i)), max_acceleration(i) * tol_factor);

                        if (prev.has_value()) {
                            const double dt = (s.time - prev->time).count();

                            const double delta_v = std::abs(s.velocity(i) - prev->velocity(i));
                            const double finite_diff_acc = delta_v / dt;
                            BOOST_TEST_CONTEXT("vel_fd: delta_v=" << delta_v << " dt=" << dt << " => " << finite_diff_acc
                                                                  << " limit=" << max_acceleration(i)) {
                                BOOST_CHECK_LE(finite_diff_acc, max_acceleration(i) * tol_factor);
                            }

                            // Trapezoidal velocity estimate from position; error is O(accel*dt).
                            const double pos_fd = (s.configuration(i) - prev->configuration(i)) / dt;
                            const double vel_avg = 0.5 * (s.velocity(i) + prev->velocity(i));
                            BOOST_TEST_CONTEXT("pos_fd=" << pos_fd << " vel_avg=" << vel_avg << " diff=" << std::abs(pos_fd - vel_avg)) {
                                BOOST_CHECK_LE(std::abs(pos_fd - vel_avg), max_acceleration(i) * dt * 10.0);
                            }
                        }
                    }
                }
            }
            prev = s;
        }
    }
}

trajectory_test_fixture::trajectory_test_fixture(size_t dof, double expectation_tolerance_percent)
    : expectation_observer_(std::make_shared<expectation_observer>(expectation_tolerance_percent)), dof_(dof) {}

trajectory_test_fixture& trajectory_test_fixture::set_waypoints_deg(const xmatrix<>& waypoints_deg) {
    waypoints_ = degrees_to_radians(waypoints_deg);
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::set_waypoints_rad(const xmatrix<>& waypoints_rad) {
    waypoints_ = waypoints_rad;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::set_max_velocity(const xvector<>& max_vel) {
    traj_opts.max_velocity = max_vel;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::set_max_acceleration(const xvector<>& max_acc) {
    traj_opts.max_acceleration = max_acc;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::set_max_blend_deviation(double max_dev) {
    path_opts.set_max_blend_deviation(max_dev);
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::expect_duration(trajectory::seconds duration, std::optional<double> tolerance_percent) {
    expected_duration_ = expected_duration{duration, tolerance_percent};
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::expect_integration_point_count(size_t count) {
    expected_integration_point_count_ = count;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::expect_path_length(arc_length length, std::optional<double> tolerance_percent) {
    expected_path_length_ = expected_path_length{length, tolerance_percent};
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::enable_legacy_comparison() {
    compare_with_legacy_ = true;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::allow_any_events() {
    allow_any_events_ = true;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::disable_joint_kinematics_validation() {
    validate_joint_kinematics_ = false;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::set_legacy_path_tolerance(double tolerance_percent) {
    legacy_path_tolerance_percent = tolerance_percent;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::set_legacy_duration_tolerance(double tolerance_percent) {
    legacy_duration_tolerance_percent = tolerance_percent;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::enable_json_output(std::string filename) {
    json_output_filename_ = std::move(filename);
    json_on_failure_only_ = false;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::enable_json_output_on_failure(std::string filename) {
    json_output_filename_ = std::move(filename);
    json_on_failure_only_ = true;
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::enable_event_collection() {
    if (!collector_) {
        collector_ = std::make_shared<trajectory_integration_event_collector>();
    }
    return *this;
}

trajectory_test_fixture& trajectory_test_fixture::suppress_create_failure_assertions() {
    assert_on_create_failure_ = false;
    return *this;
}

void trajectory_test_fixture::try_legacy_comparison(const xmatrix<>& waypoints [[maybe_unused]]) {
#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
    try {
        // Convert waypoints to legacy format (std::list<Eigen::VectorXd>)
        std::list<Eigen::VectorXd> legacy_waypoints;
        for (size_t i = 0; i < waypoints.shape(0); ++i) {
            Eigen::VectorXd wp(static_cast<Eigen::Index>(dof_));
            for (size_t j = 0; j < dof_; ++j) {
                wp(static_cast<Eigen::Index>(j)) = waypoints(i, j);
            }
            legacy_waypoints.push_back(wp);
        }

        // Convert limits to Eigen format
        Eigen::VectorXd legacy_max_vel(static_cast<Eigen::Index>(dof_));
        Eigen::VectorXd legacy_max_acc(static_cast<Eigen::Index>(dof_));
        for (size_t i = 0; i < dof_; ++i) {
            legacy_max_vel(static_cast<Eigen::Index>(i)) = (*traj_opts.max_velocity)(i);
            legacy_max_acc(static_cast<Eigen::Index>(i)) = (*traj_opts.max_acceleration)(i);
        }

        // Create legacy path and trajectory with timing
        const auto legacy_start_time = std::chrono::high_resolution_clock::now();
        const Path legacy_path(legacy_waypoints, path_opts.max_blend_deviation());
        const Trajectory legacy_traj(legacy_path, legacy_max_vel, legacy_max_acc, traj_opts.delta.count());
        const auto legacy_end_time = std::chrono::high_resolution_clock::now();
        legacy_generation_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(legacy_end_time - legacy_start_time).count();

        if (legacy_traj.isValid()) {
            legacy_duration_ = legacy_traj.getDuration();
            legacy_path_length_ = legacy_path.getLength();
            std::cout << "[LEGACY] duration=" << *legacy_duration_ << "s, path_length=" << *legacy_path_length_ << "\n";
        } else {
            std::cout << "[LEGACY] FAILED: invalid trajectory\n";
        }
    } catch (const std::exception& e) {
        std::cout << "[LEGACY] FAILED: " << e.what() << "\n";
    }
#endif
}

void trajectory_test_fixture::validate_against_legacy(const trajectory& traj) {
    BOOST_TEST_CONTEXT("Legacy comparison") {
        const double new_duration = traj.duration().count();
        const double new_path_length = static_cast<double>(traj.path().length());

        if (legacy_path_length_.has_value()) {
            const double path_ratio = new_path_length / *legacy_path_length_;
            const double path_diff_percent = std::abs((new_path_length - *legacy_path_length_) / *legacy_path_length_) * 100.0;

            std::cout << "[COMPARISON] Path: new=" << new_path_length << ", legacy=" << *legacy_path_length_ << ", "
                      << "ratio=" << path_ratio << ", diff=" << path_diff_percent << "%\n";

            // Path should be within configured tolerance
            BOOST_CHECK_CLOSE(new_path_length, *legacy_path_length_, legacy_path_tolerance_percent);
        }

        if (legacy_duration_.has_value()) {
            const double duration_ratio = new_duration / *legacy_duration_;
            const double duration_diff_percent = ((new_duration - *legacy_duration_) / *legacy_duration_) * 100.0;

            std::cout << "[COMPARISON] Duration: new=" << new_duration << "s, legacy=" << *legacy_duration_ << "s, "
                      << "ratio=" << duration_ratio << ", diff=" << duration_diff_percent << "%\n";

            // We should do NO WORSE than legacy: duration should be within configured tolerance
            const double duration_tolerance_multiplier = 1.0 + (legacy_duration_tolerance_percent / 100.0);
            BOOST_CHECK_LE(new_duration, *legacy_duration_ * duration_tolerance_multiplier);

            BOOST_TEST_MESSAGE("Duration comparison: new=" << new_duration << "s vs legacy=" << *legacy_duration_
                                                           << "s (ratio=" << duration_ratio << ", diff=" << duration_diff_percent << "%)");
        }
    }
}

trajectory trajectory_test_fixture::create_and_validate() {
    BOOST_REQUIRE_MESSAGE(waypoints_.size() > 0, "Must set waypoints before calling create_and_validate");

    // Try legacy comparison if enabled
    if (compare_with_legacy_) {
        try_legacy_comparison(waypoints_);
    }

    // Create path
    path p = path::create(waypoints_, path_opts);

    // Validate path length expectation
    if (expected_path_length_.has_value()) {
        const double tol = expected_path_length_->tolerance_percent.value_or(validation_tolerance_percent);
        BOOST_TEST_CONTEXT("Path length expectation") {
            BOOST_CHECK_CLOSE(static_cast<double>(p.length()), static_cast<double>(expected_path_length_->length), tol);
        }
    }

    // Validate that expectations follow TOTG invariant (only if expectations were specified)
    if (!allow_any_events_) {
        // Must have at least 3 events: forward_start, backward_start from endpoint, splice
        const auto& expectations = expectation_observer_->get_expectations();
        BOOST_REQUIRE_MESSAGE(expectations.size() >= 3, "Must have at least 3 expected events (TOTG invariant)");

        BOOST_TEST_CONTEXT("Expectation coherence (TOTG invariant)") {
            // First must be forward_start
            const auto* first_forward = std::get_if<expectation_observer::expected_forward_start>(&expectations.front());
            BOOST_REQUIRE_MESSAGE(first_forward != nullptr, "First expectation must be forward_start (TOTG invariant)");

            // Second-to-last must be backward_start (from endpoint)
            const auto& second_last = expectations[expectations.size() - 2];
            const auto* backward = std::get_if<expectation_observer::expected_backward_start>(&second_last);
            BOOST_REQUIRE_MESSAGE(backward != nullptr, "Second-to-last expectation must be backward_start (TOTG invariant)");

            // Backward expectation must specify k_path_end kind (TOTG always ends with backward from endpoint)
            BOOST_REQUIRE_MESSAGE(backward->kind.has_value(),
                                  "Second-to-last expectation must specify switching point kind (TOTG invariant)");
            BOOST_REQUIRE_MESSAGE(*backward->kind == trajectory::switching_point_kind::k_path_end,
                                  "Second-to-last expectation must be backward_start with k_path_end kind (TOTG invariant)");

            // Sanity check: backward expectation should be approximately at path endpoint (if s is specified)
            if (backward->s.has_value()) {
                const double backward_tol = backward->tolerance_percent.value_or(expectation_observer_->get_default_tolerance_percent());
                BOOST_CHECK_CLOSE(static_cast<double>(*backward->s), static_cast<double>(p.length()), backward_tol);
            }

            // Last must be splice
            const auto* splice = std::get_if<expectation_observer::expected_splice>(&expectations.back());
            BOOST_REQUIRE_MESSAGE(splice != nullptr, "Last expectation must be splice (TOTG invariant)");
        }
    }

    // Set up observer chain - always use composite observer
    traj_opts.observer = &composite_observer_;

    // Add expectation observer if validating events
    if (!allow_any_events_) {
        composite_observer_.add_observer(expectation_observer_);
    }

    // Add collector if JSON output enabled
    if (json_output_filename_.has_value() || collector_) {
        if (!collector_) {
            collector_ = std::make_shared<trajectory_integration_event_collector>();
        }
        composite_observer_.add_observer(collector_);
    }

    // Generate trajectory - catch failures to write diagnostic JSON before propagating
    trajectory traj = [&]() -> trajectory {
        try {
            return trajectory::create(std::move(p), traj_opts);
        } catch (const std::exception& e) {
            if (json_output_filename_.has_value() && collector_) {
                std::ofstream out(*json_output_filename_);
                write_trajectory_json(out, *collector_);
                BOOST_TEST_MESSAGE("Wrote failure JSON to " << *json_output_filename_);
            }
            if (assert_on_create_failure_) {
                BOOST_FAIL(e.what());
            }
            throw;
        }
    }();

    // Verify all expectations were met (only if expectations were specified)
    if (!allow_any_events_) {
        expectation_observer_->verify_all_expectations_met();
    }

    // Validate trajectory-wide expectations
    if (expected_duration_.has_value()) {
        const double tol = expected_duration_->tolerance_percent.value_or(validation_tolerance_percent);
        BOOST_TEST_CONTEXT("Trajectory duration expectation") {
            BOOST_CHECK_CLOSE(traj.duration().count(), expected_duration_->duration.count(), tol);
        }
    }

    if (expected_integration_point_count_.has_value()) {
        BOOST_TEST_CONTEXT("Integration point count expectation") {
            BOOST_CHECK_EQUAL(traj.get_integration_points().size(), *expected_integration_point_count_);
        }
    }

    // Compare with legacy if available
    if (legacy_duration_.has_value() || legacy_path_length_.has_value()) {
        validate_against_legacy(traj);
    }

    validate_path_quality(traj.path(), waypoint_accumulator{waypoints_}, path_opts);

    // Validate trajectory invariants
    validate_trajectory_invariants(traj, validation_tolerance_percent);

    if (validate_joint_kinematics_) {
        validate_joint_kinematics(traj, traj_opts.max_velocity, traj_opts.max_acceleration, validation_tolerance_percent);
    }

    // Write JSON if enabled, after all validation so failure-only mode can check results.
    if (json_output_filename_.has_value() && collector_) {
        const bool has_failures =
            boost::unit_test::results_collector.results(boost::unit_test::framework::current_test_case().p_id).p_assertions_failed > 0;
        if (!json_on_failure_only_ || has_failures) {
            std::ofstream out(*json_output_filename_);
            write_trajectory_json(out, *collector_, &traj);
            BOOST_TEST_MESSAGE("Wrote trajectory JSON to " << *json_output_filename_);
        }
    }

    return traj;
}

std::ostream& operator<<(std::ostream& os, const constraint_profile& profile) {
    return os << profile.name;
}

}  // namespace viam::trajex::totg::test

namespace viam::trajex::totg {

bool operator==(const trajectory::integration_observer::started_forward_event& a,
                const trajectory::integration_observer::started_forward_event& b) noexcept {
    return a.start == b.start;
}

bool operator==(const trajectory::integration_observer::limit_hit_event& a,
                const trajectory::integration_observer::limit_hit_event& b) noexcept {
    return a.breach == b.breach && a.s_dot_max_acc == b.s_dot_max_acc && a.s_dot_max_vel == b.s_dot_max_vel;
}

bool operator==(const trajectory::integration_observer::started_backward_event& a,
                const trajectory::integration_observer::started_backward_event& b) noexcept {
    return a.start == b.start && a.kind == b.kind;
}

bool operator==(const trajectory::integration_observer::splice_event& a, const trajectory::integration_observer::splice_event& b) noexcept {
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
        // NaN compares unequal to everything, so a NaN on only one side is a mismatch, as it should be.
        const auto nans = std::isnan(static_cast<double>(lhs.s_ddot)) && std::isnan(static_cast<double>(rhs.s_ddot));
        if (!nans && (lhs.s_ddot != rhs.s_ddot)) {
            return false;
        }
        return true;
    });
}

}  // namespace viam::trajex::totg
