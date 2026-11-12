// Test utility function implementations
#include <viam/trajex/totg/test/test_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/arc_length.hpp>

#if __has_include(<xtensor/reducers/xnorm.hpp>)
#include <xtensor/reducers/xnorm.hpp>
#else
#include <xtensor/xnorm.hpp>
#endif

#include <boost/test/unit_test.hpp>

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
    // most an ULP of the end, so the radius, and the curvature with it, by at most that relative to
    // the arc's length, and the division that refits it rounds once more.
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

    // The size of the coordinates where each segment's positions are computed. A position on a segment
    // sums its start or apex, no larger than its ends plus its length, and an offset no larger than its
    // length, so the larger end plus the length bounds every term. Scaling per segment rather than by the
    // whole path keeps a single far waypoint from loosening the bound everywhere else.
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

                // A unit tangent within k_tangent_error_bound_ulps of one has a norm within that many ULPs
                // of 1, and taking the norm sums dof squares, which rounds by at most (dof / 2 + 1) eps
                // relative.
                const auto tangent_norm_bound = (path::k_tangent_error_bound_ulps + (dof / 2.0) + 1.0) * eps;
                BOOST_CHECK_SMALL(xt::norm_l2(segment.tangent(segment.start()))() - 1.0, tangent_norm_bound);
                BOOST_CHECK_SMALL(xt::norm_l2(segment.tangent(segment.end()))() - 1.0, tangent_norm_bound);

                if (segment.is<path::segment::circular>()) {
                    // An arc traversed at unit speed turns its tangent by its length times its curvature,
                    // which is less than pi. The angle between two directions moves by no more than either
                    // direction does, so each end's tangent contributes at most k_tangent_error_bound_ulps.
                    // Computing that angle normalizes two vectors and takes the norms of their difference and
                    // sum, each a sum of dof terms, which rounds by under (dof + 4) eps in all. The curvature's
                    // relative error, k_curvature_error_bound_ulps, and the rounding of the product with the
                    // length carry across the angle, at most pi.
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

        // Neighbors meet in position, and, where an arc is involved, in tangent; a join of two lines is a
        // corner left unblended, which turns by design. Each side of a join is within
        // k_position_error_bound_ulps of the same exact point, scaled by the larger of the two segments'
        // coordinates, plus an ULP of the arc length at the join for addressing the end of the segment
        // before it. Refitting an arc's radius to its span on the path moves the arc's ends by at most the
        // rounding of that span, another ULP of the arc length there. Each side's tangent, a line's as much
        // as an arc's, is within k_tangent_error_bound_ulps of the exact tangent they share.
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

        // The path visits every waypoint, in order, within the deviation budget: each waypoint is within
        // budget of the path somewhere at or after where its predecessor was visited. The search walks
        // forward from that visit and takes the first point within budget. It resumes on the segment the
        // predecessor was visited on, since consecutive waypoints can share a segment, as colinearized ones
        // do. There it must not look behind the previous visit: a path that reverses comes back over a
        // waypoint's position on a later segment, but the segment the search resumes on can pass that
        // position too, earlier, before the predecessor. On a segment the distance to a point falls to a
        // minimum and rises again, along a line and along an arc turning by less than pi alike, so the
        // nearest point not before the previous visit is the segment's nearest point clamped to it.
        //
        // The exact path passes within the larger deviation budget of each waypoint, and a computed position
        // on it within k_position_error_bound_ulps of that, plus an ULP of the arc length for addressing it
        // and another for the refit of an arc.
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

}  // namespace viam::trajex::totg::test
