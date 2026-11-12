#include <viam/trajex/totg/path.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if __has_include(<xtensor/reducers/xnorm.hpp>)
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/reducers/xnorm.hpp>
#else
#include <xtensor/xnoalias.hpp>
#include <xtensor/xnorm.hpp>
#endif

#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/angles.hpp>

namespace viam::trajex::totg {

path::options::options()
    : max_blend_deviation_(k_default_max_deviation),
      max_linear_deviation_(k_default_max_deviation),
      min_blend_curvature_(k_default_min_blend_curvature),
      max_blend_curvature_(k_default_max_blend_curvature) {}

path::options& path::options::set_max_deviation(double deviation) {
    return set_max_blend_deviation(deviation).set_max_linear_deviation(deviation);
}

path::options& path::options::set_max_blend_deviation(double deviation) {
    max_blend_deviation_ = deviation;
    return *this;
}

path::options& path::options::set_max_linear_deviation(double deviation) {
    max_linear_deviation_ = deviation;
    return *this;
}

path::options& path::options::set_max_blend_curvature(double curvature) {
    max_blend_curvature_ = curvature;
    return *this;
}

path::options& path::options::set_min_blend_curvature(double curvature) {
    min_blend_curvature_ = curvature;
    return *this;
}

double path::options::max_blend_deviation() const noexcept {
    return max_blend_deviation_;
}

double path::options::max_linear_deviation() const noexcept {
    return max_linear_deviation_;
}

double path::options::max_blend_curvature() const noexcept {
    return max_blend_curvature_;
}

double path::options::min_blend_curvature() const noexcept {
    return min_blend_curvature_;
}

path::segment::linear::linear(xrank_checked<xvector<>> start, const xrank_checked<xvector<>>& end)
    : start{std::move(start).take()}, length{0.0} {
    const auto diff = *end - this->start;
    const double norm = xt::norm_l2(diff)();

    // The only truly degenerate case is coincident endpoints (norm == 0), which would
    // make the unit direction undefined. For tiny-but-positive norms, the direction is
    // well-defined: callers constructing via start + len * unit_dir produce diff = len * dir,
    // and diff / len = dir exactly. TOTG handles short linear segments without issue.
    if (norm == 0.0) {
        throw std::invalid_argument{"Linear segment: start and end must be different"};
    }

    xt::noalias(this->unit_direction) = diff / norm;
    this->length = arc_length{norm};
}

path::segment::linear::linear(xrank_checked<xvector<>> start, xrank_checked<xvector<>> unit_direction, arc_length length)
    : start{std::move(start).take()}, unit_direction{std::move(unit_direction).take()}, length{length} {
    if (static_cast<double>(length) <= 0.0) {
        throw std::invalid_argument{"Linear segment: length must be positive"};
    }
}

path::segment::circular::circular(
    xrank_checked<xvector<>> apex, xrank_checked<xvector<>> t, xrank_checked<xvector<>> n, double radius, double angle_rads)
    : apex{std::move(apex).take()}, t{std::move(t).take()}, n{std::move(n).take()}, radius{radius}, angle_rads{angle_rads} {
    // The comparisons below are false for NaN, so non-finite input must be rejected first.
    if (!xt::all(xt::isfinite(this->apex)) || !xt::all(xt::isfinite(this->t)) || !xt::all(xt::isfinite(this->n)) ||
        !std::isfinite(radius) || !std::isfinite(angle_rads)) {
        throw std::invalid_argument{"Circular segment: geometry must be finite"};
    }

    // A blend turns by less than pi, which is what leaves each joint at most one extremum on it.
    if (radius <= 0.0 || angle_rads <= 0.0 || angle_rads >= std::numbers::pi) {
        throw std::invalid_argument{"Circular segment: radius must be positive and angle must lie in (0, pi)"};
    }

    // Normalizing a vector rounds once per coordinate, as does summing its squares again here to
    // check it, and the residual a Gram-Schmidt step leaves in a dot product is of the same kind.
    // So the bound is a few ULPs per coordinate, not a tolerance on the geometry.
    constexpr std::size_t k_rounding_ulps_per_coordinate = 4;
    const auto rounding_ulps = k_rounding_ulps_per_coordinate * this->t.size();
    const auto rounding_bound = static_cast<double>(rounding_ulps) * std::numeric_limits<double>::epsilon();

    const auto t_norm = xt::norm_l2(this->t)();
    const auto n_norm = xt::norm_l2(this->n)();
    if (std::abs(t_norm - 1.0) > rounding_bound || std::abs(n_norm - 1.0) > rounding_bound) {
        throw std::invalid_argument{"Circular segment: t and n must be unit vectors"};
    }

    const auto dot_product = xt::sum(this->t * this->n)();
    if (std::abs(dot_product) > rounding_bound) {
        throw std::invalid_argument{"Circular segment: t and n must be perpendicular"};
    }
}

path::segment::segment(linear data) : data_{std::move(data)} {}

path::segment::segment(circular data) : data_{std::move(data)} {}

path::segment::view::view(const class segment& seg, arc_length start, arc_length end) noexcept : seg_{seg}, start_{start}, end_{end} {}

const path::segment& path::segment::view::segment() const noexcept {
    return seg_.get();
}

arc_length path::segment::view::start() const noexcept {
    return start_;
}

arc_length path::segment::view::end() const noexcept {
    return end_;
}

arc_length path::segment::view::length() const noexcept {
    return end_ - start_;
}

namespace {

void require_dof(std::span<const double> out, std::size_t dof) {
    if (out.size() != dof) [[unlikely]] {
        throw std::invalid_argument{"Output span size does not match path degrees of freedom"};
    }
}

}  // namespace

std::size_t path::segment::view::dof_() const {
    return std::visit(
        [](const auto& seg_data) -> std::size_t {
            using T = std::decay_t<decltype(seg_data)>;

            if constexpr (std::is_same_v<T, segment::linear>) {
                return seg_data.start.shape(0);
            } else if constexpr (std::is_same_v<T, segment::circular>) {
                return seg_data.apex.shape(0);
            }
        },
        seg_.get().data_);
}

void path::segment::view::configuration(arc_length s, std::span<double> out) const {
    // `Correction 1`: Need to offset before Kunz & Stilman equation 7.
    const arc_length local_s = s - start_;

    if (local_s < arc_length{0.0} || local_s > length()) [[unlikely]] {
        throw std::out_of_range{"Arc length outside segment bounds"};
    }

    std::visit(
        [local_s, out](const auto& seg_data) {
            using T = std::decay_t<decltype(seg_data)>;

            if constexpr (std::is_same_v<T, segment::linear>) {
                require_dof(out, seg_data.start.shape(0));

                // Linear interpolation: config = start + local_s * unit_direction
                const double distance = static_cast<double>(local_s);

                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = seg_data.start(i) + (distance * seg_data.unit_direction(i));
                }
            } else if constexpr (std::is_same_v<T, segment::circular>) {
                require_dof(out, seg_data.apex.shape(0));

                // Circular arc configuration - Kunz & Stilman equation 7, measured from the apex, so
                // that phi runs over [-angle / 2, angle / 2]. The 1 - cos(phi) term is written as
                // 2 sin^2(phi / 2), which keeps its digits when phi is small.
                const auto phi = (static_cast<double>(local_s) / seg_data.radius) - (seg_data.angle_rads / 2.0);
                const auto sin_half_phi = std::sin(phi / 2.0);
                const auto along_t = seg_data.radius * std::sin(phi);
                const auto along_n = seg_data.radius * 2.0 * sin_half_phi * sin_half_phi;

                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = seg_data.apex(i) + (along_t * seg_data.t(i)) + (along_n * seg_data.n(i));
                }
            }
        },
        seg_.get().data_);
}

void path::segment::view::tangent(arc_length s, std::span<double> out) const {
    // `Correction 1`: Need to offset before Kunz & Stilman equation 8.
    const arc_length local_s = s - start_;

    if (local_s < arc_length{0.0} || local_s > length()) [[unlikely]] {
        throw std::out_of_range{"Arc length outside segment bounds"};
    }

    std::visit(
        [local_s, out](const auto& seg_data) {
            using T = std::decay_t<decltype(seg_data)>;

            if constexpr (std::is_same_v<T, segment::linear>) {
                require_dof(out, seg_data.start.shape(0));

                // Linear segment: constant unit tangent
                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = seg_data.unit_direction(i);
                }
            } else if constexpr (std::is_same_v<T, segment::circular>) {
                require_dof(out, seg_data.apex.shape(0));

                // Circular arc unit tangent - Kunz & Stilman equation 8, measured from the apex:
                const auto phi = (static_cast<double>(local_s) / seg_data.radius) - (seg_data.angle_rads / 2.0);
                const auto cos_phi = std::cos(phi);
                const auto sin_phi = std::sin(phi);

                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = (seg_data.t(i) * cos_phi) + (seg_data.n(i) * sin_phi);
                }
            }
        },
        seg_.get().data_);
}

void path::segment::view::curvature(arc_length s, std::span<double> out) const {
    // `Correction 1`: Need to offset before Kunz & Stilman equation 9.
    const arc_length local_s = s - start_;

    if (local_s < arc_length{0.0} || local_s > length()) [[unlikely]] {
        throw std::out_of_range{"Arc length outside segment bounds"};
    }

    std::visit(
        [local_s, out](const auto& seg_data) {
            using T = std::decay_t<decltype(seg_data)>;

            if constexpr (std::is_same_v<T, segment::linear>) {
                require_dof(out, seg_data.start.shape(0));

                // Linear segment: zero curvature vector
                std::ranges::fill(out, 0.0);
            } else if constexpr (std::is_same_v<T, segment::circular>) {
                require_dof(out, seg_data.apex.shape(0));

                // Circular arc curvature vector - Kunz & Stilman equation 9, measured from the apex:
                const auto phi = (static_cast<double>(local_s) / seg_data.radius) - (seg_data.angle_rads / 2.0);
                const auto cos_phi = std::cos(phi);
                const auto sin_phi = std::sin(phi);
                const auto inverse_radius = 1.0 / seg_data.radius;

                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = inverse_radius * ((seg_data.n(i) * cos_phi) - (seg_data.t(i) * sin_phi));
                }
            }
        },
        seg_.get().data_);
}

std::optional<arc_length> path::segment::view::coordinate_extremum(std::size_t joint) const {
    return std::visit(
        [this, joint](const auto& seg_data) -> std::optional<arc_length> {
            using T = std::decay_t<decltype(seg_data)>;

            if constexpr (std::is_same_v<T, segment::linear>) {
                if (joint >= seg_data.start.shape(0)) [[unlikely]] {
                    throw std::out_of_range{"Joint index exceeds path degrees of freedom"};
                }
                return std::nullopt;
            } else if constexpr (std::is_same_v<T, segment::circular>) {
                if (joint >= seg_data.apex.shape(0)) [[unlikely]] {
                    throw std::out_of_range{"Joint index exceeds path degrees of freedom"};
                }

                // The joint's component of the tangent, cos(phi) t_i + sin(phi) n_i, is zero where
                // tan(phi) = -t_i / n_i, which within (-pi/2, pi/2) has the single root atan(-t_i / n_i).
                // The arc turns by less than pi, so it lies within that interval too, and any root on
                // the arc is this one. With n_i zero there is no root inside the interval, or, if t_i
                // is zero too, the joint does not move on the arc at all.
                //
                // The quotient is safe: t and n are unit vectors, so it is finite unless n_i is so small
                // that it overflows, and atan takes an infinity to +-pi/2, where the root then is.
                const auto t_i = seg_data.t(joint);
                const auto n_i = seg_data.n(joint);
                if (n_i == 0.0) {
                    return std::nullopt;
                }
                const auto phi = std::atan(-t_i / n_i);

                // Report the root wherever it falls, on the arc or on the circle beyond it. Whether one
                // just off the arc should count is the caller's to decide, and it needs to know how far
                // off it is to do that.
                return start_ + arc_length{seg_data.radius * (phi + (seg_data.angle_rads / 2.0))};
            }
        },
        seg_.get().data_);
}

xvector<> path::segment::view::configuration(arc_length s) const {
    auto result = xvector<>::from_shape(std::array<std::size_t, 1>{dof_()});
    configuration(s, std::span<double>{result.data(), result.size()});
    return result;
}

xvector<> path::segment::view::tangent(arc_length s) const {
    auto result = xvector<>::from_shape(std::array<std::size_t, 1>{dof_()});
    tangent(s, std::span<double>{result.data(), result.size()});
    return result;
}

xvector<> path::segment::view::curvature(arc_length s) const {
    auto result = xvector<>::from_shape(std::array<std::size_t, 1>{dof_()});
    curvature(s, std::span<double>{result.data(), result.size()});
    return result;
}

path::path(std::vector<positioned_segment> segments, size_t dof, arc_length length)
    : segments_{std::move(segments)}, dof_{dof}, length_{length} {}

path path::create(const waypoint_accumulator& waypoints, const options& opts) {
    if (waypoints.size() < 2) {
        throw std::invalid_argument{"Path requires at least 2 waypoints"};
    }

    if (!std::isfinite(opts.max_blend_deviation()) || opts.max_blend_deviation() < 0.0) {
        throw std::invalid_argument{"Max blend deviation must be finite and non-negative"};
    }

    if (!std::isfinite(opts.max_linear_deviation()) || opts.max_linear_deviation() < 0.0) {
        throw std::invalid_argument{"Max linear deviation must be finite and non-negative"};
    }

    if (!std::isfinite(opts.min_blend_curvature()) || opts.min_blend_curvature() <= 0.0) {
        throw std::invalid_argument{"Min blend curvature must be finite and positive"};
    }

    if (!std::isfinite(opts.max_blend_curvature()) || opts.max_blend_curvature() < 0.0) {
        throw std::invalid_argument{"Max blend curvature must be finite and non-negative"};
    }

    if (opts.max_blend_curvature() < opts.min_blend_curvature()) {
        throw std::invalid_argument{"Max blend curvature must be >= min blend curvature"};
    }

    // TODO(RSDK-12771): Revisit this algorithm, and write a more graceful one.

    // Build a path from waypoints using two techniques: tube coalescing to skip waypoints
    // that lie within a tolerance tube around linear segments, and circular blends at corners
    // to smooth transitions between non-colinear segments. Both techniques keep the path
    // within specified deviation tolerances of the original waypoints.
    //
    // The algorithm scans through waypoints with a "locus" at each potential corner. At each
    // locus, we look backward to the segment start and forward to the next waypoint to decide
    // if the locus can be skipped (coalesced) or if it represents an actual corner that needs
    // either a hard stop or a circular blend.

    struct blend_geometry {
        segment::circular circular_seg;
        double trim_distance;      // Distance trimmed from both segments
        bool takes_half_incoming;  // trim_distance is exactly half of the incoming segment
        bool takes_half_outgoing;  // trim_distance is exactly half of the outgoing segment
    };

    // Try to create a circular blend arc at a corner following the algorithm from
    // Kunz & Stilman Section IV. The blend arc is tangent to both the incoming and
    // outgoing segments, trimming equal distances from each side. The blend keeps
    // the path within max_blend_deviation of the original corner waypoint.
    const auto try_create_blend =
        [&opts](const xvector<>& current_pos, const xvector<>& corner, const xvector<>& next_waypoint) -> std::optional<blend_geometry> {
        if (opts.max_blend_deviation() <= 0.0) {
            return std::nullopt;
        }

        const auto incoming = corner - current_pos;
        const auto outgoing = next_waypoint - corner;

        const auto incoming_norm = xt::norm_l2(incoming)();
        const auto outgoing_norm = xt::norm_l2(outgoing)();

        // Need non-zero segments to compute angle
        if (incoming_norm <= 0.0 || outgoing_norm <= 0.0) {
            return std::nullopt;
        }

        const auto incoming_unit = incoming / incoming_norm;
        const auto outgoing_unit = outgoing / outgoing_norm;

        // If these are numerically identical, then these are FP exact colinear segments, and we do
        // not need a blend.
        if (incoming_unit == outgoing_unit) {
            return std::nullopt;
        }

        // The sum and the difference of the two unit directions carry everything the blend needs.
        // For a turn by alpha their lengths are 2 cos(alpha/2) and 2 sin(alpha/2), each good to about
        // eps absolute.
        const xvector<> sum = incoming_unit + outgoing_unit;
        const xvector<> difference = outgoing_unit - incoming_unit;
        const auto sum_norm = xt::norm_l2(sum)();
        const auto difference_norm = xt::norm_l2(difference)();

        // Kunz & Stilman equation 2. Clamp to [-1, 1]: rounding can carry the dot product of two unit
        // vectors slightly past either end, and the exact checks below treat those as the ends.
        const auto dot = std::clamp(xt::sum(incoming_unit * outgoing_unit)(), -1.0, 1.0);

        // Exact collinear (angle=0): the bisector and radius formula both have singularities here.
        // No blend is needed since the tangent is unchanged, and TOTG won't mandate s_dot=0.
        // All near-collinear cases are handled by the min_blend_curvature cap below.
        if (dot == 1.0) {
            return std::nullopt;
        }

        // Exact reversal (angle=pi): the blend formula is algebraically undefined here.
        //
        // Near-reversal is handled by the max_blend_curvature check below. We guard this case
        // explicitly rather than relying on tan(pi/2) being large enough to trip that check.
        if (dot == -1.0) {
            return std::nullopt;
        }

        // Equation 2 takes the angle as the arccosine of the dot product, which has no digits near
        // 0 or pi: a corner turning by 1.5e-13 reads as its floor of 1.5e-8 (RSDK-14911). The norms
        // give it to rounding at any angle, and the tangents of its half and quarter as plain ratios:
        // tan(alpha/2) = |d| / |s|, and tan(alpha/4) = sin(alpha/2) / (1 + cos(alpha/2)) = |d| / (2 + |s|).
        // The difference cannot vanish, since the directions differ, so the angle is positive; if
        // the sum vanishes, the angle is pi and tan(alpha/2) is infinite, which the near-reversal
        // check below rejects. So any arc built lies strictly in (0, pi), as segment::circular requires.
        const auto angle = 2.0 * std::atan2(difference_norm, sum_norm);
        const auto tan_half = difference_norm / sum_norm;
        const auto tan_quarter = difference_norm / (2.0 + sum_norm);

        // Compute trim following Kunz & Stilman equations 3-4. Three constraints apply: can't
        // trim more than half of either segment, and the deviation at the blend must stay within
        // the budget. The deviation formula inverted is: trim = deviation / tan(alpha/4). This
        // form (via the half-angle identity sin(x)/(1-cos(x)) = cot(x/2)) is well-conditioned
        // at small angles, unlike the equivalent sin(alpha/2)/(1-cos(alpha/2)) expression.
        const auto max_trim_from_segments = std::min(incoming_norm / 2.0, outgoing_norm / 2.0);
        const auto natural_trim = (max_trim_from_segments * tan_quarter <= opts.max_blend_deviation())
                                      ? max_trim_from_segments
                                      : opts.max_blend_deviation() / tan_quarter;

        // All curvature checks stay in trim space to avoid chains of divisions. The natural
        // radius is trim/tan(alpha/2), so:
        //   1/radius > kmax  <=>  tan(alpha/2) > kmax * trim  (near-reversal)
        //   1/radius < kmin  <=>  trim > tan(alpha/2) / kmin  (near-collinear)
        //
        // Near-reversal: emit L-L; the TOTG tangent continuity check enforces s_dot=0.
        if (tan_half > opts.max_blend_curvature() * natural_trim) {
            return std::nullopt;
        }

        // Near-collinear: cap trim at tan(alpha/2)/kmin. The capped radius is exactly
        // 1/kmin with no rounding; the arc retains exact C1 continuity at both boundaries.
        // Between the two curvature checks the radius lies in [1/kmax, 1/kmin], so dividing by
        // tan(alpha/2) here can neither blow up nor vanish.
        const auto trim_distance = std::min(natural_trim, tan_half / opts.min_blend_curvature());
        const auto radius = trim_distance / tan_half;

        // Construct the arc from its apex rather than from its center as Kunz & Stilman equations
        // 5-6 do. The paper puts the center on the bisector at radius / cos(alpha/2) from the corner,
        // but the bisector is the difference of two unit vectors, which at a shallow corner are
        // nearly equal, so its direction has few correct digits, and at the min_blend_curvature
        // radius cap that error carries the center, and the whole arc with it, far off the path
        // (RSDK-14911). Everything about the apex is bounded by the arc's length instead.
        //
        // The frame comes from the sum and the difference too: the tangent at the apex lies along
        // the sum, and the normal, toward the inside of the corner, along the difference. The
        // longer of the two is at least sqrt(2), while the shorter cancels nearly equal or nearly
        // opposite vectors and has a direction good only to about eps over its length: the
        // difference at a shallow corner, the sum at a near reversal. So normalize the longer and
        // project it out of the shorter. Within the plane, the shorter's direction then follows
        // from the longer's, and what error remains tilts the plane, which the ends of the arc see
        // scaled by the shorter's length, so they meet the incoming and outgoing directions to
        // rounding at any angle. Normalizing introduces no scale, since a vector's direction does
        // not depend on its length, but if the shorter cancels completely there is no plane to put
        // an arc in, and no blend.
        const bool sum_is_longer = (sum_norm >= difference_norm);

        const xvector<> longer_unit = sum_is_longer ? xvector<>{sum / sum_norm} : xvector<>{difference / difference_norm};
        const auto& shorter = sum_is_longer ? difference : sum;
        const xvector<> shorter_perpendicular = shorter - (xt::sum(shorter * longer_unit)() * longer_unit);
        const auto shorter_perpendicular_norm = xt::norm_l2(shorter_perpendicular)();
        if (shorter_perpendicular_norm == 0.0) {
            return std::nullopt;
        }
        const xvector<> shorter_unit = shorter_perpendicular / shorter_perpendicular_norm;

        const auto& apex_tangent = sum_is_longer ? longer_unit : shorter_unit;
        const auto& apex_normal = sum_is_longer ? shorter_unit : longer_unit;

        // The apex is the point of the arc nearest the corner, at the blend's deviation from it,
        // which is trim * tan(alpha/4) by the relation the trim was computed from above. This needs
        // no cosine of the half angle, which vanishes at a reversal.
        const auto apex = corner + ((trim_distance * tan_quarter) * apex_normal);

        // Report which halves the trim took, by the same expressions that capped it. Halving is exact,
        // so these are equalities between identically computed values, not tolerances. The construction
        // loop uses them to recognize a segment that two blends consume completely.
        return blend_geometry{.circular_seg = segment::circular{apex, apex_tangent, apex_normal, radius, angle},
                              .trim_distance = trim_distance,
                              .takes_half_incoming = (trim_distance == incoming_norm / 2.0),
                              .takes_half_outgoing = (trim_distance == outgoing_norm / 2.0)};
    };

    // Run colinearization as a single pass before constructing circular blends.

    auto waypoints_range = std::views::all(waypoints);
    std::optional<waypoint_accumulator> colinearized;
    if (opts.max_linear_deviation() != 0.0) {
        // The colinearity test splits into a half depending only on the segment and a half
        // depending on the locus. The loop below tests every waypoint it has skipped against
        // the same segment, so computing the first half once per segment rather than once per
        // locus removes most of the work.
        //
        // The two lambdas are not independent: `segment_deviation_context` holds state between
        // them, so `begin_segment` must run for a segment before any `locus_within_tolerance`
        // against it, and the scratch vector must not be read across a later `begin_segment`.
        struct segment_deviation_context {
            xvector<> start_to_next;
            double length_sq;
        };
        segment_deviation_context segment_ctx{xvector<>::from_shape(std::array<std::size_t, 1>{waypoints.dof()}), 0.0};

        // Deviation is measured against a tube of this radius around the segment. Hoisted
        // because the enclosing branch has already established that the tolerance is non-zero.
        const auto radius = opts.max_linear_deviation() / 2.0;

        // Establishes the segment that loci are tested against. Returns false when start and
        // next are the same position, because the locus is then an intentional intermediate
        // goal rather than a point along a line, and nothing may be coalesced through it.
        const auto begin_segment = [&segment_ctx](const auto& start, const auto& next) -> bool {
            xt::noalias(segment_ctx.start_to_next) = next - start;

            if (xt::all(xt::equal(segment_ctx.start_to_next, 0.0))) {
                return false;
            }

            segment_ctx.length_sq = xt::sum(segment_ctx.start_to_next * segment_ctx.start_to_next)();
            return true;
        };

        const auto locus_within_tolerance = [&segment_ctx, radius](const auto& start, const auto& locus) -> bool {
            const auto start_to_locus = locus - start;
            const auto start_to_locus_dot_direction = xt::sum(start_to_locus * segment_ctx.start_to_next)();

            // Monotonic advancement check: reject if locus would require going backward from
            // start or forward past next. This ensures we only skip waypoints that maintain
            // forward progress along the segment direction.
            //
            // We check the bounds BEFORE dividing to avoid division-by-near-zero issues.
            // Since length_sq > 0, we can multiply through the inequality:
            //   t < 0  becomes  dot < 0
            //   t > 1  becomes  dot > length_sq
            if (start_to_locus_dot_direction < 0.0 || start_to_locus_dot_direction > segment_ctx.length_sq) {
                return false;
            }

            // Now safe to divide: bounds check guarantees 0 <= t <= 1
            const auto t = start_to_locus_dot_direction / segment_ctx.length_sq;
            const auto projected_point = start + (t * segment_ctx.start_to_next);
            const auto deviation_vector = locus - projected_point;
            const auto deviation = xt::norm_l2(deviation_vector)();

            return deviation <= radius;
        };

        std::vector<decltype(waypoints_range.begin())> skipped;
        for (auto base = waypoints_range.begin(); base != waypoints_range.end();) {
            if (!colinearized) {
                colinearized.emplace(*base);
            } else {
                colinearized->add_waypoint(*base);
            }

            skipped.clear();
            auto candidate = std::next(base);

            while (candidate != waypoints_range.end()) {
                auto target = std::next(candidate);
                if (target == waypoints_range.end()) {
                    break;
                }
                if (!begin_segment(*base, *target)) {
                    break;
                }
                if (!locus_within_tolerance(*base, *candidate)) {
                    break;
                }
                if (!std::ranges::all_of(skipped, [&](auto elt) { return locus_within_tolerance(*base, *elt); })) {
                    break;
                }
                skipped.push_back(candidate);
                candidate = target;
            }

            base = candidate;
        }
        waypoints_range = std::views::all(std::as_const(*colinearized));
    }

    std::vector<positioned_segment> segments;
    arc_length cumulative_length{0.0};
    auto segment_start = waypoints_range.begin();

    // Track the current position on the path we're building. This must be stored as an actual
    // configuration copy (not just an iterator) because after creating a circular blend, the
    // current position becomes the blend exit point, which is a computed position between
    // waypoints rather than one of the original waypoints in the accumulator.
    xvector<> current_position = *segment_start;

    // Whether the blend at segment_start took exactly half of the segment from segment_start to
    // the current locus. False when segment_start is a hard waypoint rather than a blend.
    bool previous_blend_takes_half_of_segment = false;

    for (auto locus = std::next(waypoints_range.begin()); locus != waypoints_range.end(); ++locus) {
        auto next = std::next(locus);

        // If we're at the last waypoint, we must emit the final segment
        const bool at_last = (next == waypoints_range.end());

        // Locus represents a corner (or the final waypoint). We need to emit the segment
        // from current_position to this locus, and if it's a corner, decide whether to
        // create a circular blend or stop at the hard waypoint.

        // Pass *segment_start (the original segment's start waypoint) rather than
        // current_position (which may be a blend exit point part-way along the segment).
        // This ensures the trim distance cap uses the full original segment length, not
        // just the remaining portion after a prior blend.
        const auto blend_opt = !at_last ? try_create_blend(*segment_start, *locus, *next) : std::nullopt;

        if (blend_opt) {
            const auto& blend = *blend_opt;
            const auto incoming = *locus - current_position;
            const auto dist_to_locus = xt::norm_l2(incoming)();
            const auto incoming_unit = incoming / dist_to_locus;

            // `Divergent Behavior 5`: When adjacent blends fully consume the connecting segment,
            // the two circular arcs are emitted adjacent (C-C). The paper assumes L-C-L topology.
            //
            // Every trim is capped at half of each adjacent segment, so the only way two trims can
            // add up to the whole segment is for each to take exactly half of it. Both blends report
            // that from the comparison that set their trims, so we know the segment is gone without
            // measuring what remains. The measurement cannot be trusted to say so: the distance left
            // after the previous blend is rebuilt from its exit point, and subtracting this trim from
            // it can leave a residue of a few ULPs rather than zero, enough to survive the guard below
            // and emit a sliver that the integrator then has to stand on.
            const bool segment_fully_consumed = previous_blend_takes_half_of_segment && blend.takes_half_incoming;

            // Otherwise, compute the incoming linear segment length as a scalar subtraction to avoid
            // catastrophic cancellation from reconstructing two nearly-equal endpoint positions.
            //
            // A segment is only worth emitting if there is somewhere to be inside it. Positions
            // are addressed in the path's own arc length, so that means a representable value
            // strictly between the endpoints: step one value inward from each end and require
            // that they have not crossed. One ULP leaves nowhere to stand between the endpoints
            // and no way to advance through the segment; two leaves exactly one position, which
            // is the least that makes the segment a place rather than a boundary. This asks the
            // number system rather than a tolerance, so it introduces no epsilon.
            const auto incoming_length = dist_to_locus - blend.trim_distance;
            const auto span_start = static_cast<double>(cumulative_length);
            const auto span_end = span_start + incoming_length;
            if (!segment_fully_consumed && (span_end > span_start) &&
                (std::nextafter(span_start, span_end) <= std::nextafter(span_end, span_start))) {
                segment::linear incoming_segment{current_position, incoming_unit, arc_length{incoming_length}};
                segments.push_back({.seg = segment{std::move(incoming_segment)}, .start = cumulative_length});
                cumulative_length += arc_length{incoming_length};
            }

            segments.push_back({.seg = segment{blend.circular_seg}, .start = cumulative_length});
            cumulative_length += arc_length{blend.circular_seg.radius * blend.circular_seg.angle_rads};

            // After the blend, we're positioned at the blend exit point on the outgoing segment.
            // This position doesn't correspond to any waypoint in the accumulator, which is why
            // current_position must be a configuration copy rather than an iterator.
            const auto outgoing = *next - *locus;
            const auto outgoing_unit = outgoing / xt::norm_l2(outgoing)();
            xt::noalias(current_position) = *locus + (blend.trim_distance * outgoing_unit);

            // Move segment_start forward for coalescing calculations. Even though current_position
            // is between locus and next, we use locus as the reference point for determining if
            // future waypoints can be coalesced.
            segment_start = locus;
            previous_blend_takes_half_of_segment = blend.takes_half_outgoing;
        } else {
            // No blend: emit linear from current_position to locus.
            segment::linear linear_data{current_position, *locus};
            const auto linear_data_length = linear_data.length;
            segments.push_back({.seg = segment{std::move(linear_data)}, .start = cumulative_length});
            cumulative_length += linear_data_length;

            xt::noalias(current_position) = *locus;
            segment_start = locus;
            previous_blend_takes_half_of_segment = false;
        }
    }

    if (segments.empty()) {
        throw std::invalid_argument{"Path requires at least one non-zero length segment"};
    }

    return path{std::move(segments), waypoints.dof(), cumulative_length};
}

arc_length path::length() const noexcept {
    return length_;
}

size_t path::size() const noexcept {
    return segments_.size();
}

bool path::empty() const noexcept {
    return segments_.empty();
}

path::const_iterator path::begin() const noexcept {
    return const_iterator{this, segments_.begin()};
}

path::const_iterator path::end() const noexcept {
    return const_iterator{this, segments_.end()};
}

path::const_iterator path::cbegin() const noexcept {
    return begin();
}

path::const_iterator path::cend() const noexcept {
    return end();
}

path::const_iterator::const_iterator() noexcept : path_{nullptr} {}

path::const_iterator::const_iterator(const path* p, std::vector<positioned_segment>::const_iterator it) noexcept : path_{p}, it_{it} {}

path::segment::view path::const_iterator::operator*() const {
    // Compute end: next segment's start, or path length
    auto next = std::next(it_);
    const arc_length end = (next != path_->segments_.end()) ? next->start : path_->length_;

    return {it_->seg, it_->start, end};
}

path::const_iterator& path::const_iterator::operator++() noexcept {
    ++it_;
    return *this;
}

path::const_iterator path::const_iterator::operator++(int) noexcept {
    auto tmp = *this;
    ++(*this);
    return tmp;
}

path::const_iterator& path::const_iterator::operator--() noexcept {
    --it_;
    return *this;
}

path::const_iterator path::const_iterator::operator--(int) noexcept {
    auto tmp = *this;
    --(*this);
    return tmp;
}

bool path::const_iterator::operator==(const const_iterator& other) const noexcept {
    return it_ == other.it_;
}

size_t path::dof() const noexcept {
    return dof_;
}

path::segment::view path::operator()(arc_length s) const {
    if (segments_.empty()) [[unlikely]] {
        throw std::runtime_error{"Path has no segments"};
    }

    if (s > length_) [[unlikely]] {
        throw std::out_of_range{"Arc length exceeds path length"};
    }

    // Use binary search to find the segment containing s. We use upper_bound to find the
    // first segment that starts AFTER s, then step back one position to get the segment
    // that actually contains s. This gives O(log n) lookup for arbitrary queries.
    auto it = std::upper_bound(
        segments_.begin(), segments_.end(), s, [](arc_length value, const positioned_segment& entry) { return value < entry.start; });

    if (it == segments_.begin()) [[unlikely]] {
        throw std::runtime_error{"Arc length before path start"};
    }

    --it;

    // Calculate end position: either the start of next segment, or path length
    const arc_length segment_start = it->start;
    const arc_length segment_end = (it + 1 != segments_.end()) ? (it + 1)->start : length_;

    return {it->seg, segment_start, segment_end};
}

xvector<> path::configuration(arc_length s) const {
    return (*this)(s).configuration(s);
}

xvector<> path::tangent(arc_length s) const {
    return (*this)(s).tangent(s);
}

xvector<> path::curvature(arc_length s) const {
    return (*this)(s).curvature(s);
}

// TOOD: This clamping is sort of weird. I think it should either throw, or decay to the singular end cursor.
path::cursor::cursor(const class path* p, arc_length s)
    : path_{p}, position_{std::clamp(s, arc_length{0.0}, p->length())}, hint_{path_->begin()} {
    if (path_->empty()) {
        throw std::invalid_argument{"Cannot create cursor for empty path"};
    }
    update_hint_();
}

const path& path::cursor::path() const noexcept {
    return *path_;
}

arc_length path::cursor::position() const noexcept {
    return position_;
}

path::segment::view path::cursor::operator*() const {
    return *hint_;
}

path::cursor& path::cursor::seek(arc_length s) noexcept {
    // Use +/-infinity sentinels to signal out-of-bounds positions, allowing the cursor to
    // continue operating without throwing exceptions. This is important for algorithms that
    // might overstep slightly and need to detect saturation. The sentinels are detectable
    // via comparison with end().
    if (s < arc_length{0.0}) {
        position_ = arc_length{-std::numeric_limits<double>::infinity()};
    } else if (s > path_->length()) {
        position_ = arc_length{std::numeric_limits<double>::infinity()};
    } else {
        position_ = s;
    }

    if (*this != end()) {
        update_hint_();
    }
    return *this;
}

path::cursor& path::cursor::seek_by(arc_length delta) noexcept {
    return seek(position_ + delta);
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static): Must be non-static for consistent cursor API
std::default_sentinel_t path::cursor::end() const noexcept {
    return std::default_sentinel;
}

path::const_iterator path::cursor::base() const noexcept {
    if (*this == end()) {
        return path_->end();
    }
    return hint_;
}

bool operator==(const path::cursor& c, std::default_sentinel_t) noexcept {
    return !std::isfinite(static_cast<double>(c.position_));
}

bool operator==(std::default_sentinel_t, const path::cursor& c) noexcept {
    return !std::isfinite(static_cast<double>(c.position_));
}

void path::cursor::configuration(std::span<double> out) const {
    if (*this == end()) [[unlikely]] {
        throw std::out_of_range{"Cannot query cursor at sentinel position"};
    }

    auto view = *hint_;
    view.configuration(position_, out);
}

void path::cursor::tangent(std::span<double> out) const {
    if (*this == end()) [[unlikely]] {
        throw std::out_of_range{"Cannot query cursor at sentinel position"};
    }

    auto view = *hint_;
    view.tangent(position_, out);
}

void path::cursor::curvature(std::span<double> out) const {
    if (*this == end()) [[unlikely]] {
        throw std::out_of_range{"Cannot query cursor at sentinel position"};
    }

    auto view = *hint_;
    view.curvature(position_, out);
}

xvector<> path::cursor::configuration() const {
    auto result = xvector<>::from_shape(std::array<std::size_t, 1>{path_->dof()});
    configuration(std::span<double>{result.data(), result.size()});
    return result;
}

xvector<> path::cursor::tangent() const {
    auto result = xvector<>::from_shape(std::array<std::size_t, 1>{path_->dof()});
    tangent(std::span<double>{result.data(), result.size()});
    return result;
}

xvector<> path::cursor::curvature() const {
    auto result = xvector<>::from_shape(std::array<std::size_t, 1>{path_->dof()});
    curvature(std::span<double>{result.data(), result.size()});
    return result;
}

path::cursor::rich path::cursor::enrich() const {
    return rich{*this};
}

path::cursor::rich::rich(cursor c) : cursor{std::move(c)} {
    // Size all three once, here, so that no accessor ever allocates. A rich cursor on a path
    // whose dof is zero is still well-formed, and its accessors fill nothing.
    const std::array<std::size_t, 1> shape{this->path().dof()};

    configuration_ = xvector<>::from_shape(shape);
    tangent_ = xvector<>::from_shape(shape);
    curvature_ = xvector<>::from_shape(shape);
}

path::cursor path::cursor::rich::plain() const {
    return static_cast<const cursor&>(*this);
}

void path::cursor::rich::invalidate_() noexcept {
    // Assigning a fresh bitset rather than calling reset() was worth about 2% of
    // trajectory::create against libc++, where reset() at this size still went through a
    // bit-range fill. Unmeasured on other standard libraries.
    cached_bits_ = decltype(cached_bits_){};
}

// The integration loop seeks to each step's start, which is usually where the previous step's probe
// left the cursor, so a seek that does not move the cursor keeps the cache. The segment needs no
// check of its own, since the hint does not move when the position does not.
path::cursor::rich& path::cursor::rich::seek(arc_length s) noexcept {
    const auto before = position();
    cursor::seek(s);
    if (position() != before) {
        invalidate_();
    }
    return *this;
}

path::cursor::rich& path::cursor::rich::seek_by(arc_length delta) noexcept {
    return this->rich::seek(position() + delta);
}

// A throwing fill leaves the bit clear, so a query that failed is not later mistaken for one that
// succeeded.
template <typename Fill>
const xvector<>& path::cursor::rich::cached_(xvector<>& storage, std::size_t bit, Fill&& fill) const {
    if (!cached_bits_.test(bit)) {
        std::forward<Fill>(fill)(std::span<double>{storage.data(), storage.size()});
        cached_bits_.set(bit);
    }

    return storage;
}

const xvector<>& path::cursor::rich::configuration_ref() const {
    return cached_(configuration_, k_configuration_bit_, [this](std::span<double> out) { cursor::configuration(out); });
}

const xvector<>& path::cursor::rich::tangent_ref() const {
    return cached_(tangent_, k_tangent_bit_, [this](std::span<double> out) { cursor::tangent(out); });
}

const xvector<>& path::cursor::rich::curvature_ref() const {
    return cached_(curvature_, k_curvature_bit_, [this](std::span<double> out) { cursor::curvature(out); });
}

xvector<> path::cursor::rich::configuration() const {
    return configuration_ref();
}

xvector<> path::cursor::rich::tangent() const {
    return tangent_ref();
}

xvector<> path::cursor::rich::curvature() const {
    return curvature_ref();
}

void path::cursor::rich::configuration(std::span<double> out) const {
    const auto& cached = configuration_ref();
    require_dof(out, cached.size());
    std::copy_n(cached.data(), cached.size(), out.data());
}

void path::cursor::rich::tangent(std::span<double> out) const {
    const auto& cached = tangent_ref();
    require_dof(out, cached.size());
    std::copy_n(cached.data(), cached.size(), out.data());
}

void path::cursor::rich::curvature(std::span<double> out) const {
    const auto& cached = curvature_ref();
    require_dof(out, cached.size());
    std::copy_n(cached.data(), cached.size(), out.data());
}

bool operator==(const path::cursor::rich& c, std::default_sentinel_t) noexcept {
    return static_cast<const path::cursor&>(c) == std::default_sentinel;
}

bool operator==(std::default_sentinel_t, const path::cursor::rich& c) noexcept {
    return static_cast<const path::cursor&>(c) == std::default_sentinel;
}

void path::cursor::update_hint_() noexcept {
    // Maintain a hint iterator pointing to the segment containing the current position.
    // This optimization provides O(1) amortized performance for sequential traversal
    // (the common case during trajectory integration) by checking nearby segments before
    // falling back to binary search for large jumps.
    //
    // TODO(RSDK-12770): Optimize hint update by detecting seek direction (forward vs backward from previous
    // position). We can skip checks that won't work based on direction (e.g., don't check prev
    // segment when moving forward), simplifying logic and improving performance for directional
    // sequential access patterns.

    auto current_view = *hint_;

    // Most common case: position is still within the current hint segment.
    if (position_ >= current_view.start() && position_ < current_view.end()) {
        return;
    }

    // Check if we're exactly at a segment boundary, which happens frequently during
    // integration when stepping lands precisely on a segment transition.
    if (position_ == current_view.end()) {
        auto next_hint = hint_;
        ++next_hint;
        if (next_hint != path_->end()) {
            auto next_view = *next_hint;
            if (position_ >= next_view.start() && position_ < next_view.end()) {
                hint_ = next_hint;
                return;
            }
        }
        return;
    }

    // Check adjacent segments (both directions) before resorting to binary search.
    // This handles small steps backward (rare but possible during some algorithms)
    // and forward steps that skip a segment (possible with larger integration steps).

    if (hint_ != path_->begin()) {
        auto prev_hint = hint_;
        --prev_hint;
        auto prev_view = *prev_hint;
        if (position_ >= prev_view.start() && position_ < prev_view.end()) {
            hint_ = prev_hint;
            return;
        }
    }

    if (hint_ != path_->end()) {
        auto next_hint = hint_;
        ++next_hint;
        if (next_hint != path_->end()) {
            auto next_view = *next_hint;
            if (position_ >= next_view.start() && position_ < next_view.end()) {
                hint_ = next_hint;
                return;
            }
        }
    }

    // Position is far from the hint (large jump). Use binary search to find the correct
    // segment. This is O(log n) but should be rare during normal sequential traversal.
    auto seg_it = std::upper_bound(
        path_->segments_.begin(), path_->segments_.end(), position_, [](arc_length value, const positioned_segment& entry) {
            return value < entry.start;
        });

    if (seg_it == path_->segments_.begin()) {
        hint_ = path_->begin();
    } else {
        --seg_it;
        hint_ = const_iterator{path_, seg_it};
    }
}

path::cursor path::create_cursor(arc_length s) const {
    if (empty()) {
        throw std::invalid_argument{"Cannot create cursor for empty path"};
    }
    return cursor{this, s};
}

}  // namespace viam::trajex::totg
