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

namespace {

void require_dof(std::span<const double> out, std::size_t dof) {
    if (out.size() != dof) [[unlikely]] {
        throw std::invalid_argument{"Output span size does not match path degrees of freedom"};
    }
}

// Whether a representable arc length lies strictly between a and b, in either order. Step one value
// from a toward b: if that falls short of b, a value lies between. Path construction asks before it
// builds a segment, since a segment with no length cannot be built at all.
bool has_interior(arc_length a, arc_length b) noexcept {
    const auto from = static_cast<double>(a);
    const auto to = static_cast<double>(b);
    return (from != to) && (std::nextafter(from, to) != to);
}

}  // namespace

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

bool path::segment::view::has_interior() const noexcept {
    return ::viam::trajex::totg::has_interior(start_, end_);
}

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
        [local_s, length = length(), out](const auto& seg_data) {
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
                // that theta runs over [-angle / 2, angle / 2]. The arc's span on the path is
                // radius * angle rounded into the path's arc length, so theta is taken in proportion to
                // that span rather than as local_s / radius: both ends then land exactly on
                // +-angle / 2, where the rounding divided by a small radius would turn the tangent
                // away from its neighbour's. The 1 - cos(theta) term is written as 2 sin^2(theta / 2),
                // which keeps its digits when theta is small.
                const auto theta = seg_data.angle_rads * ((local_s / length) - 0.5);
                const auto sin_half_theta = std::sin(theta / 2.0);
                const auto along_t = seg_data.radius * std::sin(theta);
                const auto along_n = seg_data.radius * 2.0 * sin_half_theta * sin_half_theta;

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
        [local_s, length = length(), out](const auto& seg_data) {
            using T = std::decay_t<decltype(seg_data)>;

            if constexpr (std::is_same_v<T, segment::linear>) {
                require_dof(out, seg_data.start.shape(0));

                // Linear segment: constant unit tangent
                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = seg_data.unit_direction(i);
                }
            } else if constexpr (std::is_same_v<T, segment::circular>) {
                require_dof(out, seg_data.apex.shape(0));

                // Circular arc unit tangent - Kunz & Stilman equation 8, with theta as for the configuration:
                const auto theta = seg_data.angle_rads * ((local_s / length) - 0.5);
                const auto cos_theta = std::cos(theta);
                const auto sin_theta = std::sin(theta);

                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = (seg_data.t(i) * cos_theta) + (seg_data.n(i) * sin_theta);
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
        [local_s, length = length(), out](const auto& seg_data) {
            using T = std::decay_t<decltype(seg_data)>;

            if constexpr (std::is_same_v<T, segment::linear>) {
                require_dof(out, seg_data.start.shape(0));

                // Linear segment: zero curvature vector
                std::ranges::fill(out, 0.0);
            } else if constexpr (std::is_same_v<T, segment::circular>) {
                require_dof(out, seg_data.apex.shape(0));

                // Circular arc curvature vector - Kunz & Stilman equation 9, with theta as for the configuration:
                const auto theta = seg_data.angle_rads * ((local_s / length) - 0.5);
                const auto cos_theta = std::cos(theta);
                const auto sin_theta = std::sin(theta);
                const auto inverse_radius = 1.0 / seg_data.radius;

                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = inverse_radius * ((seg_data.n(i) * cos_theta) - (seg_data.t(i) * sin_theta));
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

                // The joint's component of the tangent, cos(theta) t_i + sin(theta) n_i, is zero where
                // tan(theta) = -t_i / n_i, which within (-pi/2, pi/2) has the single root atan(-t_i / n_i).
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
                const auto theta = std::atan(-t_i / n_i);

                // Report the root wherever it falls, on the arc or on the circle beyond it. Whether one
                // just off the arc should count is the caller's to decide, and it needs to know how far
                // off it is to do that.
                return start_ + (length() * ((theta / seg_data.angle_rads) + 0.5));
            }
        },
        seg_.get().data_);
}

arc_length path::segment::view::nearest(std::span<const double> configuration) const {
    return std::visit(
        [this, configuration](const auto& seg_data) -> arc_length {
            using T = std::decay_t<decltype(seg_data)>;

            if constexpr (std::is_same_v<T, segment::linear>) {
                require_dof(configuration, seg_data.start.shape(0));

                // Project onto the line, and clamp to the segment.
                double along = 0.0;
                for (std::size_t i = 0; i < configuration.size(); ++i) {
                    along += (configuration[i] - seg_data.start(i)) * seg_data.unit_direction(i);
                }
                return std::clamp(start_ + arc_length{along}, start_, end_);
            } else if constexpr (std::is_same_v<T, segment::circular>) {
                require_dof(configuration, seg_data.apex.shape(0));

                // The arc lies on a circle about apex + radius * n, and the configuration's direction
                // from that center within the arc's plane gives the angle of the nearest point of the
                // circle. The arc turns by less than pi, so clamping that angle to the arc gives the
                // nearest point of the arc. Forming the center would cost about radius * eps at a capped
                // radius, so the two in-plane components are taken relative to the apex instead.
                double along_t = 0.0;
                double along_n = 0.0;
                for (std::size_t i = 0; i < configuration.size(); ++i) {
                    const auto offset = configuration[i] - seg_data.apex(i);
                    along_t += offset * seg_data.t(i);
                    along_n += offset * seg_data.n(i);
                }
                const auto half_angle = seg_data.angle_rads / 2.0;
                const auto theta = std::clamp(std::atan2(along_t, seg_data.radius - along_n), -half_angle, half_angle);
                return std::clamp(start_ + (length() * ((theta / seg_data.angle_rads) + 0.5)), start_, end_);
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

    // Build a path from waypoints in two passes. The first, colinearization, drops waypoints that lie
    // within a tolerance tube of a straight run through their neighbours, so a run of nearly straight
    // waypoints becomes one leg. The second walks the waypoints that remain, leg by leg: each corner
    // between two legs either takes a circular blend, tangent to both legs and trimming each by the
    // same distance, or is left as a hard corner, and the stretch of each leg that its blends leave
    // behind is emitted as a line. Both passes keep the path within the deviation budgets in the
    // options.

    // The arc a corner would get, as raw geometry rather than a constructed segment. The loop decides
    // whether the arc has room on the path, and only then builds the one segment::circular, with its
    // radius refit to that room, so the arc is constructed and validated once, on exactly what is stored.
    struct blend_geometry {
        xvector<> apex;          // Point of the arc nearest the corner
        xvector<> apex_tangent;  // Unit tangent at the apex
        xvector<> apex_normal;   // Unit normal at the apex, toward the center
        double radius;           // Natural radius, trim / tan(alpha/2), before the refit
        double angle_rads;
        double trim_distance;      // Distance trimmed from each leg at the corner
        bool takes_half_incoming;  // trim_distance is exactly half of the incoming segment
        bool takes_half_outgoing;  // trim_distance is exactly half of the outgoing segment
    };

    // A leg is the straight run from one waypoint to the next. Its direction and length are measured
    // once, from the two waypoints, and everything built on the leg is built from that one measurement:
    // the blend at either end takes its direction and caps its trim at half the length, and the line
    // left between the blends runs along the same direction. A direction rebuilt from a blend's exit
    // point would not do, because the exit point is rounded, and over the half leg or so left to the
    // locus that rounding would turn the rebuilt direction by up to eps |q| over that distance.
    struct leg {
        xvector<> unit;
        double length;

        // Whether the blend at the start of the leg took exactly half of it. False when that end of
        // the leg is a hard waypoint rather than a blend.
        bool half_taken_at_start;
    };

    // A zero-length leg has no direction, and nothing built on it could have one either, so it is
    // refused here, where the cause can be named, rather than where a segment would later reject it.
    const auto measure_leg = [](leg& measured, const auto& from, const auto& to) {
        xt::noalias(measured.unit) = to - from;
        measured.length = xt::norm_l2(measured.unit)();
        if (measured.length == 0.0) {
            throw std::invalid_argument{"Path requires consecutive waypoints to be distinct"};
        }
        measured.unit /= measured.length;
    };

    // Try to create a circular blend arc at a corner following the algorithm from
    // Kunz & Stilman Section IV. The blend arc is tangent to both the incoming and
    // outgoing segments, trimming equal distances from each side. The blend keeps
    // the path within max_blend_deviation of the original corner waypoint.
    const auto try_create_blend = [&opts](const auto& corner, const leg& incoming, const leg& outgoing) -> std::optional<blend_geometry> {
        if (opts.max_blend_deviation() <= 0.0) {
            return std::nullopt;
        }

        // The two unit directions span a rhombus, and its diagonals carry everything the blend needs.
        // Their sum lies along the arc's tangent at its apex, and their difference along its normal
        // there, toward the inside of the corner. For a turn by alpha their lengths are
        // 2 cos(alpha/2) and 2 sin(alpha/2), each good to about eps absolute.
        xvector<> apex_tangent = incoming.unit + outgoing.unit;
        xvector<> apex_normal = outgoing.unit - incoming.unit;
        const auto apex_tangent_length = xt::norm_l2(apex_tangent)();
        const auto apex_normal_length = xt::norm_l2(apex_normal)();

        // Make the two diagonals an orthonormal frame. A rhombus's diagonals cross at right angles,
        // because its sides are equal, so in exact arithmetic these two are already perpendicular. And
        // because its sides here are unit, the diagonals share a fixed budget of length between them:
        // as one shrinks the other grows, and the longer is never shorter than sqrt(2). So the longer
        // is always well measured. The shorter is not. It is what is left when two nearly equal
        // directions cancel, at a shallow corner, or two nearly opposite ones do, at a near reversal,
        // and the rounding in that cancellation is about the same size whatever is left; when little is
        // left, its direction is mostly rounding, and no longer perpendicular to the longer in fact.
        // So trust the longer: make it unit, and keep only the part of the shorter perpendicular to
        // it, which is the part the geometry guarantees. What rounding the shorter still carries can
        // only tilt the plane the arc lies in, and the arc's ends feel that tilt scaled by the shorter's
        // length, which is small exactly when the tilt is large, so they meet the incoming and outgoing
        // directions to rounding at any angle.
        auto [longer, shorter] =
            (apex_normal_length > apex_tangent_length) ? std::tie(apex_normal, apex_tangent) : std::tie(apex_tangent, apex_normal);
        longer /= std::max(apex_tangent_length, apex_normal_length);
        xt::noalias(shorter) -= xt::sum(shorter * longer)() * longer;
        const auto shorter_length = xt::norm_l2(shorter)();

        // Nothing perpendicular remains only when the directions are parallel, the same or opposite,
        // to rounding: equal directions, an exact reversal, or directions on one line whose
        // normalizations rounded differently. Such a corner has no plane to put an arc in. Where the
        // directions are the same it does not turn, and needs no blend, and TOTG won't mandate
        // s_dot = 0 there; where they are opposite no arc is defined. These are the only exclusions:
        // a corner turning by any amount between, however near 0 or pi, is left to the curvature
        // bounds below. The dot product of the directions cannot draw this line, since it is
        // 1 - alpha^2 / 2, which rounds to exactly 1 for turns below about 1e-8.
        if (shorter_length == 0.0) {
            return std::nullopt;
        }
        shorter /= shorter_length;

        // Kunz & Stilman equation 2 takes the angle as the arccosine of the dot product, which has
        // no digits near 0 or pi: a corner turning by 1.5e-13 reads as its floor of 1.5e-8
        // (see RSDK-14911). The diagonals give it to rounding at any angle, and the tangents of its half
        // and quarter as plain ratios, the quarter by tan(alpha/4) = sin(alpha/2) / (1 + cos(alpha/2)).
        // Neither length is zero, so the angle is positive. It can round to pi, which
        // segment::circular rejects, only where the tangent diagonal is below about eps of the normal
        // diagonal, and there tan(alpha/2) exceeds about 1/eps, which the near-reversal check below
        // rejects for any max_blend_curvature short of 1 / (eps * trim).
        const auto angle = 2.0 * std::atan2(apex_normal_length, apex_tangent_length);
        const auto tan_half = apex_normal_length / apex_tangent_length;
        const auto tan_quarter = apex_normal_length / (2.0 + apex_tangent_length);

        // Compute trim following Kunz & Stilman equations 3-4. Three constraints apply: can't
        // trim more than half of either segment, and the deviation at the blend must stay within
        // the budget. The deviation formula inverted is: trim = deviation / tan(alpha/4). This
        // form (via the half-angle identity sin(x)/(1-cos(x)) = cot(x/2)) is well-conditioned
        // at small angles, unlike the equivalent sin(alpha/2)/(1-cos(alpha/2)) expression.
        const auto max_trim_from_segments = std::min(incoming.length / 2.0, outgoing.length / 2.0);
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

        // Near-collinear: cap trim at tan(alpha/2)/kmin, which makes the radius 1/kmin to within the
        // rounding of the two divisions. Between the two curvature checks the radius lies in
        // [1/kmax, 1/kmin], so dividing by tan(alpha/2) here can neither blow up nor vanish.
        const auto trim_distance = std::min(natural_trim, tan_half / opts.min_blend_curvature());
        const auto radius = trim_distance / tan_half;

        // Construct the arc from its apex rather than from its center as Kunz & Stilman equations
        // 5-6 do. The paper puts the center on the bisector at radius / cos(alpha/2) from the corner,
        // but the bisector is the difference of two unit vectors, which at a shallow corner are
        // nearly equal, so its direction has few correct digits, and at the min_blend_curvature
        // radius cap that error carries the center, and the whole arc with it, far off the path
        // (see RSDK-14911). Everything about the apex is bounded by the arc's length instead.
        //
        // The apex is the point of the arc nearest the corner, at the blend's deviation from it,
        // which is trim * tan(alpha/4) by the relation the trim was computed from above. This needs
        // no cosine of the half angle, which vanishes at a reversal.
        xvector<> apex = corner + ((trim_distance * tan_quarter) * apex_normal);

        // Report which halves the trim took, by the same expressions that capped it. Halving is exact,
        // so these are equalities between identically computed values, not tolerances. The construction
        // loop uses them to recognize a segment that two blends consume completely.
        return blend_geometry{.apex = std::move(apex),
                              .apex_tangent = std::move(apex_tangent),
                              .apex_normal = std::move(apex_normal),
                              .radius = radius,
                              .angle_rads = angle,
                              .trim_distance = trim_distance,
                              .takes_half_incoming = (trim_distance == incoming.length / 2.0),
                              .takes_half_outgoing = (trim_distance == outgoing.length / 2.0)};
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
            xvector<> start_to_locus;  // Scratch for the locus test, reused across calls
        };
        segment_deviation_context segment_ctx{xvector<>::from_shape(std::array<std::size_t, 1>{waypoints.dof()}),
                                              0.0,
                                              xvector<>::from_shape(std::array<std::size_t, 1>{waypoints.dof()})};

        // Deviation is measured against a tube of this radius around the segment. Hoisted
        // because the enclosing branch has already established that the tolerance is non-zero.
        // The test below compares squared lengths, so the radius is squared once here and no
        // square root is taken per locus.
        const auto radius = opts.max_linear_deviation() / 2.0;
        const auto radius_sq = radius * radius;

        // Establishes the segment that loci are tested against. Returns false when start and
        // next are the same position, because the locus is then an intentional intermediate
        // goal rather than a point along a line, and nothing may be coalesced through it.
        //
        // The squared length is computed first and stands in for the elementwise check on every
        // real segment: a nonzero sum of squares cannot come from a zero vector. A zero sum can,
        // or can come from components so small their squares underflow, so only then does the
        // elementwise check decide.
        const auto begin_segment = [&segment_ctx](const auto& start, const auto& next) -> bool {
            xt::noalias(segment_ctx.start_to_next) = next - start;
            segment_ctx.length_sq = xt::sum(segment_ctx.start_to_next * segment_ctx.start_to_next)();

            if (segment_ctx.length_sq != 0.0) {
                return true;
            }
            return !xt::all(xt::equal(segment_ctx.start_to_next, 0.0));
        };

        const auto locus_within_tolerance = [&segment_ctx, radius_sq](const auto& start, const auto& locus) -> bool {
            xt::noalias(segment_ctx.start_to_locus) = locus - start;
            const auto start_to_locus_dot_direction = xt::sum(segment_ctx.start_to_locus * segment_ctx.start_to_next)();

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
            const auto deviation_vector = segment_ctx.start_to_locus - (t * segment_ctx.start_to_next);
            const auto deviation_sq = xt::norm_sq(deviation_vector)();

            return deviation_sq <= radius_sq;
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

    // Each locus emits at most a line and an arc, so this bounds the segment count. Real paths fill
    // about nine tenths of it, and the path keeps the slack, in exchange for never moving a segment
    // to grow the vector.
    std::vector<positioned_segment> segments;
    segments.reserve(2 * (std::ranges::size(waypoints_range) - 1));
    arc_length cumulative_length{0.0};

    // Track the current position on the path we're building. This must be stored as an actual
    // configuration copy (not just an iterator) because after creating a circular blend, the
    // current position becomes the blend exit point, which is a computed position between
    // waypoints rather than one of the original waypoints in the accumulator.
    xvector<> current_position = *waypoints_range.begin();

    // The leg into the locus and the leg out of it. Each leg is measured once, looking ahead from
    // the locus it leaves, and is carried over to be the incoming leg of the locus it reaches.
    leg incoming{xvector<>::from_shape(std::array<std::size_t, 1>{waypoints.dof()}), 0.0, false};
    leg outgoing{xvector<>::from_shape(std::array<std::size_t, 1>{waypoints.dof()}), 0.0, false};
    measure_leg(outgoing, *waypoints_range.begin(), *std::next(waypoints_range.begin()));

    for (auto locus = std::next(waypoints_range.begin()); locus != waypoints_range.end(); ++locus) {
        auto next = std::next(locus);

        // The leg measured ahead of the previous locus is the leg into this one. If there is a leg
        // beyond this locus, measure it, and see whether the corner between the two takes a blend.
        std::swap(incoming, outgoing);
        std::optional<blend_geometry> blend_opt;
        if (next != waypoints_range.end()) {
            measure_leg(outgoing, *locus, *next);
            blend_opt = try_create_blend(*locus, incoming, outgoing);
        }

        // The line into the locus starts at current_position and runs along the incoming leg. Its
        // length is a scalar, which carries the rounding of current_position only absolutely.
        const auto dist_to_locus = xt::norm_l2(*locus - current_position)();

        if (blend_opt) {
            auto& blend = *blend_opt;

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
            const bool segment_fully_consumed = incoming.half_taken_at_start && blend.takes_half_incoming;

            // A segment is only worth emitting if there is somewhere to be inside it. Positions
            // are addressed in the path's own arc length, so that means a representable value
            // strictly between the endpoints. One ULP leaves nowhere to stand between the endpoints
            // and no way to advance through the segment; two leaves exactly one position, which
            // is the least that makes the segment a place rather than a boundary. This asks the
            // number system rather than a tolerance, so it introduces no epsilon. The length must
            // also be positive: the same rebuilt distance can fall a few ULPs short of the trim.
            const auto incoming_line_length = dist_to_locus - blend.trim_distance;
            const bool has_incoming_line = !segment_fully_consumed && (incoming_line_length > 0.0) &&
                                           has_interior(cumulative_length, cumulative_length + arc_length{incoming_line_length});

            // The arc's span on the path is radius * angle rounded into the path's arc length, which
            // is the length it will be traversed over. An arc needs somewhere to be inside it just as a
            // line does, and a corner that turns by little, far along the path, can have an arc whose
            // span rounds to nothing. Such a corner is left unblended, below.
            auto arc_start = cumulative_length;
            if (has_incoming_line) {
                arc_start += arc_length{incoming_line_length};
            }
            const auto arc_end = arc_start + arc_length{blend.radius * blend.angle_rads};
            if (has_interior(arc_start, arc_end)) {
                if (has_incoming_line) {
                    segments.push_back({.seg = segment{segment::linear{current_position, incoming.unit, arc_length{incoming_line_length}}},
                                        .start = cumulative_length});
                }

                // Refit the radius to the span, so the arc is traversed at unit speed and still turns
                // by exactly its angle. Its ends move by about the rounding, in position only; the
                // tangent at either end depends on the angle alone.
                //
                // This is the arc's only construction, so the geometry is validated here and not before
                // the interior check above. Nothing is lost by the order: the frame can only fail the
                // constructor's checks when the corner's legs were not finite, and then the radius or
                // the angle is NaN, the span is NaN, and has_interior, whose comparisons are all true
                // for NaN, brings us here to throw just as a construction before the check would have.
                const auto span = static_cast<double>(arc_end - arc_start);
                segments.push_back({.seg = segment{segment::circular{std::move(blend.apex),
                                                                     std::move(blend.apex_tangent),
                                                                     std::move(blend.apex_normal),
                                                                     span / blend.angle_rads,
                                                                     blend.angle_rads}},
                                    .start = arc_start});
                cumulative_length = arc_end;

                // After the blend, we're positioned at the blend exit point on the outgoing segment.
                // This position doesn't correspond to any waypoint in the accumulator, which is why
                // current_position must be a configuration copy rather than an iterator.
                xt::noalias(current_position) = *locus + (blend.trim_distance * outgoing.unit);
                outgoing.half_taken_at_start = blend.takes_half_outgoing;
                continue;
            }
        }

        // No blend, or none with room on the path: emit the line from current_position to the locus,
        // and stop there.
        segments.push_back(
            {.seg = segment{segment::linear{current_position, incoming.unit, arc_length{dist_to_locus}}}, .start = cumulative_length});
        cumulative_length += arc_length{dist_to_locus};

        xt::noalias(current_position) = *locus;
        outgoing.half_taken_at_start = false;
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
