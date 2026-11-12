// Test utility function implementations
#include <viam/trajex/totg/test/test_utils.hpp>

#include <cmath>
#include <limits>
#include <string>

#include <viam/trajex/totg/path.hpp>
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

}  // namespace viam::trajex::totg::test
