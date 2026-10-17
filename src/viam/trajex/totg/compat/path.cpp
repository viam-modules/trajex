#include <viam/trajex/totg/compat/path.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <viam/trajex/totg/compat/private/path_impl.hpp>
#include <viam/trajex/types/arc_length.hpp>

namespace viam::trajex::totg::compat {

namespace {

xmatrix<> to_xmatrix(const std::list<Eigen::VectorXd>& waypoints) {
    const std::size_t dof = waypoints.empty() ? 0 : static_cast<std::size_t>(waypoints.front().size());
    xmatrix<> result(xmatrix<>::shape_type{waypoints.size(), dof});
    double* row = result.data();
    for (const auto& waypoint : waypoints) {
        if (static_cast<std::size_t>(waypoint.size()) != dof) {
            throw std::invalid_argument{"Path: waypoints differ in dimension"};
        }
        row = std::copy_n(waypoint.data(), waypoint.size(), row);
    }
    return result;
}

// The geometry queries share their handling of failed paths and out of range arguments, and
// differ only in which trajex query they forward to.
template <typename Query>
Eigen::VectorXd evaluate(const std::optional<totg::path>& path, double s, Query query) {
    if (!path || std::isnan(s)) {
        return {};
    }
    const arc_length clamped{std::clamp(s, 0.0, static_cast<double>(path->length()))};
    return to_eigen(query(*path, clamped));
}

}  // namespace

Path::Path(const std::list<Eigen::VectorXd>& path, double maxDeviation) : impl_(new impl) {
    try {
        impl_->path.emplace(totg::path::create(to_xmatrix(path), totg::path::options{}.set_max_blend_deviation(maxDeviation)));
    } catch (...) {
        impl_->error = std::current_exception();
    }
}

Path::Path(const Path& path) : impl_(new impl(*path.impl_)) {}

Path& Path::operator=(const Path& path) {
    Path copy(path);
    std::swap(impl_, copy.impl_);
    return *this;
}

Path::~Path() {
    delete impl_;
}

double Path::getLength() const {
    return impl_->path ? static_cast<double>(impl_->path->length()) : 0.0;
}

Eigen::VectorXd Path::getConfig(double s) const {
    return evaluate(impl_->path, s, [](const totg::path& p, arc_length at) { return p.configuration(at); });
}

Eigen::VectorXd Path::getTangent(double s) const {
    return evaluate(impl_->path, s, [](const totg::path& p, arc_length at) { return p.tangent(at); });
}

Eigen::VectorXd Path::getCurvature(double s) const {
    return evaluate(impl_->path, s, [](const totg::path& p, arc_length at) { return p.curvature(at); });
}

double Path::getNextSwitchingPoint(double, bool& discontinuity) const {
    discontinuity = true;
    return getLength();
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static): Must be a const member to match the original Path
std::list<std::pair<double, bool>> Path::getSwitchingPoints() const {
    return {};
}

std::exception_ptr Path::getError() const {
    return impl_->error;
}

}  // namespace viam::trajex::totg::compat
