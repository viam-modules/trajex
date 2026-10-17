#include <viam/trajex/totg/compat/trajectory.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <optional>
#include <utility>

#include <viam/trajex/totg/compat/private/path_impl.hpp>
#include <viam/trajex/totg/trajectory.hpp>

namespace viam::trajex::totg::compat {

// Exactly one of the two is engaged.
struct Trajectory::impl {
    std::optional<totg::trajectory> trajectory;
    std::exception_ptr error;
};

namespace {

// The time queries share their handling of failed trajectories and out of range arguments, and
// differ only in which part of the sample they return. The selected part may be a reference
// into the sample. That's safe, because the sample is a temporary that lives until the end of
// the full expression, after the conversion.
template <typename Select>
Eigen::VectorXd evaluate(const std::optional<totg::trajectory>& trajectory, double time, Select select) {
    if (!trajectory || std::isnan(time)) {
        return {};
    }
    const totg::trajectory::seconds clamped{std::clamp(time, 0.0, trajectory->duration().count())};
    return to_eigen(select(trajectory->sample(clamped)));
}

}  // namespace

Trajectory::Trajectory(const Path& path, const Eigen::VectorXd& maxVelocity, const Eigen::VectorXd& maxAcceleration, double timeStep)
    : impl_(new impl) {
    if (!path.impl_->path) {
        impl_->error = path.impl_->error;
        return;
    }

    try {
        totg::trajectory::options options;
        options.max_velocity = to_xvector(maxVelocity);
        options.max_acceleration = to_xvector(maxAcceleration);
        options.delta = totg::trajectory::seconds{timeStep};
        impl_->trajectory.emplace(totg::trajectory::create(*path.impl_->path, std::move(options)));
    } catch (...) {
        impl_->error = std::current_exception();
    }
}

Trajectory::Trajectory(const Trajectory& trajectory) : impl_(new impl(*trajectory.impl_)) {}

Trajectory& Trajectory::operator=(const Trajectory& trajectory) {
    Trajectory copy(trajectory);
    std::swap(impl_, copy.impl_);
    return *this;
}

Trajectory::~Trajectory() {
    delete impl_;
}

bool Trajectory::isValid() const {
    return impl_->trajectory.has_value();
}

double Trajectory::getDuration() const {
    return impl_->trajectory ? impl_->trajectory->duration().count() : 0.0;
}

Eigen::VectorXd Trajectory::getPosition(double time) const {
    return evaluate(impl_->trajectory, time, [](const auto& sample) -> const xvector<>& { return sample.configuration; });
}

Eigen::VectorXd Trajectory::getVelocity(double time) const {
    return evaluate(impl_->trajectory, time, [](const auto& sample) -> const xvector<>& { return sample.velocity; });
}

void Trajectory::outputPhasePlaneTrajectory() const {}

std::exception_ptr Trajectory::getError() const {
    return impl_->error;
}

}  // namespace viam::trajex::totg::compat
