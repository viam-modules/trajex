#include <viam/trajex/jacobian/forward_kinematics.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#include <viam/trajex/jacobian/private/transforms.hpp>

namespace viam::trajex::jacobian {

namespace detail {

Eigen::Matrix4d dh_transform(double d, double theta, double a, double alpha) {
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);

    Eigen::Matrix4d T;
    T << ct, -st * ca,  st * sa,  a * ct,
         st,  ct * ca, -ct * sa,  a * st,
        0.0,       sa,       ca,       d,
        0.0,      0.0,      0.0,     1.0;
    return T;
}

}  // namespace detail

void compute_forward_kinematics(const model& m, const Eigen::VectorXd& q, data& d) {
    if (static_cast<size_t>(q.size()) != m.joints.size()) {
        throw std::invalid_argument(
            "q size mismatch: expected " + std::to_string(m.joints.size())
            + ", got " + std::to_string(q.size()));
    }

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (size_t i = 0; i < m.joints.size(); ++i) {
        d.joint_transforms[i] = T;
        const auto& j = m.joints[i];
        T = T * detail::dh_transform(j.d, j.theta + q[static_cast<Eigen::Index>(i)], j.a, j.alpha);
    }
    d.end_effector_transform = T;
    d.fk_computed = true;
}

}  // namespace viam::trajex::jacobian
