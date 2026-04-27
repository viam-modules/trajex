#include <viam/trajex/jacobian/forward_kinematics.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#include <viam/trajex/jacobian/private/transforms.hpp>

namespace viam::trajex::jacobian {

namespace detail {

xt::xarray<double> dh_transform(double d, double theta, double a, double alpha) {
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);

    return xt::xarray<double>{
        { ct, -st * ca,  st * sa,  a * ct },
        { st,  ct * ca, -ct * sa,  a * st },
        {0.0,       sa,       ca,       d },
        {0.0,      0.0,      0.0,     1.0 },
    };
}

void matmul_4x4(const xt::xarray<double>& A,
                const xt::xarray<double>& B,
                xt::xarray<double>& out) {
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                s += A(i, k) * B(k, j);
            }
            out(i, j) = s;
        }
    }
}

}  // namespace detail

void compute_forward_kinematics(const model& m, const xt::xarray<double>& q, data& d) {
    if (q.size() != m.joints.size()) {
        throw std::invalid_argument(
            "q size mismatch: expected " + std::to_string(m.joints.size())
            + ", got " + std::to_string(q.size()));
    }

    xt::xarray<double> T = xt::eye<double>(4);
    xt::xarray<double> next = xt::zeros<double>({std::size_t{4}, std::size_t{4}});
    for (std::size_t i = 0; i < m.joints.size(); ++i) {
        d.joint_transforms[i] = T;
        const auto& j = m.joints[i];
        const auto step = detail::dh_transform(j.d, j.theta + q(i), j.a, j.alpha);
        detail::matmul_4x4(T, step, next);
        T = next;
    }
    d.end_effector_transform = T;
    d.fk_computed = true;
}

}  // namespace viam::trajex::jacobian
