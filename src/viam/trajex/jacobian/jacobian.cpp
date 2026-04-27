#include <viam/trajex/jacobian/jacobian.hpp>

#include <array>
#include <cassert>
#include <cstddef>

namespace viam::trajex::jacobian {

namespace {

// 3-element cross product written into J(0..2, col).
inline void write_cross_into_J_col(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b,
    xt::xarray<double>& J,
    std::size_t col) {
    J(0, col) = a[1] * b[2] - a[2] * b[1];
    J(1, col) = a[2] * b[0] - a[0] * b[2];
    J(2, col) = a[0] * b[1] - a[1] * b[0];
}

}  // namespace

void compute_jacobian(const model& m, data& d) {
    assert(d.fk_computed && "compute_jacobian: FK must be computed first "
                            "(call compute_forward_kinematics or the 3-argument overload)");
    const auto& T_e = d.end_effector_transform;
    const std::array<double, 3> p_e{T_e(0, 3), T_e(1, 3), T_e(2, 3)};

    for (std::size_t i = 0; i < m.joints.size(); ++i) {
        const auto& T = d.joint_transforms[i];
        // Standard DH: joint i rotates about Z of frame i-1. The 3rd column
        // of the stored rotation matrix is local Z expressed in the base frame.
        const std::array<double, 3> z{T(0, 2), T(1, 2), T(2, 2)};
        const std::array<double, 3> p{T(0, 3), T(1, 3), T(2, 3)};
        const std::array<double, 3> r{p_e[0] - p[0], p_e[1] - p[1], p_e[2] - p[2]};

        write_cross_into_J_col(z, r, d.J, i);
        d.J(3, i) = z[0];
        d.J(4, i) = z[1];
        d.J(5, i) = z[2];
    }
}

void compute_jacobian(const model& m, const xt::xarray<double>& q, data& d) {
    compute_forward_kinematics(m, q, d);
    compute_jacobian(m, d);
}

}  // namespace viam::trajex::jacobian
