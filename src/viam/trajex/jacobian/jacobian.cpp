#include <viam/trajex/jacobian/jacobian.hpp>

#include <cassert>
#include <cstddef>

#include <Eigen/Geometry>

namespace viam::trajex::jacobian {

void compute_jacobian(const model& m, data& d) {
    assert(d.fk_computed && "compute_jacobian: FK must be computed first "
                            "(call compute_forward_kinematics or the 3-argument overload)");
    const Eigen::Vector3d p_e = d.end_effector_transform.block<3, 1>(0, 3);

    for (size_t i = 0; i < m.joints.size(); ++i) {
        const Eigen::Matrix4d& T = d.joint_transforms[i];
        // Standard DH: joint i rotates about Z of frame i-1. The 3rd column
        // of the stored rotation matrix is local Z expressed in the base frame.
        const Eigen::Vector3d z = T.block<3, 1>(0, 2);
        const Eigen::Vector3d p = T.block<3, 1>(0, 3);

        const Eigen::Index col = static_cast<Eigen::Index>(i);
        d.J.block<3, 1>(0, col) = z.cross(p_e - p);
        d.J.block<3, 1>(3, col) = z;
    }
}

void compute_jacobian(const model& m, const Eigen::VectorXd& q, data& d) {
    compute_forward_kinematics(m, q, d);
    compute_jacobian(m, d);
}

}  // namespace viam::trajex::jacobian
