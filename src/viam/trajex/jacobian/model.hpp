#pragma once

#include <string>
#include <vector>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

namespace viam::trajex::jacobian {

// One row of a standard (distal) Denavit-Hartenberg table, for a single
// revolute joint. Per-joint transform:
//   T_{i-1,i}(q) = Rz(theta + q) * Tz(d) * Tx(a) * Rx(alpha)
// theta is the zero-config offset; the runtime joint angle is theta + q.
// Units: metres for d and a, radians for theta and alpha.
struct dh_joint {
    std::string name;
    double d;       // translation along Z_{i-1}
    double theta;   // zero-config rotation about Z_{i-1}
    double a;       // translation along X_i
    double alpha;   // rotation about X_i
};

struct model {
    std::string name;
    std::vector<dh_joint> joints;  // one per revolute joint; N = joints.size()
};

struct data {
    // Base-frame transform to frame i-1 for each joint i (each is 4x4). The
    // Jacobian reads z_{i-1} and p_{i-1} from these.
    std::vector<xt::xarray<double>> joint_transforms;
    xt::xarray<double> end_effector_transform;  // 4x4
    xt::xarray<double> J;                       // 6 x N
    bool fk_computed = false;

    explicit data(const model& m)
        : joint_transforms(m.joints.size(), xt::eye<double>(4))
        , end_effector_transform(xt::eye<double>(4))
        , J(xt::zeros<double>({std::size_t{6}, m.joints.size()}))
    {}
};

}  // namespace viam::trajex::jacobian
