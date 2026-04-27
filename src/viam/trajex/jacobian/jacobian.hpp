#pragma once

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/jacobian/forward_kinematics.hpp>
#include <viam/trajex/jacobian/model.hpp>

namespace viam::trajex::jacobian {

// Compute full Jacobian (6 x N, where N = model.joints.size()):
// [linear velocity; angular velocity].
// Requires FK to have been computed first (data.joint_transforms and
// data.end_effector_transform must be valid). Writes result into data.J.
void compute_jacobian(const model& m, data& d);

// Convenience: compute both FK and Jacobian.
// q must have model.joints.size() elements.
void compute_jacobian(const model& m, const xt::xarray<double>& q, data& d);

}  // namespace viam::trajex::jacobian
