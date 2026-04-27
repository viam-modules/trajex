#pragma once

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/jacobian/model.hpp>

namespace viam::trajex::jacobian {

// Compute forward kinematics for N-DOF robot.
// q must have model.joints.size() elements.
// Writes joint_transforms and end_effector_transform into data.
void compute_forward_kinematics(const model& m, const xt::xarray<double>& q, data& d);

}  // namespace viam::trajex::jacobian
