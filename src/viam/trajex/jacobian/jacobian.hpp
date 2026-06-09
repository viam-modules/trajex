#pragma once

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

namespace viam::trajex::jacobian {

// Compute the geometric Jacobian for a URDF-style model table at joint
// positions q.
//
// model_table: (n, 10) tensor in the viam::sdk::ModelTable format.
// q:           (N_actuated,) vector. One element per revolute row in the
//              table, in chain order. Fixed rows do not consume a q entry.
//
// Returns a (6, N_actuated) xarray:
//   rows 0..2: linear-velocity columns J_v_i = w_i x (p_e - p_i)
//   rows 3..5: angular-velocity columns J_w_i = w_i
// where w_i is the world-frame axis of revolute joint i, p_i is its world
// position, and p_e is the end-effector position.
//
// Supports only revolute and fixed joints. Throws:
//   - std::invalid_argument on q-size mismatch, unsupported joint type
//     (continuous or prismatic), or revolute row with zero-magnitude axis.
//   - viam::sdk::Exception on malformed tensor shape or invalid joint-type
//     encoding (propagated from sdk::ModelTable::from).
xt::xarray<double> compute_jacobian(const xt::xarray<double>& model_table, const xt::xarray<double>& q);

}  // namespace viam::trajex::jacobian
