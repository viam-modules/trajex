#pragma once

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

namespace viam::trajex::jacobian {

///
/// Computes the geometric Jacobian for a URDF-style model table at joint
/// positions q.
///
/// Supports only revolute and fixed joints.
///
/// @param model_table (n, 10) tensor in the viam::sdk::ModelTable format
/// @param q (N_actuated,) vector with one element per revolute row in the
///        table, in chain order. Fixed rows do not consume a q entry.
/// @return A (6, N_actuated) xarray where rows 0..2 are the linear-velocity
///         columns J_v_i = w_i x (p_e - p_i) and rows 3..5 are the
///         angular-velocity columns J_w_i = w_i, with w_i the world-frame axis
///         of revolute joint i, p_i its world position, and p_e the
///         end-effector position.
/// @throws std::invalid_argument on malformed tensor shape, invalid joint-type
///         encoding, q-size mismatch, unsupported joint type (continuous or
///         prismatic), or revolute row with zero-magnitude axis
///
xt::xarray<double> compute_jacobian(const xt::xarray<double>& model_table, const xt::xarray<double>& q);

}  // namespace viam::trajex::jacobian
