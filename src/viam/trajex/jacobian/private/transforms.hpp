#pragma once

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

namespace viam::trajex::jacobian::detail {

// Standard (distal) DH transform: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha).
// Returns a 4x4 xtensor array.
xt::xarray<double> dh_transform(double d, double theta, double a, double alpha);

// In-place 4x4 matrix multiply: out = A * B. The three arguments must be
// distinct buffers (no aliasing). xtensor lacks a built-in matmul without
// xtensor-blas, and a hand-rolled 4x4 loop is faster than xt::sum-based
// expressions for this fixed shape.
void matmul_4x4(const xt::xarray<double>& A,
                const xt::xarray<double>& B,
                xt::xarray<double>& out);

}  // namespace viam::trajex::jacobian::detail
