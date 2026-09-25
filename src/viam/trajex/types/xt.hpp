#pragma once

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#else
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#endif

// The array types trajex holds geometry in. They are named apart from their definitions so
// that the definitions can change in one place: what a configuration is stored in is a
// decision about the whole codebase, not one taken separately by every declaration that
// mentions one.
//
// Extent is dynamic throughout, because degrees of freedom is a runtime property.

namespace viam::trajex {

///
/// One-dimensional array: a configuration, or a quantity shaped like one.
///
template <typename T = double>
using xvector = xt::xtensor<T, 1>;

///
/// Two-dimensional array: a stack of xvector rows.
///
template <typename T = double>
using xmatrix = xt::xtensor<T, 2>;

}  // namespace viam::trajex
