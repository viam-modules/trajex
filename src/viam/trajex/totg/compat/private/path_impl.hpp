#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <optional>

#include <Eigen/Core>

#include <viam/trajex/totg/compat/path.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg::compat {

// This lives in a header because trajectory.cpp needs it too, for the trajex path that a compat
// Path wraps and the error the Path captured. Either the path or the error is set, never both.
struct Path::impl {
    std::optional<totg::path> path;
    std::exception_ptr error;
};

inline Eigen::VectorXd to_eigen(const xvector<>& v) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(v.size()));
    std::copy_n(v.data(), v.size(), result.data());
    return result;
}

inline xvector<> to_xvector(const Eigen::VectorXd& v) {
    // We spell out the shape type, because a braced size alone would select the initializer
    // list constructor and build a one-element vector holding the size.
    xvector<> result(xvector<>::shape_type{static_cast<std::size_t>(v.size())});
    std::copy_n(v.data(), v.size(), result.data());
    return result;
}

}  // namespace viam::trajex::totg::compat
