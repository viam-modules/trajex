#pragma once

#include <Eigen/Core>

namespace viam::trajex::jacobian::detail {

// Standard (distal) DH transform: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha).
Eigen::Matrix4d dh_transform(double d, double theta, double a, double alpha);

}  // namespace viam::trajex::jacobian::detail
