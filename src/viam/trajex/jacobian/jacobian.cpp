#include <viam/trajex/jacobian/jacobian.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace viam::trajex::jacobian {

namespace {

using vec3 = std::array<double, 3>;

double dot(const vec3& a, const vec3& b) {
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}
vec3 cross(const vec3& a, const vec3& b) {
    return {(a[1] * b[2]) - (a[2] * b[1]), (a[2] * b[0]) - (a[0] * b[2]), (a[0] * b[1]) - (a[1] * b[0])};
}
double norm(const vec3& a) {
    return std::sqrt(dot(a, a));
}
vec3 normalized(const vec3& a) {
    const double n = norm(a);
    return {a[0] / n, a[1] / n, a[2] / n};
}

xt::xarray<double> identity4() {
    return xt::eye<double>(4);
}

// Multiply two 4x4 homogeneous transforms. Inputs are always 4x4 here, so the
// bounds are fixed rather than read from the operands. Hand-rolled because
// xtensor has no matmul without xtensor-blas, which the core library avoids.
xt::xarray<double> matmul4(const xt::xarray<double>& a, const xt::xarray<double>& b) {
    xt::xarray<double> c = xt::zeros<double>({std::size_t{4}, std::size_t{4}});
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                s += a(i, k) * b(k, j);
            }
            c(i, j) = s;
        }
    }
    return c;
}

// 4x4 rotation about a unit axis by theta radians (Rodrigues).
xt::xarray<double> axis_rotation4(const vec3& axis, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double t = 1.0 - c;
    const double x = axis[0];
    const double y = axis[1];
    const double z = axis[2];

    xt::xarray<double> r = identity4();
    r(0, 0) = (t * x * x) + c;
    r(0, 1) = (t * x * y) - (s * z);
    r(0, 2) = (t * x * z) + (s * y);
    r(1, 0) = (t * x * y) + (s * z);
    r(1, 1) = (t * y * y) + c;
    r(1, 2) = (t * y * z) - (s * x);
    r(2, 0) = (t * x * z) - (s * y);
    r(2, 1) = (t * y * z) + (s * x);
    r(2, 2) = (t * z * z) + c;
    return r;
}

// 4x4 transform for a URDF link: rotation Rz(yaw) * Ry(pitch) * Rx(roll) with
// translation xyz.
xt::xarray<double> link_transform(const vec3& xyz, const vec3& rpy) {
    const xt::xarray<double> rx = axis_rotation4({1.0, 0.0, 0.0}, rpy[0]);
    const xt::xarray<double> ry = axis_rotation4({0.0, 1.0, 0.0}, rpy[1]);
    const xt::xarray<double> rz = axis_rotation4({0.0, 0.0, 1.0}, rpy[2]);
    xt::xarray<double> out = matmul4(matmul4(rz, ry), rx);
    out(0, 3) = xyz[0];
    out(1, 3) = xyz[1];
    out(2, 3) = xyz[2];
    return out;
}

// Rotate a 3-vector by the rotation part of a 4x4 transform.
vec3 rotate(const xt::xarray<double>& transform, const vec3& v) {
    return {
        (transform(0, 0) * v[0]) + (transform(0, 1) * v[1]) + (transform(0, 2) * v[2]),
        (transform(1, 0) * v[0]) + (transform(1, 1) * v[1]) + (transform(1, 2) * v[2]),
        (transform(2, 0) * v[0]) + (transform(2, 1) * v[1]) + (transform(2, 2) * v[2]),
    };
}

vec3 translation(const xt::xarray<double>& transform) {
    return {transform(0, 3), transform(1, 3), transform(2, 3)};
}

}  // namespace

// Per-revolute-joint world-frame axis and origin, plus the end-effector
// position, captured while evaluating the forward kinematics. With these in
// hand, Jacobian column i is J_v_i = axes[i] x (p_e - positions[i]) (linear)
// stacked on J_w_i = axes[i] (angular).
//
// axes and positions are std::vector<vec3>, not (N, 3) xtensor arrays: they
// are filled one joint at a time during the walk and read back one row at a
// time by the per-joint cross product, so a row vector is the natural unit. A
// vectorized xtensor assembly is deferred to the linear-algebra cleanup.
struct kinematic_chain::chain_state {
    std::vector<vec3> axes;
    std::vector<vec3> positions;
    vec3 p_e;
};

// Evaluate the forward kinematics row by row (base to end-effector),
// accumulating the running link transform. For each revolute joint, capture
// its world-frame axis and origin before applying joint motion.
kinematic_chain::chain_state kinematic_chain::compute_chain_state_(const xt::xarray<double>& q) const {
    if (q.size() != actuated_count_) {
        throw std::invalid_argument("viam::trajex::jacobian: q size mismatch: expected " + std::to_string(actuated_count_) +
                                    " (actuated joints), got " + std::to_string(q.size()));
    }

    chain_state state;
    state.axes.reserve(actuated_count_);
    state.positions.reserve(actuated_count_);

    xt::xarray<double> running_transform = identity4();
    std::size_t qi = 0;
    for (const joint_row& row : rows_) {
        running_transform = matmul4(running_transform, link_transform(row.xyz, row.rpy));

        if (row.type == joint_type_::k_revolute) {
            const vec3 axis_local = normalized(row.axis);
            state.axes.push_back(rotate(running_transform, axis_local));
            state.positions.push_back(translation(running_transform));

            running_transform = matmul4(running_transform, axis_rotation4(axis_local, q(qi)));
            ++qi;
        }
        // fixed: no motion to apply.
    }
    state.p_e = translation(running_transform);
    return state;
}

kinematic_chain::kinematic_chain(std::vector<joint_row> rows) : rows_(std::move(rows)) {
    if (rows_.empty()) {
        throw std::invalid_argument("viam::trajex::jacobian: empty model table");
    }
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        const joint_row& row = rows_[i];
        switch (row.type) {
            case joint_type_::k_revolute:
                if (dot(row.axis, row.axis) == 0.0) {
                    throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) +
                                                " is a revolute joint with zero-magnitude axis");
                }
                ++actuated_count_;
                break;
            case joint_type_::k_fixed:
                break;
            case joint_type_::k_continuous:
            case joint_type_::k_prismatic:
                throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) +
                                            " has unsupported joint type (only revolute and fixed are supported)");
            default:
                throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) +
                                            " joint type does not match any joint_type value");
        }
    }
}

kinematic_chain kinematic_chain::from(const xt::xarray<double>& tensor) {
    if (tensor.dimension() != 2) {
        throw std::invalid_argument("viam::trajex::jacobian: expected 2D model-table tensor, got " + std::to_string(tensor.dimension()) +
                                    "D");
    }
    if (tensor.shape()[1] != 10) {
        throw std::invalid_argument("viam::trajex::jacobian: expected model-table shape (n, 10), got (n, " +
                                    std::to_string(tensor.shape()[1]) + ")");
    }

    std::vector<joint_row> rows;
    rows.reserve(tensor.shape()[0]);
    for (std::size_t i = 0; i < tensor.shape()[0]; ++i) {
        const double type_value = tensor(i, 9);
        const int type_int = static_cast<int>(type_value);
        if (static_cast<double>(type_int) != type_value) {
            throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) + " joint type value " +
                                        std::to_string(type_value) + " is not an integer");
        }

        joint_row row;
        row.xyz = {tensor(i, 0), tensor(i, 1), tensor(i, 2)};
        row.rpy = {tensor(i, 3), tensor(i, 4), tensor(i, 5)};
        row.axis = {tensor(i, 6), tensor(i, 7), tensor(i, 8)};
        row.type = static_cast<joint_type_>(type_int);
        rows.push_back(row);
    }
    return kinematic_chain(std::move(rows));
}

xt::xarray<double> kinematic_chain::jacobian(const xt::xarray<double>& q) const {
    const chain_state state = compute_chain_state_(q);

    xt::xarray<double> J = xt::zeros<double>({std::size_t{6}, actuated_count_});
    for (std::size_t i = 0; i < actuated_count_; ++i) {
        const vec3& w = state.axes[i];
        const vec3& p = state.positions[i];
        const vec3 jv = cross(w, {state.p_e[0] - p[0], state.p_e[1] - p[1], state.p_e[2] - p[2]});
        J(0, i) = jv[0];
        J(1, i) = jv[1];
        J(2, i) = jv[2];
        J(3, i) = w[0];
        J(4, i) = w[1];
        J(5, i) = w[2];
    }
    return J;
}

xt::xarray<double> kinematic_chain::linear_jacobian(const xt::xarray<double>& q) const {
    const chain_state state = compute_chain_state_(q);

    xt::xarray<double> J = xt::zeros<double>({std::size_t{3}, actuated_count_});
    for (std::size_t i = 0; i < actuated_count_; ++i) {
        const vec3& w = state.axes[i];
        const vec3& p = state.positions[i];
        const vec3 jv = cross(w, {state.p_e[0] - p[0], state.p_e[1] - p[1], state.p_e[2] - p[2]});
        J(0, i) = jv[0];
        J(1, i) = jv[1];
        J(2, i) = jv[2];
    }
    return J;
}

}  // namespace viam::trajex::jacobian
