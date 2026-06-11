#include <viam/trajex/jacobian/jacobian.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace viam::trajex::jacobian {

namespace {

// TODO(RSDK-14104): Remove duplicated code that exists in the viam-cpp-sdk
using vec3 = std::array<double, 3>;

// Joint-type encodings for column 9 of the model-table tensor. These match
// viam::sdk::ModelTable::JointType: revolute=0, continuous=1, prismatic=2,
// fixed=3.
enum class joint_type : int { k_revolute = 0, k_continuous = 1, k_prismatic = 2, k_fixed = 3 };

// One unpacked row of the model-table tensor: columns 0..2 xyz, 3..5 rpy
// (fixed-axis XYZ), 6..8 axis, 9 joint type.
struct joint_row {
    vec3 xyz;
    vec3 rpy;
    vec3 axis;
    joint_type type;
};

double dot(const vec3& a, const vec3& b) {
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}
vec3 cross(const vec3& a, const vec3& b) {
    return {(a[1] * b[2]) - (a[2] * b[1]), (a[2] * b[0]) - (a[0] * b[2]), (a[0] * b[1]) - (a[1] * b[0])};
}
double norm(const vec3& a) {
    return std::sqrt(dot(a, a));
}

// Validate the model-table tensor and unpack it into rows. Throws
// std::invalid_argument on malformed shape, invalid joint-type encoding,
// unsupported joint type (continuous or prismatic), or revolute row with
// zero-magnitude axis.
std::vector<joint_row> parse_model_table(const xt::xarray<double>& tensor) {
    if (tensor.dimension() != 2) {
        throw std::invalid_argument("viam::trajex::jacobian: expected 2D model-table tensor, got " + std::to_string(tensor.dimension()) +
                                    "D");
    }
    if (tensor.shape()[1] != 10) {
        throw std::invalid_argument("viam::trajex::jacobian: expected model-table shape (n, 10), got (n, " +
                                    std::to_string(tensor.shape()[1]) + ")");
    }
    if (tensor.shape()[0] == 0) {
        throw std::invalid_argument("viam::trajex::jacobian: empty model table");
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
        row.type = static_cast<joint_type>(type_int);

        switch (row.type) {
            case joint_type::k_revolute:
                if (dot(row.axis, row.axis) < 1e-24) {
                    throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) +
                                                " is a revolute joint with zero-magnitude axis");
                }
                break;
            case joint_type::k_fixed:
                break;
            case joint_type::k_continuous:
            case joint_type::k_prismatic:
                throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) +
                                            " has unsupported joint type (only revolute and fixed are supported)");
            default:
                throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) + " joint type value " +
                                            std::to_string(type_int) + " does not match any joint type");
        }
        rows.push_back(row);
    }
    return rows;
}

xt::xarray<double> identity4() {
    xt::xarray<double> t = xt::zeros<double>({std::size_t{4}, std::size_t{4}});
    for (std::size_t i = 0; i < 4; ++i) {
        t(i, i) = 1.0;
    }
    return t;
}

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

void check_q_size(std::size_t n_actuated, std::size_t q_size) {
    if (q_size != n_actuated) {
        throw std::invalid_argument("viam::trajex::jacobian: q size mismatch: expected " + std::to_string(n_actuated) +
                                    " (actuated joints), got " + std::to_string(q_size));
    }
}

// Accumulated end-effector transform plus, for each revolute joint, its
// world-frame axis and origin captured before its motion is applied.
struct chain_state {
    xt::xarray<double> transform;
    std::vector<vec3> axes;
    std::vector<vec3> positions;
};

// Walk the chain. For each revolute joint, capture its world-frame axis and
// origin before applying joint motion (equivalent to post-motion for rotation
// about own axis; using pre-motion is clearer).
chain_state run_chain(const std::vector<joint_row>& rows, const xt::xarray<double>& q) {
    std::size_t n_actuated = 0;
    for (const auto& row : rows) {
        if (row.type == joint_type::k_revolute) {
            ++n_actuated;
        }
    }
    check_q_size(n_actuated, q.size());

    chain_state state{identity4(), {}, {}};
    state.axes.reserve(n_actuated);
    state.positions.reserve(n_actuated);

    std::size_t qi = 0;
    for (const auto& row : rows) {
        state.transform = matmul4(state.transform, link_transform(row.xyz, row.rpy));

        if (row.type == joint_type::k_revolute) {
            const double n = norm(row.axis);
            const vec3 axis_local{row.axis[0] / n, row.axis[1] / n, row.axis[2] / n};
            state.axes.push_back(rotate(state.transform, axis_local));
            state.positions.push_back(translation(state.transform));

            state.transform = matmul4(state.transform, axis_rotation4(axis_local, q(qi)));
            ++qi;
        }
        // fixed: no motion to apply.
    }
    return state;
}

}  // namespace

xt::xarray<double> compute_jacobian(const xt::xarray<double>& model_table, const xt::xarray<double>& q) {
    const std::vector<joint_row> rows = parse_model_table(model_table);
    const chain_state state = run_chain(rows, q);
    const std::size_t n_actuated = state.axes.size();
    const vec3 p_e = translation(state.transform);

    xt::xarray<double> J = xt::zeros<double>({std::size_t{6}, n_actuated});
    for (std::size_t i = 0; i < n_actuated; ++i) {
        const vec3& w = state.axes[i];
        const vec3& p = state.positions[i];
        const vec3 jv = cross(w, {p_e[0] - p[0], p_e[1] - p[1], p_e[2] - p[2]});
        J(0, i) = jv[0];
        J(1, i) = jv[1];
        J(2, i) = jv[2];
        J(3, i) = w[0];
        J(4, i) = w[1];
        J(5, i) = w[2];
    }
    return J;
}

}  // namespace viam::trajex::jacobian
