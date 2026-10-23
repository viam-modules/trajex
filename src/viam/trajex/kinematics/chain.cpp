#include <viam/trajex/kinematics/chain.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace viam::trajex::kinematics {

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

// The FK walk holds its running rigid transform as a unit quaternion and a translation rather
// than a 4x4 matrix. That is the form joint origins arrive in, and it makes the per-joint
// rotation cheap to build and to compose, which matters because the walk runs once or twice per
// integration step.
//
// Quaternions are w, x, y, z in the Hamilton convention, and rotate a vector actively,
// v' = q v q*.
using quat = std::array<double, 4>;

quat multiply(const quat& a, const quat& b) {
    return {
        (a[0] * b[0]) - (a[1] * b[1]) - (a[2] * b[2]) - (a[3] * b[3]),
        (a[0] * b[1]) + (a[1] * b[0]) + (a[2] * b[3]) - (a[3] * b[2]),
        (a[0] * b[2]) - (a[1] * b[3]) + (a[2] * b[0]) + (a[3] * b[1]),
        (a[0] * b[3]) + (a[1] * b[2]) - (a[2] * b[1]) + (a[3] * b[0]),
    };
}

// Rotate v by a unit quaternion, as v + w t + u x t with t = 2 u x v, which is q v q* without
// forming the products in full.
vec3 rotate(const quat& q, const vec3& v) {
    const vec3 u = {q[1], q[2], q[3]};
    const vec3 uv = cross(u, v);
    const vec3 t = {2.0 * uv[0], 2.0 * uv[1], 2.0 * uv[2]};
    const vec3 ut = cross(u, t);
    return {v[0] + (q[0] * t[0]) + ut[0], v[1] + (q[0] * t[1]) + ut[1], v[2] + (q[0] * t[2]) + ut[2]};
}

// Rotation about a unit axis by theta radians.
quat axis_rotation(const vec3& axis, double theta) {
    const double half = 0.5 * theta;
    const double s = std::sin(half);
    return {std::cos(half), s * axis[0], s * axis[1], s * axis[2]};
}

// A rigid transform, applied as rotation then translation. The default is the identity.
struct rigid_transform {
    quat rotation{1.0, 0.0, 0.0, 0.0};
    vec3 translation{0.0, 0.0, 0.0};
};

// The transform `parent` followed by (rotation, translation) expressed in parent's frame, which
// is what the matrix product parent * [R t; 0 1] gives.
rigid_transform compose(const rigid_transform& parent, const quat& rotation, const vec3& translation) {
    const vec3 moved = rotate(parent.rotation, translation);
    return {
        .rotation = multiply(parent.rotation, rotation),
        .translation = {parent.translation[0] + moved[0], parent.translation[1] + moved[1], parent.translation[2] + moved[2]},
    };
}

bool all_finite(const auto& values) {
    return std::ranges::all_of(values, [](double v) { return std::isfinite(v); });
}

// The linear-velocity Jacobian column of a moving joint whose world-frame axis and origin were
// captured by the forward-kinematics walk. A translating joint moves the end effector along its
// axis; a rotating joint moves it tangentially about its axis.
vec3 linear_column(const auto& joint, const vec3& p_e) {
    if (joint.translates) {
        return joint.axis;
    }
    return cross(joint.axis, {p_e[0] - joint.position[0], p_e[1] - joint.position[1], p_e[2] - joint.position[2]});
}

}  // namespace

// The world-frame axis and origin of each moving joint the forward kinematics passed through, and
// the position p_e of the frame where it stopped. With these, a rotating joint's Jacobian column
// is the linear part J_v_i = axis x (p_e - position) over the angular part J_w_i = axis, and a
// translating joint's is J_v_i = axis over zero.
//
// The joints are a std::vector of small structs, not (N, 3) xtensor arrays: they are filled one
// joint at a time during the walk and read back one joint at a time by the per-joint column
// assembly, so a joint is the natural unit. A vectorized xtensor assembly is deferred to the
// linear-algebra cleanup.
struct chain::chain_state_ {
    struct moving_joint {
        vec3 axis;
        vec3 position;
        bool translates;
    };
    std::vector<moving_joint> joints;
    vec3 p_e;
};

// Evaluate the forward kinematics joint by joint from the base through last_joint, accumulating
// the running transform. For each moving joint we capture its world-frame axis and origin before
// applying its motion, which is safe because a joint neither rotates nor translates its own axis.
// The origin transforms and unit axes do not depend on q, so they were computed once at
// construction. Joints after last_joint cannot move anything before them, so we stop there, but q
// is still checked against the whole chain, because callers always hand us a full joint vector.
chain::chain_state_ chain::compute_chain_state_(const xvector<>& q, std::size_t last_joint) const {
    if (q.size() != actuated_count_) {
        throw std::invalid_argument("viam::trajex::kinematics: q size mismatch: expected " + std::to_string(actuated_count_) +
                                    " (actuated joints), got " + std::to_string(q.size()));
    }

    chain_state_ state;
    state.joints.reserve(actuated_count_);

    rigid_transform running;
    std::size_t qi = 0;
    for (std::size_t index = 0; index <= last_joint; ++index) {
        std::visit(
            [&]<typename Joint>(const Joint& j) {
                running = compose(running, j.origin_rotation, j.origin.translation);
                if constexpr (!std::is_same_v<Joint, fixed_joint_>) {
                    constexpr bool translates = std::is_same_v<Joint, translating_joint_>;
                    const vec3 axis = rotate(running.rotation, j.unit_axis);
                    state.joints.push_back({axis, running.translation, translates});

                    // A translation along the joint's own axis moves the frame along that axis as
                    // it lies in the world, which we have just computed.
                    if constexpr (translates) {
                        const double d = q(qi);
                        running.translation = {running.translation[0] + (axis[0] * d),
                                               running.translation[1] + (axis[1] * d),
                                               running.translation[2] + (axis[2] * d)};
                    } else {
                        running.rotation = multiply(running.rotation, axis_rotation(j.unit_axis, q(qi)));
                    }
                    ++qi;
                }
            },
            joints_[index]);
    }
    state.p_e = running.translation;
    return state;
}

chain::chain(std::vector<joint_> joints) : joints_(std::move(joints)) {
    actuated_count_ = static_cast<std::size_t>(
        std::ranges::count_if(joints_, [](const joint_& joint) { return !std::holds_alternative<fixed_joint_>(joint); }));
}

xmatrix<> chain::jacobian(const xvector<>& q) const {
    const chain_state_ state = compute_chain_state_(q, joints_.size() - 1);

    xmatrix<> J = xt::zeros<double>({std::size_t{6}, actuated_count_});
    for (std::size_t i = 0; i < actuated_count_; ++i) {
        const auto& joint = state.joints[i];
        const vec3 jv = linear_column(joint, state.p_e);
        J(0, i) = jv[0];
        J(1, i) = jv[1];
        J(2, i) = jv[2];
        if (!joint.translates) {
            J(3, i) = joint.axis[0];
            J(4, i) = joint.axis[1];
            J(5, i) = joint.axis[2];
        }
    }
    return J;
}

xmatrix<> chain::linear_jacobian(const xvector<>& q) const {
    const chain_state_ state = compute_chain_state_(q, joints_.size() - 1);

    xmatrix<> J = xt::zeros<double>({std::size_t{3}, actuated_count_});
    for (std::size_t i = 0; i < actuated_count_; ++i) {
        const vec3 jv = linear_column(state.joints[i], state.p_e);
        J(0, i) = jv[0];
        J(1, i) = jv[1];
        J(2, i) = jv[2];
    }
    return J;
}

chain::frame_kinematics chain::forward_kinematics(std::size_t joint,
                                                  const xvector<>& q,
                                                  const xvector<>& q_dot,
                                                  const xvector<>& q_ddot) const {
    if (joint >= joints_.size()) {
        throw std::out_of_range("viam::trajex::kinematics: joint " + std::to_string(joint) + " is past the end of a chain of " +
                                std::to_string(joints_.size()) + " joints");
    }
    const auto require_actuated_size = [this](const char* name, const xvector<>& values) {
        if (values.size() != actuated_count_) {
            throw std::invalid_argument(std::string{"viam::trajex::kinematics: "} + name + " size mismatch: expected " +
                                        std::to_string(actuated_count_) + " (actuated joints), got " + std::to_string(values.size()));
        }
    };
    require_actuated_size("q_dot", q_dot);
    require_actuated_size("q_ddot", q_ddot);

    // Only the moving joints up to and including `joint` move its frame. They come first in q, so
    // the leading entries of q_dot and q_ddot are the ones that apply to them.
    const chain_state_ state = compute_chain_state_(q, joint);
    const std::size_t moving = state.joints.size();

    vec3 v{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < moving; ++i) {
        const vec3 col = linear_column(state.joints[i], state.p_e);
        v = {v[0] + (col[0] * q_dot(i)), v[1] + (col[1] * q_dot(i)), v[2] + (col[2] * q_dot(i))};
    }

    // Differentiate v = sum(col_i * q_dot_i) from the base to the tip. Everything after joint i
    // moves with the joints before it, so we carry their motion along as we go. omega is the
    // angular velocity that the rotating joints so far impart, and a point p moves at
    // omega x p - c + t, where c sums (w_j q_dot_j) x p_j over those rotating joints and t sums
    // w_j q_dot_j over the translating ones. A joint's axis therefore turns at omega x w, and the
    // lever arm r = p_e - p of a rotating joint changes at v minus the velocity of p.
    vec3 dv{0.0, 0.0, 0.0};
    vec3 omega{0.0, 0.0, 0.0};
    vec3 c{0.0, 0.0, 0.0};
    vec3 t{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < moving; ++i) {
        const auto& captured = state.joints[i];
        const vec3& w = captured.axis;
        const vec3& p = captured.position;
        const vec3 dw = cross(omega, w);

        vec3 col = w;
        vec3 d_col = dw;
        if (!captured.translates) {
            const vec3 r = {state.p_e[0] - p[0], state.p_e[1] - p[1], state.p_e[2] - p[2]};
            col = cross(w, r);
            const vec3 cp = cross(omega, p);
            const vec3 v_point = {cp[0] - c[0] + t[0], cp[1] - c[1] + t[1], cp[2] - c[2] + t[2]};
            const vec3 dr = {v[0] - v_point[0], v[1] - v_point[1], v[2] - v_point[2]};
            const vec3 d_col_a = cross(dw, r);
            const vec3 d_col_b = cross(w, dr);
            d_col = {d_col_a[0] + d_col_b[0], d_col_a[1] + d_col_b[1], d_col_a[2] + d_col_b[2]};
        }
        dv = {dv[0] + (d_col[0] * q_dot(i)) + (col[0] * q_ddot(i)),
              dv[1] + (d_col[1] * q_dot(i)) + (col[1] * q_ddot(i)),
              dv[2] + (d_col[2] * q_dot(i)) + (col[2] * q_ddot(i))};

        const vec3 wq = {w[0] * q_dot(i), w[1] * q_dot(i), w[2] * q_dot(i)};
        if (captured.translates) {
            t = {t[0] + wq[0], t[1] + wq[1], t[2] + wq[2]};
        } else {
            omega = {omega[0] + wq[0], omega[1] + wq[1], omega[2] + wq[2]};
            const vec3 wqp = cross(wq, p);
            c = {c[0] + wqp[0], c[1] + wqp[1], c[2] + wqp[2]};
        }
    }

    return {.position = state.p_e, .linear_velocity = v, .linear_acceleration = dv};
}

chain::builder& chain::builder::extend(const joint_origin& origin) & {
    joints_.emplace_back(fixed_joint_{{origin, {}}});
    return *this;
}

chain::builder& chain::builder::extend(const joint_origin& origin, const rotation_axis& axis) & {
    joints_.emplace_back(rotating_joint_{{origin, {}}, axis, {}});
    return *this;
}

chain::builder& chain::builder::extend(const joint_origin& origin, const translation_axis& axis) & {
    joints_.emplace_back(translating_joint_{{origin, {}}, axis, {}});
    return *this;
}

chain chain::builder::build() && {
    if (joints_.empty()) {
        throw std::invalid_argument("viam::trajex::kinematics: a kinematic chain needs at least one joint");
    }

    for (std::size_t i = 0; i < joints_.size(); ++i) {
        const std::string which = "viam::trajex::kinematics: joint " + std::to_string(i);
        std::visit(
            [&]<typename Joint>(Joint& joint) {
                const auto& [translation, rotation] = joint.origin;
                if (!all_finite(translation)) {
                    throw std::invalid_argument(which + " has a non-finite translation");
                }

                // Any nonzero quaternion denotes exactly one rotation, so we normalize rather than
                // demand that the caller already did. Only a zero or non-finite quaternion is
                // refused, since there is no rotation to recover from it. The norm is checked as
                // well as the components because large finite components can still overflow it.
                // The quaternion as given is kept for visit_joints; only the walk sees the
                // normalized one.
                const auto& [w, x, y, z] = rotation;
                const double rotation_norm = std::sqrt((w * w) + (x * x) + (y * y) + (z * z));
                if (!all_finite(rotation) || !std::isfinite(rotation_norm) || rotation_norm == 0.0) {
                    throw std::invalid_argument(which + " has a zero or non-finite quaternion");
                }
                joint.origin_rotation = {w / rotation_norm, x / rotation_norm, y / rotation_norm, z / rotation_norm};

                if constexpr (!std::is_same_v<Joint, fixed_joint_>) {
                    const auto& direction = joint.axis.direction;
                    const double axis_norm = norm(direction);
                    if (!all_finite(direction) || !std::isfinite(axis_norm) || axis_norm == 0.0) {
                        throw std::invalid_argument(which + " has a zero or non-finite axis");
                    }
                    joint.unit_axis = normalized(direction);
                }
            },
            joints_[i]);
    }
    return chain(std::move(joints_));
}

}  // namespace viam::trajex::kinematics
