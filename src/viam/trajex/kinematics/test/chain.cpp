#include <viam/trajex/kinematics/chain.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/xt.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>

namespace {

using viam::trajex::xmatrix;
using viam::trajex::xvector;
using viam::trajex::kinematics::chain;

double matrix_diff_norm(const xmatrix<>& A, const xmatrix<>& B) {
    double s = 0.0;
    const auto* a = A.begin();
    const auto* b = B.begin();
    for (; a != A.end(); ++a, ++b) {
        const double d = *a - *b;
        s += d * d;
    }
    return std::sqrt(s);
}

double vec3_diff_norm(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

struct twist {
    std::array<double, 3> v;
    std::array<double, 3> w;
};
twist J_times_qdot(const xmatrix<>& J, const xvector<>& q_dot) {
    twist t{{0, 0, 0}, {0, 0, 0}};
    const std::size_t n = q_dot.size();
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < 3; ++i) {
            t.v[i] += J(i, j) * q_dot(j);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            t.w[i] += J(3 + i, j) * q_dot(j);
        }
    }
    return t;
}

// A chain as the builder takes it, kept as plain data so the reference forward kinematics below
// can walk it without going through kinematics::chain.
enum class test_joint_kind : std::uint8_t { k_fixed, k_rotational, k_translational };

struct test_joint {
    test_joint_kind kind;
    std::array<double, 3> xyz;
    std::array<double, 4> wxyz;
    std::array<double, 3> axis;
};

using test_chain = std::vector<test_joint>;

constexpr std::array<double, 4> k_identity_rotation = {1.0, 0.0, 0.0, 0.0};

test_joint rotating(std::array<double, 3> xyz, std::array<double, 3> axis, std::array<double, 4> wxyz = k_identity_rotation) {
    return {.kind = test_joint_kind::k_rotational, .xyz = xyz, .wxyz = wxyz, .axis = axis};
}

test_joint translating(std::array<double, 3> xyz, std::array<double, 3> axis, std::array<double, 4> wxyz = k_identity_rotation) {
    return {.kind = test_joint_kind::k_translational, .xyz = xyz, .wxyz = wxyz, .axis = axis};
}

test_joint fixed(std::array<double, 3> xyz, std::array<double, 4> wxyz = k_identity_rotation) {
    return {.kind = test_joint_kind::k_fixed, .xyz = xyz, .wxyz = wxyz, .axis = {}};
}

chain build_chain(const test_chain& joints) {
    chain::builder builder;
    for (const auto& j : joints) {
        const chain::joint_origin origin{.translation = j.xyz, .rotation = j.wxyz};
        switch (j.kind) {
            case test_joint_kind::k_fixed:
                builder.extend(origin);
                break;
            case test_joint_kind::k_rotational:
                builder.extend(origin, chain::rotation_axis{j.axis});
                break;
            case test_joint_kind::k_translational:
                builder.extend(origin, chain::translation_axis{j.axis});
                break;
        }
    }
    return std::move(builder).build();
}

// One-shot convenience for tests: build the chain and evaluate at q.
xmatrix<> compute_jacobian(const test_chain& joints, const xvector<>& q) {
    return build_chain(joints).jacobian(q);
}

// The quaternion for a fixed-axis XYZ rpy, which composes as Rz(yaw) * Ry(pitch) * Rx(roll).
std::array<double, 4> rpy_to_quaternion(double roll, double pitch, double yaw) {
    const double cr = std::cos(roll / 2.0);
    const double sr = std::sin(roll / 2.0);
    const double cp = std::cos(pitch / 2.0);
    const double sp = std::sin(pitch / 2.0);
    const double cy = std::cos(yaw / 2.0);
    const double sy = std::sin(yaw / 2.0);
    return {
        (cr * cp * cy) + (sr * sp * sy), (sr * cp * cy) - (cr * sp * sy), (cr * sp * cy) + (sr * cp * sy), (cr * cp * sy) - (sr * sp * cy)};
}

// Two 1m planar links rotating about z, ending in a 1m fixed flange.
test_chain twolink_chain() {
    return {rotating({0, 0, 0}, {0, 0, 1}), rotating({1, 0, 0}, {0, 0, 1}), fixed({1, 0, 0})};
}

// 3 revolute joints separated by fixed 1m spacers, ending in a 1m flange.
test_chain threelink_with_spacers_chain() {
    return {
        rotating({0, 0, 0}, {0, 0, 1}),
        fixed({1, 0, 0}),
        rotating({0, 0, 0}, {0, 0, 1}),
        fixed({1, 0, 0}),
        rotating({0, 0, 0}, {0, 0, 1}),
        fixed({1, 0, 0}),
    };
}

// 6-revolute spatial chain mimicking a UR-like structure.
test_chain sixdof_arm_chain() {
    return {
        rotating({0, 0, 0.10}, {0, 0, 1}),
        rotating({0, 0, 0.15}, {0, 1, 0}),
        rotating({0.4, 0, 0}, {0, 1, 0}),
        rotating({0.4, 0, 0}, {0, 1, 0}),
        rotating({0, 0, 0.10}, {1, 0, 0}),
        rotating({0, 0, 0.10}, {0, 0, 1}),
        fixed({0, 0, 0.05}),
    };
}

// Spatial origins with real rotations throughout, which none of the chains above have, ending in
// a tool flipped and pitched the way a URDF tool frame often is.
test_chain rotated_origins_chain() {
    return {
        rotating({0, 0, 0.10}, {0, 0, 1}, rpy_to_quaternion(0.3, -0.2, 0.5)),
        rotating({0.05, 0, 0.15}, {0, 1, 0}, rpy_to_quaternion(-0.4, 0.1, 0.2)),
        rotating({0.4, 0.02, 0}, {0, 1, 0}, rpy_to_quaternion(0.0, 0.6, -0.3)),
        fixed({0.1, 0, 0.05}, rpy_to_quaternion(3.14159265, -1.570796, 0)),
    };
}

// A SCARA: two rotations about z, a vertical prismatic quill, then a wrist rotation.
test_chain scara_chain() {
    return {
        rotating({0, 0, 0.3}, {0, 0, 1}),
        rotating({0.35, 0, 0}, {0, 0, 1}),
        translating({0.3, 0, 0}, {0, 0, -1}),
        rotating({0, 0, -0.05}, {0, 0, 1}),
        fixed({0, 0, -0.1}),
    };
}

// Prismatic and rotating joints interleaved, with rotated origins and axes that are neither unit
// length nor aligned with a coordinate axis, so a translating joint's axis is carried around by
// rotations upstream of it and moves points downstream of it.
test_chain mixed_chain() {
    return {
        translating({0, 0, 0}, {2, 0, 0}),
        rotating({0, 0, 0.2}, {0, 1, 1}, rpy_to_quaternion(0.2, 0.1, -0.3)),
        translating({0.3, 0.1, 0}, {1, 1, 0}, rpy_to_quaternion(-0.5, 0.4, 0.2)),
        rotating({0, 0.2, 0.1}, {1, 0, 0}, rpy_to_quaternion(0.7, 0, 0.3)),
        fixed({0.1, 0, 0.05}, rpy_to_quaternion(0, -0.4, 0)),
    };
}

// The analytic Jacobian in kinematics::chain is the thing being tested, so it
// cannot be its own evidence. The Jacobian's defining property is that it is
// the derivative of the forward kinematics; the numerical-consistency suite
// verifies exactly that property by finite-differencing the reference forward
// kinematics below, which is written independently of kinematics::chain.

xmatrix<> reference_identity4() {
    xmatrix<> t = xt::zeros<double>({std::size_t{4}, std::size_t{4}});
    for (std::size_t i = 0; i < 4; ++i) {
        t(i, i) = 1.0;
    }
    return t;
}

xmatrix<> reference_matmul(const xmatrix<>& a, const xmatrix<>& b) {
    xmatrix<> c = xt::zeros<double>({std::size_t{4}, std::size_t{4}});
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

// 4x4 rotation about a unit axis by angle radians (Rodrigues).
xmatrix<> reference_axis_rotation(double x, double y, double z, double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double t = 1.0 - c;
    xmatrix<> r = reference_identity4();
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

// The rotation a quaternion denotes, built by way of its axis and angle rather than the direct
// matrix formula the implementation uses, so the two are independent.
xmatrix<> reference_quaternion_rotation(const std::array<double, 4>& wxyz) {
    const double n = std::sqrt((wxyz[0] * wxyz[0]) + (wxyz[1] * wxyz[1]) + (wxyz[2] * wxyz[2]) + (wxyz[3] * wxyz[3]));
    const double w = wxyz[0] / n;
    const std::array<double, 3> v = {wxyz[1] / n, wxyz[2] / n, wxyz[3] / n};
    const double s = std::sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]));
    if (s == 0.0) {
        return reference_identity4();
    }
    return reference_axis_rotation(v[0] / s, v[1] / s, v[2] / s, 2.0 * std::atan2(s, w));
}

xmatrix<> forward_transform(const test_chain& joints, const xvector<>& q) {
    xmatrix<> T = reference_identity4();
    std::size_t qi = 0;
    for (const auto& j : joints) {
        xmatrix<> origin = reference_quaternion_rotation(j.wxyz);
        origin(0, 3) = j.xyz[0];
        origin(1, 3) = j.xyz[1];
        origin(2, 3) = j.xyz[2];
        T = reference_matmul(T, origin);

        if (j.kind == test_joint_kind::k_fixed) {
            continue;
        }
        const double an = std::sqrt((j.axis[0] * j.axis[0]) + (j.axis[1] * j.axis[1]) + (j.axis[2] * j.axis[2]));
        if (j.kind == test_joint_kind::k_rotational) {
            T = reference_matmul(T, reference_axis_rotation(j.axis[0] / an, j.axis[1] / an, j.axis[2] / an, q(qi)));
        } else {
            xmatrix<> slide = reference_identity4();
            slide(0, 3) = q(qi) * j.axis[0] / an;
            slide(1, 3) = q(qi) * j.axis[1] / an;
            slide(2, 3) = q(qi) * j.axis[2] / an;
            T = reference_matmul(T, slide);
        }
        ++qi;
    }
    return T;
}

// Numerical geometric Jacobian via central differences on the reference
// forward_transform.
xmatrix<> numerical_jacobian(const test_chain& joints, const xvector<>& q, double delta = 1e-7) {
    const std::size_t n_actuated = q.size();
    xmatrix<> J_num = xt::zeros<double>({std::size_t{6}, n_actuated});

    for (std::size_t i = 0; i < n_actuated; ++i) {
        xvector<> q_plus = q;
        xvector<> q_minus = q;
        q_plus(i) += delta;
        q_minus(i) -= delta;

        const xmatrix<> Tp = forward_transform(joints, q_plus);
        const xmatrix<> Tm = forward_transform(joints, q_minus);

        for (std::size_t r = 0; r < 3; ++r) {
            J_num(r, i) = (Tp(r, 3) - Tm(r, 3)) / (2.0 * delta);
        }

        std::array<std::array<double, 3>, 3> dR{};
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t b = 0; b < 3; ++b) {
                double s = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    s += Tp(a, k) * Tm(b, k);
                }
                dR[a][b] = s;
            }
        }
        const double scale = 1.0 / (2.0 * delta);
        J_num(3, i) = 0.5 * (dR[2][1] - dR[1][2]) * scale;
        J_num(4, i) = 0.5 * (dR[0][2] - dR[2][0]) * scale;
        J_num(5, i) = 0.5 * (dR[1][0] - dR[0][1]) * scale;
    }
    return J_num;
}

void check_matches_numerical(const test_chain& joints, const xvector<>& q, double tol = 1e-6) {
    const auto J = compute_jacobian(joints, q);
    const auto J_num = numerical_jacobian(joints, q);
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_num), tol);
}

// The end point's linear velocity J_v(q) * q_dot, computed with linear_jacobian. That is checked
// against the reference forward kinematics elsewhere in this file, so it can serve as a reference
// here.
std::array<double, 3> linear_velocity_via_jacobian(const chain& chain, const xvector<>& q, const xvector<>& q_dot) {
    const auto J = chain.linear_jacobian(q);
    std::array<double, 3> v{0.0, 0.0, 0.0};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t j = 0; j < q_dot.size(); ++j) {
            v[r] += J(r, j) * q_dot(j);
        }
    }
    return v;
}

double norm(const std::array<double, 3>& v) {
    return std::sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]));
}

// Checks forward_kinematics at the end of the chain. The position is compared with the reference
// forward kinematics and the linear velocity with J_v(q) * q_dot. The linear acceleration is
// compared with a central difference of that velocity along the motion q + t q_dot,
// q_dot + t q_ddot, whose derivative at t = 0 is J_v_dot q_dot + J_v q_ddot.
void check_forward_kinematics(
    const test_chain& joints, const xvector<>& q, const xvector<>& q_dot, const xvector<>& q_ddot, double tol = 1e-6) {
    const auto chain = build_chain(joints);
    const auto fk = chain.forward_kinematics(q, q_dot, q_ddot);

    const auto T = forward_transform(joints, q);
    BOOST_CHECK_SMALL(vec3_diff_norm(fk.position, {T(0, 3), T(1, 3), T(2, 3)}), 1e-12);

    BOOST_CHECK_SMALL(vec3_diff_norm(fk.linear_velocity, linear_velocity_via_jacobian(chain, q, q_dot)), 1e-12);

    const double h = 1e-6;
    const auto v_plus = linear_velocity_via_jacobian(chain, q + (h * q_dot), q_dot + (h * q_ddot));
    const auto v_minus = linear_velocity_via_jacobian(chain, q - (h * q_dot), q_dot - (h * q_ddot));
    const std::array<double, 3> numeric = {
        (v_plus[0] - v_minus[0]) / (2.0 * h), (v_plus[1] - v_minus[1]) / (2.0 * h), (v_plus[2] - v_minus[2]) / (2.0 * h)};
    BOOST_CHECK_SMALL(vec3_diff_norm(fk.linear_acceleration, numeric), tol);
}

// The leading n entries of v.
xvector<> leading(const xvector<>& v, std::size_t n) {
    xvector<> out = xt::zeros<double>({n});
    for (std::size_t i = 0; i < n; ++i) {
        out(i) = v(i);
    }
    return out;
}

// Checks forward_kinematics at every frame of the chain. Frame k depends only on joints 0..k, so
// it must agree with the end frame of a chain built from those joints alone, given the leading
// entries of q, q_dot and q_ddot that belong to them. We check that shorter chain's end frame
// against the references first, so that each frame is compared with an independent answer rather
// than only with another result from the same implementation.
void check_forward_kinematics_at_every_frame(const test_chain& joints,
                                             const xvector<>& q,
                                             const xvector<>& q_dot,
                                             const xvector<>& q_ddot) {
    const auto chain = build_chain(joints);
    BOOST_REQUIRE_EQUAL(chain.size(), joints.size());

    for (std::size_t k = 0; k < joints.size(); ++k) {
        const test_chain prefix(joints.begin(), joints.begin() + static_cast<std::ptrdiff_t>(k + 1));
        const auto n = static_cast<std::size_t>(
            std::count_if(prefix.begin(), prefix.end(), [](const test_joint& j) { return j.kind != test_joint_kind::k_fixed; }));
        const xvector<> q_p = leading(q, n);
        const xvector<> q_dot_p = leading(q_dot, n);
        const xvector<> q_ddot_p = leading(q_ddot, n);

        BOOST_TEST_CONTEXT("frame " << k) {
            check_forward_kinematics(prefix, q_p, q_dot_p, q_ddot_p);

            const auto expected = build_chain(prefix).forward_kinematics(q_p, q_dot_p, q_ddot_p);
            const auto actual = chain.forward_kinematics(k, q, q_dot, q_ddot);
            BOOST_CHECK_SMALL(vec3_diff_norm(actual.position, expected.position), 1e-12);
            BOOST_CHECK_SMALL(vec3_diff_norm(actual.linear_velocity, expected.linear_velocity), 1e-12);
            BOOST_CHECK_SMALL(vec3_diff_norm(actual.linear_acceleration, expected.linear_acceleration), 1e-12);
        }
    }

    // The three-argument form must report the last frame.
    const auto end = chain.forward_kinematics(q, q_dot, q_ddot);
    const auto last = chain.forward_kinematics(chain.size() - 1, q, q_dot, q_ddot);
    BOOST_CHECK(end.position == last.position);
    BOOST_CHECK(end.linear_velocity == last.linear_velocity);
    BOOST_CHECK(end.linear_acceleration == last.linear_acceleration);
}

using viam::trajex::totg::trajectory;

// Checks the evaluator that tcp_limits::from builds. Its value, the TCP's linear speed per unit of
// arc velocity, is compared with ||J_v(q) * q_prime||, and its arc length derivative with a central
// difference of that along the path.
void check_linear_speed_per_arc_velocity(
    const test_chain& joints, const xvector<>& q, const xvector<>& q_prime, const xvector<>& q_double_prime, double tol = 1e-6) {
    const auto chain = build_chain(joints);
    const auto limits = trajectory::tcp_limits::from(chain, 1.0);
    const auto speed_ratio_at = [&](const xvector<>& qq, const xvector<>& qp) { return norm(linear_velocity_via_jacobian(chain, qq, qp)); };

    const auto r = limits.evaluator(q, q_prime, q_double_prime);
    BOOST_REQUIRE(r.has_value());
    BOOST_CHECK_CLOSE(r->value, speed_ratio_at(q, q_prime), 1e-7);

    const double h = 1e-6;
    const double numeric = (speed_ratio_at(q + (h * q_prime), q_prime + (h * q_double_prime)) -
                            speed_ratio_at(q - (h * q_prime), q_prime - (h * q_double_prime))) /
                           (2.0 * h);
    BOOST_CHECK_SMALL(r->arc_length_derivative - numeric, tol);
}

}  // namespace

// The tests in this suite validate that `J * q_dot` produces the expected
// Cartesian velocity.
BOOST_AUTO_TEST_SUITE(jacobian_velocity_tests)

BOOST_AUTO_TEST_CASE(twolink_base_spin_extended) {
    const xvector<> q = xt::zeros<double>({std::size_t{2}});
    const xvector<> q_dot = {1.0, 0.0};

    const auto t = J_times_qdot(compute_jacobian(twolink_chain(), q), q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {0.0, 2.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_base_spin_rotated) {
    const xvector<> q = {std::numbers::pi / 2.0, 0.0};
    const xvector<> q_dot = {1.0, 0.0};

    const auto t = J_times_qdot(compute_jacobian(twolink_chain(), q), q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {-2.0, 0.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_base_spin_bent) {
    const xvector<> q = {0.0, std::numbers::pi / 2.0};
    const xvector<> q_dot = {1.0, 0.0};

    const auto t = J_times_qdot(compute_jacobian(twolink_chain(), q), q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {-1.0, 1.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite check chain::jacobian against hand-computed
// Jacobian values.
BOOST_AUTO_TEST_SUITE(jacobian_ground_truth_tests)

BOOST_AUTO_TEST_CASE(twolink_zero) {
    const auto J = compute_jacobian(twolink_chain(), xt::zeros<double>({std::size_t{2}}));

    const xmatrix<> J_expected = {
        {0.0, 0.0},
        {2.0, 1.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_q1_ninety) {
    const auto J = compute_jacobian(twolink_chain(), xvector<>{std::numbers::pi / 2.0, 0.0});

    const xmatrix<> J_expected = {
        {-2.0, -1.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_q2_ninety) {
    const auto J = compute_jacobian(twolink_chain(), xvector<>{0.0, std::numbers::pi / 2.0});

    const xmatrix<> J_expected = {
        {-1.0, -1.0},
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_CASE(prismatic_column_is_the_world_axis) {
    // At q = 0 the SCARA's quill points straight down, so its linear column is -z and it
    // contributes nothing angular.
    const auto J = compute_jacobian(scara_chain(), xvector<>{0.0, 0.0, 0.0, 0.0});
    for (std::size_t r = 0; r < 6; ++r) {
        BOOST_CHECK_SMALL(J(r, 2) - (r == 2 ? -1.0 : 0.0), 1e-15);
    }
}

BOOST_AUTO_TEST_SUITE_END()

// The numerical-consistency suite depends on the reference forward kinematics
// being correct, so the tests in this suite verify a few easy cases directly.
BOOST_AUTO_TEST_SUITE(fk_tests)

BOOST_AUTO_TEST_CASE(twolink_zero_ee_at_2_0_0) {
    const xmatrix<> T = forward_transform(twolink_chain(), xt::zeros<double>({std::size_t{2}}));
    BOOST_CHECK_CLOSE(T(0, 3), 2.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(1, 3)), 1e-12);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(twolink_q1_pi_over_2_ee_at_0_2_0) {
    const xmatrix<> T = forward_transform(twolink_chain(), xvector<>{std::numbers::pi / 2.0, 0.0});
    BOOST_CHECK_SMALL(std::abs(T(0, 3)), 1e-9);
    BOOST_CHECK_CLOSE(T(1, 3), 2.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(twolink_q2_pi_over_2_ee_at_1_1_0) {
    const xmatrix<> T = forward_transform(twolink_chain(), xvector<>{0.0, std::numbers::pi / 2.0});
    BOOST_CHECK_CLOSE(T(0, 3), 1.0, 1e-9);
    BOOST_CHECK_CLOSE(T(1, 3), 1.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(scara_quill_extends_downward) {
    const xmatrix<> T = forward_transform(scara_chain(), xvector<>{0.0, 0.0, 0.2, 0.0});
    BOOST_CHECK_CLOSE(T(0, 3), 0.65, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(1, 3)), 1e-12);
    BOOST_CHECK_CLOSE(T(2, 3), -0.05, 1e-9);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite check the analytical Jacobian against a
// central-difference numerical Jacobian.
BOOST_AUTO_TEST_SUITE(jacobian_numerical_consistency_tests)

BOOST_AUTO_TEST_CASE(twolink_zero) {
    check_matches_numerical(twolink_chain(), xt::zeros<double>({std::size_t{2}}));
}

BOOST_AUTO_TEST_CASE(twolink_q1_q2_forty_five) {
    check_matches_numerical(twolink_chain(), xvector<>{std::numbers::pi / 4.0, std::numbers::pi / 4.0});
}

BOOST_AUTO_TEST_CASE(twolink_folded_back) {
    check_matches_numerical(twolink_chain(), xvector<>{0.0, std::numbers::pi});
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_zero) {
    check_matches_numerical(threelink_with_spacers_chain(), xt::zeros<double>({std::size_t{3}}));
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_typical) {
    check_matches_numerical(threelink_with_spacers_chain(), xvector<>{0.3, -0.5, 0.8});
}

BOOST_AUTO_TEST_CASE(sixdof_zero) {
    check_matches_numerical(sixdof_arm_chain(), xt::zeros<double>({std::size_t{6}}));
}

BOOST_AUTO_TEST_CASE(sixdof_typical) {
    check_matches_numerical(sixdof_arm_chain(), xvector<>{0.1, -0.5, 1.2, -0.3, 0.6, -0.1});
}

BOOST_AUTO_TEST_CASE(rotated_origins_typical) {
    check_matches_numerical(rotated_origins_chain(), xvector<>{0.4, -0.9, 0.3});
}

BOOST_AUTO_TEST_CASE(scara) {
    check_matches_numerical(scara_chain(), xvector<>{0.0, 0.0, 0.0, 0.0});
    check_matches_numerical(scara_chain(), xvector<>{0.4, -1.1, 0.12, 0.7});
}

BOOST_AUTO_TEST_CASE(mixed) {
    check_matches_numerical(mixed_chain(), xvector<>{0.3, -0.6, 0.2, 1.1});
    check_matches_numerical(mixed_chain(), xvector<>{-0.8, 1.4, -0.3, -0.5});
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite cover kinematics::chain itself: reuse of one built chain across
// evaluations, agreement between its entry points, and the joint vectors it accepts.
BOOST_AUTO_TEST_SUITE(chain_tests)

BOOST_AUTO_TEST_CASE(built_chain_is_reusable_across_evaluations) {
    const auto chain = build_chain(twolink_chain());
    const xvector<> q1 = {0.3, -0.7};
    const xvector<> q2 = {-1.1, 0.4};

    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q1), compute_jacobian(twolink_chain(), q1)), 1e-15);
    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q2), compute_jacobian(twolink_chain(), q2)), 1e-15);
}

BOOST_AUTO_TEST_CASE(linear_jacobian_matches_jacobian_linear_block) {
    const auto chain = build_chain(sixdof_arm_chain());
    const xvector<> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};

    const auto J = chain.jacobian(q);
    const auto J_lin = chain.linear_jacobian(q);
    BOOST_REQUIRE_EQUAL(J_lin.shape()[0], 3U);
    BOOST_REQUIRE_EQUAL(J_lin.shape()[1], 6U);

    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            max_abs_diff = std::max(max_abs_diff, std::abs(J_lin(i, j) - J(i, j)));
        }
    }
    BOOST_CHECK_SMALL(max_abs_diff, 1e-15);
}

BOOST_AUTO_TEST_CASE(end_position_matches_reference_forward_kinematics) {
    for (const auto& [joints, q] : {std::pair{scara_chain(), xvector<>{0.4, -1.1, 0.12, 0.7}},
                                    std::pair{mixed_chain(), xvector<>{0.3, -0.6, 0.2, 1.1}},
                                    std::pair{rotated_origins_chain(), xvector<>{0.4, -0.9, 0.3}}}) {
        const auto T = forward_transform(joints, q);
        BOOST_CHECK_SMALL(vec3_diff_norm(build_chain(joints).end_position(q), {T(0, 3), T(1, 3), T(2, 3)}), 1e-12);
    }
}

// The evaluations run several times per integration step, so an xvector has to go through with no
// copy, and anything that is not rank one is refused.
BOOST_AUTO_TEST_CASE(evaluation_refuses_a_rank_that_is_not_one) {
    using namespace viam::trajex;

    const auto chain = build_chain(twolink_chain());

    const auto evaluable = [&]<typename T>() { return requires(const T& t) { chain.jacobian(t); }; };
    const auto linear_evaluable = [&]<typename T>() { return requires(const T& t) { chain.linear_jacobian(t); }; };
    const auto position_evaluable = [&]<typename T>() { return requires(const T& t) { chain.end_position(t); }; };
    const auto frame_evaluable = [&]<typename T>() { return requires(const T& t) { chain.forward_kinematics(std::size_t{0}, t, t, t); }; };
    const auto end_frame_evaluable = [&]<typename T>() { return requires(const T& t) { chain.forward_kinematics(t, t, t); }; };

    using rank_three = xt::xtensor<double, 3>;

    BOOST_CHECK(evaluable.template operator()<xvector<>>());
    BOOST_CHECK(evaluable.template operator()<xt::xarray<double>>());
    BOOST_CHECK(!evaluable.template operator()<xmatrix<>>());
    BOOST_CHECK(!evaluable.template operator()<rank_three>());

    BOOST_CHECK(linear_evaluable.template operator()<xvector<>>());
    BOOST_CHECK(!linear_evaluable.template operator()<xmatrix<>>());

    BOOST_CHECK(position_evaluable.template operator()<xvector<>>());
    BOOST_CHECK(!position_evaluable.template operator()<xmatrix<>>());

    BOOST_CHECK(frame_evaluable.template operator()<xvector<>>());
    BOOST_CHECK(frame_evaluable.template operator()<xt::xarray<double>>());
    BOOST_CHECK(!frame_evaluable.template operator()<xmatrix<>>());
    BOOST_CHECK(!frame_evaluable.template operator()<rank_three>());

    BOOST_CHECK(end_frame_evaluable.template operator()<xvector<>>());
    BOOST_CHECK(end_frame_evaluable.template operator()<xt::xarray<double>>());
    BOOST_CHECK(!end_frame_evaluable.template operator()<xmatrix<>>());
    BOOST_CHECK(!end_frame_evaluable.template operator()<rank_three>());

    // A rank settled at runtime reaches the check.
    const xt::xarray<double> good = {0.3, -0.7};
    const xt::xarray<double> two_d = {{0.3, -0.7}, {0.1, 0.2}};
    BOOST_CHECK_NO_THROW(static_cast<void>(chain.jacobian(good)));
    BOOST_CHECK_NO_THROW(static_cast<void>(chain.forward_kinematics(good, good, good)));
    BOOST_CHECK_THROW(static_cast<void>(chain.jacobian(two_d)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.linear_jacobian(two_d)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.end_position(two_d)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.forward_kinematics(0, two_d, good, good)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.forward_kinematics(0, good, two_d, good)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.forward_kinematics(0, good, good, two_d)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.forward_kinematics(two_d, good, good)), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_wrong_q_size) {
    const auto chain = build_chain(twolink_chain());
    const xvector<> q = {0.0, 0.0};
    const xvector<> q_bad = {0.0, 0.0, 0.0};
    BOOST_CHECK_THROW(static_cast<void>(chain.jacobian(q_bad)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.linear_jacobian(q_bad)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.end_position(q_bad)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.forward_kinematics(q_bad, q, q)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.forward_kinematics(q, q_bad, q)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.forward_kinematics(q, q, q_bad)), std::invalid_argument);
}

// size() counts every joint given to the builder, which is every frame forward_kinematics can report
// on, while dof() counts only the moving joints, which is the length of q.
BOOST_AUTO_TEST_CASE(size_counts_joints_and_dof_counts_moving_joints) {
    const auto chain = build_chain(scara_chain());
    BOOST_CHECK_EQUAL(chain.size(), 5U);
    BOOST_CHECK_EQUAL(chain.dof(), 4U);
}

BOOST_AUTO_TEST_CASE(rejects_a_frame_past_the_end) {
    const auto chain = build_chain(scara_chain());
    const xvector<> q = {0.4, -1.1, 0.12, 0.7};
    BOOST_CHECK_NO_THROW(static_cast<void>(chain.forward_kinematics(chain.size() - 1, q, q, q)));
    BOOST_CHECK_THROW(static_cast<void>(chain.forward_kinematics(chain.size(), q, q, q)), std::out_of_range);
}

// end_position reports the position that forward_kinematics reports for the chain at rest.
BOOST_AUTO_TEST_CASE(end_position_is_the_resting_end_frame_position) {
    const auto chain = build_chain(mixed_chain());
    const xvector<> q = {0.3, -0.6, 0.2, 1.1};
    const xvector<> zeros = xt::zeros<double>({std::size_t{4}});
    BOOST_CHECK(chain.end_position(q) == chain.forward_kinematics(q, zeros, zeros).position);
}

BOOST_AUTO_TEST_CASE(fixed_joints_consume_no_q) {
    const auto chain = build_chain(scara_chain());
    BOOST_CHECK_EQUAL(chain.jacobian(xvector<>{0.0, 0.0, 0.0, 0.0}).shape(1), 4U);
    BOOST_CHECK_THROW(static_cast<void>(chain.jacobian(xvector<>{0.0, 0.0, 0.0, 0.0, 0.0})), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.end_position(xvector<>{0.0, 0.0, 0.0})), std::invalid_argument);
}

// We never consult a position limit, so the representative a driver reports for an angle does not
// matter. A continuous joint is the one a driver is liable to report as an accumulated angle.
BOOST_AUTO_TEST_CASE(indifferent_to_full_turn_offsets) {
    const auto chain = build_chain(scara_chain());
    const xvector<> q = {0.4, -1.1, 0.12, 0.7};
    const xvector<> q_offset = {0.4 + (4.0 * std::numbers::pi), -1.1 - (2.0 * std::numbers::pi), 0.12, 0.7 + (6.0 * std::numbers::pi)};
    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q), chain.jacobian(q_offset)), 1e-13);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite validate forward_kinematics, which reports a frame's position and the
// linear velocity and acceleration of its origin for a given joint velocity and acceleration.
BOOST_AUTO_TEST_SUITE(forward_kinematics_tests)

BOOST_AUTO_TEST_CASE(twolink_typical) {
    check_forward_kinematics(twolink_chain(), {0.3, -0.7}, {1.0, 0.5}, {0.2, -0.4});
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_typical) {
    check_forward_kinematics(threelink_with_spacers_chain(), {0.3, -0.5, 0.8}, {0.6, 0.1, -0.7}, {-0.2, 0.4, 0.3});
}

BOOST_AUTO_TEST_CASE(sixdof_typical) {
    check_forward_kinematics(
        sixdof_arm_chain(), {0.1, -0.5, 1.2, -0.3, 0.6, -0.1}, {0.2, 0.7, -0.4, 0.5, -0.6, 0.3}, {-0.1, 0.3, 0.2, -0.5, 0.4, -0.2});
}

BOOST_AUTO_TEST_CASE(rotated_origins_typical) {
    check_forward_kinematics(rotated_origins_chain(), {0.4, -0.9, 0.3}, {0.6, -0.2, 0.5}, {0.1, 0.4, -0.3});
}

BOOST_AUTO_TEST_CASE(with_prismatic_joints) {
    check_forward_kinematics(scara_chain(), {0.4, -1.1, 0.12, 0.7}, {0.3, -0.5, 0.8, 0.2}, {0.1, 0.2, -0.4, 0.3});
    check_forward_kinematics(mixed_chain(), {0.3, -0.6, 0.2, 1.1}, {0.5, 0.4, -0.7, 0.2}, {-0.3, 0.1, 0.2, 0.6});
}

BOOST_AUTO_TEST_CASE(every_frame_matches_its_prefix_chain) {
    check_forward_kinematics_at_every_frame(threelink_with_spacers_chain(), {0.3, -0.5, 0.8}, {0.6, 0.1, -0.7}, {-0.2, 0.4, 0.3});
    check_forward_kinematics_at_every_frame(
        sixdof_arm_chain(), {0.1, -0.5, 1.2, -0.3, 0.6, -0.1}, {0.2, 0.7, -0.4, 0.5, -0.6, 0.3}, {-0.1, 0.3, 0.2, -0.5, 0.4, -0.2});
    check_forward_kinematics_at_every_frame(rotated_origins_chain(), {0.4, -0.9, 0.3}, {0.6, -0.2, 0.5}, {0.1, 0.4, -0.3});
    check_forward_kinematics_at_every_frame(mixed_chain(), {0.3, -0.6, 0.2, 1.1}, {0.5, 0.4, -0.7, 0.2}, {-0.3, 0.1, 0.2, 0.6});
}

// A workspace is storage reused from call to call, so reuse is where it could go wrong. One
// workspace is carried through every frame of chains of different lengths, a longer one after a
// shorter so that its storage has to grow, and each result must be exactly what a call without
// it gives.
BOOST_AUTO_TEST_CASE(reused_workspace_matches_fresh_call) {
    chain::workspace storage;

    const auto check = [&storage](const test_chain& joints, const xvector<>& q, const xvector<>& q_dot, const xvector<>& q_ddot) {
        const auto c = build_chain(joints);
        for (std::size_t frame = 0; frame < c.size(); ++frame) {
            BOOST_TEST_CONTEXT("frame " << frame) {
                const auto fresh = c.forward_kinematics(frame, q, q_dot, q_ddot);
                const auto reused = c.forward_kinematics(frame, q, q_dot, q_ddot, storage);
                BOOST_CHECK(reused.position == fresh.position);
                BOOST_CHECK(reused.linear_velocity == fresh.linear_velocity);
                BOOST_CHECK(reused.linear_acceleration == fresh.linear_acceleration);
            }
        }
    };

    check(twolink_chain(), {0.3, -0.7}, {1.0, 0.5}, {0.2, -0.4});
    check(sixdof_arm_chain(), {0.1, -0.5, 1.2, -0.3, 0.6, -0.1}, {0.2, 0.7, -0.4, 0.5, -0.6, 0.3}, {-0.1, 0.3, 0.2, -0.5, 0.4, -0.2});
    check(mixed_chain(), {0.3, -0.6, 0.2, 1.1}, {0.5, 0.4, -0.7, 0.2}, {-0.3, 0.1, 0.2, 0.6});
    check(sixdof_arm_chain(), {-0.4, 0.9, -0.2, 1.1, -0.7, 0.5}, {-0.3, 0.2, 0.6, -0.1, 0.4, -0.5}, {0.2, -0.1, 0.3, 0.4, -0.3, 0.1});
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite validate trajectory::tcp_limits::from, whose evaluator reports the TCP's
// linear speed per unit of arc velocity, ||J_v(q) * f'||, and its derivative along the path.
BOOST_AUTO_TEST_SUITE(tcp_limits_from_tests)

BOOST_AUTO_TEST_CASE(twolink_typical) {
    check_linear_speed_per_arc_velocity(twolink_chain(), {0.3, -0.7}, {1.0, 0.5}, {0.2, -0.4});
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_typical) {
    check_linear_speed_per_arc_velocity(threelink_with_spacers_chain(), {0.3, -0.5, 0.8}, {0.6, 0.1, -0.7}, {-0.2, 0.4, 0.3});
}

BOOST_AUTO_TEST_CASE(sixdof_typical) {
    check_linear_speed_per_arc_velocity(
        sixdof_arm_chain(), {0.1, -0.5, 1.2, -0.3, 0.6, -0.1}, {0.2, 0.7, -0.4, 0.5, -0.6, 0.3}, {-0.1, 0.3, 0.2, -0.5, 0.4, -0.2});
}

BOOST_AUTO_TEST_CASE(rotated_origins_typical) {
    check_linear_speed_per_arc_velocity(rotated_origins_chain(), {0.4, -0.9, 0.3}, {0.6, -0.2, 0.5}, {0.1, 0.4, -0.3});
}

BOOST_AUTO_TEST_CASE(with_prismatic_joints) {
    check_linear_speed_per_arc_velocity(scara_chain(), {0.4, -1.1, 0.12, 0.7}, {0.3, -0.5, 0.8, 0.2}, {0.1, 0.2, -0.4, 0.3});
    check_linear_speed_per_arc_velocity(mixed_chain(), {0.3, -0.6, 0.2, 1.1}, {0.5, 0.4, -0.7, 0.2}, {-0.3, 0.1, 0.2, 0.6});
}

BOOST_AUTO_TEST_CASE(binds_the_cap_and_answers_repeatably) {
    const auto limits = trajectory::tcp_limits::from(build_chain(sixdof_arm_chain()), 0.5);
    const xvector<> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};
    const xvector<> q_prime = {0.2, 0.7, -0.4, 0.5, -0.6, 0.3};
    const xvector<> q_double_prime = {-0.1, 0.3, 0.2, -0.5, 0.4, -0.2};

    const auto first = limits.evaluator(q, q_prime, q_double_prime);
    const auto second = limits.evaluator(q, q_prime, q_double_prime);
    BOOST_REQUIRE(first.has_value());
    BOOST_REQUIRE(second.has_value());
    BOOST_CHECK_EQUAL(first->value, second->value);
    BOOST_CHECK_EQUAL(first->arc_length_derivative, second->arc_length_derivative);

    BOOST_CHECK_EQUAL(limits.max_linear_speed, 0.5);
}

// Where the TCP does not move relative to the path, ||J*f'|| is zero and its derivative
// dot(v, a) / ||v|| is 0 / 0, which does not exist, so the evaluator returns nothing. That happens
// when the path tangent is zero, and also when the joints move but along a direction the Jacobian
// sends to zero, as the two-link arm's do when stretched straight out with the elbow turning back
// twice as fast as the shoulder turns forward.
BOOST_AUTO_TEST_CASE(singular_ratio_yields_nothing) {
    const auto limits = trajectory::tcp_limits::from(build_chain(twolink_chain()), 0.5);
    BOOST_CHECK(!limits.evaluator(xvector<>{0.3, -0.7}, xvector<>{0.0, 0.0}, xvector<>{0.2, -0.4}).has_value());
    BOOST_CHECK(!limits.evaluator(xvector<>{0.0, 0.0}, xvector<>{1.0, -2.0}, xvector<>{0.2, -0.4}).has_value());
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite validate chain::builder and its inverse, visit_joints: that
// build() normalizes what it may and refuses what it cannot represent, and that a chain reports
// its joints exactly as it was given them.
BOOST_AUTO_TEST_SUITE(builder_tests)

// build() normalizes, so a quaternion and an axis given at some other length describe the same
// chain.
BOOST_AUTO_TEST_CASE(normalizes_quaternions_and_axes) {
    auto scaled = mixed_chain();
    for (auto& j : scaled) {
        for (auto& c : j.wxyz) {
            c *= 3.7;
        }
        for (auto& c : j.axis) {
            c *= 0.25;
        }
    }
    const xvector<> q = {0.3, -0.6, 0.2, 1.1};
    BOOST_CHECK_SMALL(matrix_diff_norm(compute_jacobian(mixed_chain(), q), compute_jacobian(scaled, q)), 1e-14);
}

BOOST_AUTO_TEST_CASE(build_refuses_what_it_cannot_represent) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const auto refuses = [](const test_chain& joints) { BOOST_CHECK_THROW(static_cast<void>(build_chain(joints)), std::invalid_argument); };

    refuses({});
    refuses({rotating({}, {0, 0, 0})});
    refuses({translating({}, {0, 0, 0})});
    refuses({rotating({}, {0, nan, 1})});
    refuses({translating({}, {inf, 0, 0})});
    refuses({fixed({}, {0, 0, 0, 0})});
    refuses({fixed({}, {nan, 0, 0, 0})});
    refuses({fixed({}, {1e200, 1e200, 0, 0})});
    refuses({fixed({0, inf, 0})});

    // A fixed joint has no axis to validate, and a fixed-only chain is legal: it has no q at all.
    const auto fixed_only = build_chain({fixed({1, 2, 3})});
    BOOST_CHECK_SMALL(vec3_diff_norm(fixed_only.end_position(xvector<>(xt::zeros<double>({std::size_t{0}}))), {1, 2, 3}), 1e-15);
}

// visit_joints reports what the builder was given, not what it derived, so unnormalized
// quaternions and axes come back as they went in, and feeding the visits back into a builder
// rebuilds a chain that evaluates identically.
BOOST_AUTO_TEST_CASE(visit_joints_reports_joints_as_given) {
    auto given = mixed_chain();
    given[1].wxyz = {2.0, 0.1, -0.3, 0.4};
    given[2].axis = {0.0, 3.0, 4.0};

    test_chain reported;
    build_chain(given).visit_joints([&]<typename... Axis>(const chain::joint_origin& origin, const Axis&... axis) {
        if constexpr (sizeof...(Axis) == 0) {
            reported.push_back(fixed(origin.translation, origin.rotation));
        } else if constexpr ((std::is_same_v<Axis, chain::rotation_axis> && ...)) {
            reported.push_back(rotating(origin.translation, axis.direction..., origin.rotation));
        } else {
            reported.push_back(translating(origin.translation, axis.direction..., origin.rotation));
        }
    });

    BOOST_REQUIRE_EQUAL(reported.size(), given.size());
    for (std::size_t i = 0; i < given.size(); ++i) {
        BOOST_CHECK(reported[i].kind == given[i].kind);
        BOOST_CHECK(reported[i].xyz == given[i].xyz);
        BOOST_CHECK(reported[i].wxyz == given[i].wxyz);
        BOOST_CHECK(reported[i].axis == given[i].axis);
    }

    const xvector<> q = {0.3, -0.6, 0.2, 1.1};
    BOOST_CHECK_EQUAL(matrix_diff_norm(compute_jacobian(given, q), compute_jacobian(reported, q)), 0.0);
}

BOOST_AUTO_TEST_SUITE_END()
