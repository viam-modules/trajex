// Jacobian + forward-kinematics tests.
//
// Three kinds of coverage:
//   1. Velocity projection: J * q_dot produces the expected Cartesian
//      velocity for hand-chosen 2-link configurations.
//   2. Ground truth: the full 6xN Jacobian matches a hand-computed
//      reference for simple 2-link configurations.
//   3. Numerical consistency: the analytical Jacobian matches a
//      central-difference numerical Jacobian for 2-link and UR20 configs.

#include <viam/trajex/jacobian/forward_kinematics.hpp>
#include <viam/trajex/jacobian/jacobian.hpp>
#include <viam/trajex/jacobian/model.hpp>

#include <array>
#include <cmath>
#include <numbers>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif
#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xview.hpp>
#endif

#include <boost/test/unit_test.hpp>

namespace {

using viam::trajex::jacobian::compute_forward_kinematics;
using viam::trajex::jacobian::compute_jacobian;
using viam::trajex::jacobian::data;
using viam::trajex::jacobian::model;

// Frobenius norm of (A - B). Both must have identical shape.
double matrix_diff_norm(const xt::xarray<double>& A, const xt::xarray<double>& B) {
    double s = 0.0;
    auto a = A.begin();
    auto b = B.begin();
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
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// J (6 x N) * q_dot (N) -> 6-vector returned as two 3-arrays (linear, angular).
struct twist {
    std::array<double, 3> v;
    std::array<double, 3> w;
};
twist J_times_qdot(const xt::xarray<double>& J, const xt::xarray<double>& q_dot) {
    twist t{{0, 0, 0}, {0, 0, 0}};
    const std::size_t n = q_dot.size();
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < 3; ++i) t.v[i] += J(i, j) * q_dot(j);
        for (std::size_t i = 0; i < 3; ++i) t.w[i] += J(3 + i, j) * q_dot(j);
    }
    return t;
}

// Numerical geometric Jacobian via central differences on FK.
xt::xarray<double> numerical_jacobian(const model& m, const xt::xarray<double>& q, double delta = 1e-7) {
    const std::size_t n = m.joints.size();
    xt::xarray<double> J_num = xt::zeros<double>({std::size_t{6}, n});
    data d_plus(m);
    data d_minus(m);

    for (std::size_t i = 0; i < n; ++i) {
        xt::xarray<double> q_plus = q;
        xt::xarray<double> q_minus = q;
        q_plus(i) += delta;
        q_minus(i) -= delta;

        compute_forward_kinematics(m, q_plus, d_plus);
        compute_forward_kinematics(m, q_minus, d_minus);

        const auto& Tp = d_plus.end_effector_transform;
        const auto& Tm = d_minus.end_effector_transform;
        for (std::size_t r = 0; r < 3; ++r) {
            J_num(r, i) = (Tp(r, 3) - Tm(r, 3)) / (2.0 * delta);
        }

        // dR = R_plus * R_minus^T, then extract omega from skew-symmetric part.
        std::array<std::array<double, 3>, 3> dR{};
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t c = 0; c < 3; ++c) {
                double s = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    s += Tp(r, k) * Tm(c, k);  // R_minus^T(k,c) == Tm(c,k)
                }
                dR[r][c] = s;
            }
        }
        const double scale = 1.0 / (4.0 * delta);
        J_num(3, i) = (dR[2][1] - dR[1][2]) * scale;
        J_num(4, i) = (dR[0][2] - dR[2][0]) * scale;
        J_num(5, i) = (dR[1][0] - dR[0][1]) * scale;
    }
    return J_num;
}

model twolink() {
    // Two 1m links, both revolute about Z, no offsets.
    return model{
        "2link_planar",
        {
            {"joint1", 0.0, 0.0, 1.0, 0.0},
            {"joint2", 0.0, 0.0, 1.0, 0.0},
        },
    };
}

model ur20() {
    // DH derived from urdfs/ur20FK.urdf via viam-cpp-sdk's urdf_to_dh_params,
    // pruned to end at ft_frame to match the URDF parser's end-effector choice.
    return model{
        "UR20",
        {
            {"shoulder_pan_joint",  0.23630000000000001,  -1.2246467991473532e-16,  0.0,                      -1.570796327},
            {"shoulder_lift_joint", 0.0,                  -0.0,                     0.86199999999999999,       0.0},
            {"elbow_joint",         5.9732854069808984e-18, -1.2188533337963318e-18, 0.72870000000000001,      0.0},
            {"wrist_1_joint",       0.20100000000000004,   1.2188533337963318e-18, -2.4615400662861804e-17,   -1.570796327},
            {"wrist_2_joint",       0.1593,               -1.224646798896174e-16,   8.0025735243201159e-27,    1.570796327},
            {"wrist_3_joint",       0.15429999999999999,  -3.1415926535897931,      3.8756978341028043e-27,    3.1415926535897931},
        },
    };
}

}  // namespace

BOOST_AUTO_TEST_SUITE(jacobian_velocity_tests)

// q=[0,0], q_dot=[1,0]: base spinning at 1 rad/s, EE is 2m along X.
// EE traces a circle of radius 2 about Z -> v = [0, 2, 0], w = [0, 0, 1].
BOOST_AUTO_TEST_CASE(twolink_base_spin_extended) {
    const model m = twolink();
    const xt::xarray<double> q = xt::zeros<double>({std::size_t{2}});
    const xt::xarray<double> q_dot = {1.0, 0.0};

    data d(m);
    compute_jacobian(m, q, d);
    const auto t = J_times_qdot(d.J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {0.0, 2.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

// q=[pi/2, 0], q_dot=[1,0]: base spinning, arm along +Y, EE at (0,2,0).
// EE sweeps in -X -> v = [-2, 0, 0], w = [0, 0, 1].
BOOST_AUTO_TEST_CASE(twolink_base_spin_rotated) {
    const model m = twolink();
    const xt::xarray<double> q = {std::numbers::pi / 2.0, 0.0};
    const xt::xarray<double> q_dot = {1.0, 0.0};

    data d(m);
    compute_jacobian(m, q, d);
    const auto t = J_times_qdot(d.J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {-2.0, 0.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

// q=[0, pi/2], q_dot=[1,0]: base spinning, link2 bent 90deg, EE at (1,1,0).
// -> v = [-1, 1, 0], w = [0, 0, 1].
BOOST_AUTO_TEST_CASE(twolink_base_spin_bent) {
    const model m = twolink();
    const xt::xarray<double> q = {0.0, std::numbers::pi / 2.0};
    const xt::xarray<double> q_dot = {1.0, 0.0};

    data d(m);
    compute_jacobian(m, q, d);
    const auto t = J_times_qdot(d.J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {-1.0, 1.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(jacobian_ground_truth_tests)

// q=[0,0]: links along X, EE at (2,0,0).
BOOST_AUTO_TEST_CASE(twolink_zero) {
    const model m = twolink();
    const xt::xarray<double> q = xt::zeros<double>({std::size_t{2}});
    data d(m);
    compute_jacobian(m, q, d);

    const xt::xarray<double> J_expected = {
        {0.0, 0.0},
        {2.0, 1.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };

    BOOST_CHECK_SMALL(matrix_diff_norm(d.J, J_expected), 1e-10);
}

// q=[pi/2, 0]: links along Y, EE at (0,2,0).
BOOST_AUTO_TEST_CASE(twolink_q1_ninety) {
    const model m = twolink();
    const xt::xarray<double> q = {std::numbers::pi / 2.0, 0.0};
    data d(m);
    compute_jacobian(m, q, d);

    const xt::xarray<double> J_expected = {
        {-2.0, -1.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 1.0,  1.0},
    };

    BOOST_CHECK_SMALL(matrix_diff_norm(d.J, J_expected), 1e-10);
}

// q=[0, pi/2]: link1 along X, link2 along Y, EE at (1,1,0).
BOOST_AUTO_TEST_CASE(twolink_q2_ninety) {
    const model m = twolink();
    const xt::xarray<double> q = {0.0, std::numbers::pi / 2.0};
    data d(m);
    compute_jacobian(m, q, d);

    const xt::xarray<double> J_expected = {
        {-1.0, -1.0},
        { 1.0,  0.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 1.0,  1.0},
    };

    BOOST_CHECK_SMALL(matrix_diff_norm(d.J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(jacobian_numerical_consistency_tests)

namespace {

void check_matches_numerical(const model& m, const xt::xarray<double>& q, double tol = 1e-6) {
    data d(m);
    compute_jacobian(m, q, d);
    const xt::xarray<double> J_num = numerical_jacobian(m, q);
    BOOST_CHECK_SMALL(matrix_diff_norm(d.J, J_num), tol);
}

}  // namespace

BOOST_AUTO_TEST_CASE(twolink_zero) {
    check_matches_numerical(twolink(), xt::zeros<double>({std::size_t{2}}));
}

BOOST_AUTO_TEST_CASE(twolink_q1_ninety) {
    const xt::xarray<double> q = {std::numbers::pi / 2.0, 0.0};
    check_matches_numerical(twolink(), q);
}

BOOST_AUTO_TEST_CASE(twolink_q1_q2_forty_five) {
    const xt::xarray<double> q = {std::numbers::pi / 4.0, std::numbers::pi / 4.0};
    check_matches_numerical(twolink(), q);
}

BOOST_AUTO_TEST_CASE(twolink_folded_back) {
    const xt::xarray<double> q = {0.0, std::numbers::pi};
    check_matches_numerical(twolink(), q);
}

BOOST_AUTO_TEST_CASE(ur20_zero) {
    check_matches_numerical(ur20(), xt::zeros<double>({std::size_t{6}}));
}

BOOST_AUTO_TEST_CASE(ur20_small_angles) {
    const xt::xarray<double> q = {0.1, 0.2, -0.1, 0.15, -0.05, 0.1};
    check_matches_numerical(ur20(), q);
}

BOOST_AUTO_TEST_CASE(ur20_typical) {
    const xt::xarray<double> q = {
        0.0, -std::numbers::pi / 2.0, std::numbers::pi / 2.0,
        0.0, std::numbers::pi / 2.0, 0.0};
    check_matches_numerical(ur20(), q);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(jacobian_error_tests)

BOOST_AUTO_TEST_CASE(fk_rejects_wrong_q_size) {
    const model m = twolink();
    data d(m);
    const xt::xarray<double> q_bad = {0.0, 0.0, 0.0};
    BOOST_CHECK_THROW(compute_forward_kinematics(m, q_bad, d), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
