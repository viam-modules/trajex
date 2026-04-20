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

#include <cmath>
#include <numbers>

#include <boost/test/unit_test.hpp>

namespace {

using viam::trajex::jacobian::compute_forward_kinematics;
using viam::trajex::jacobian::compute_jacobian;
using viam::trajex::jacobian::data;
using viam::trajex::jacobian::model;

// Numerical geometric Jacobian via central differences on FK.
Eigen::MatrixXd numerical_jacobian(const model& m, const Eigen::VectorXd& q, double delta = 1e-7) {
    const Eigen::Index n = static_cast<Eigen::Index>(m.joints.size());
    Eigen::MatrixXd J_num(6, n);
    data d_plus(m);
    data d_minus(m);

    for (Eigen::Index i = 0; i < n; ++i) {
        Eigen::VectorXd q_plus = q;
        Eigen::VectorXd q_minus = q;
        q_plus[i] += delta;
        q_minus[i] -= delta;

        compute_forward_kinematics(m, q_plus, d_plus);
        compute_forward_kinematics(m, q_minus, d_minus);

        const Eigen::Vector3d p_plus = d_plus.end_effector_transform.block<3, 1>(0, 3);
        const Eigen::Vector3d p_minus = d_minus.end_effector_transform.block<3, 1>(0, 3);
        J_num.block<3, 1>(0, i) = (p_plus - p_minus) / (2.0 * delta);

        const Eigen::Matrix3d R_plus = d_plus.end_effector_transform.block<3, 3>(0, 0);
        const Eigen::Matrix3d R_minus = d_minus.end_effector_transform.block<3, 3>(0, 0);
        const Eigen::Matrix3d dR = R_plus * R_minus.transpose();

        Eigen::Vector3d omega;
        omega << (dR(2, 1) - dR(1, 2)), (dR(0, 2) - dR(2, 0)), (dR(1, 0) - dR(0, 1));
        omega /= (4.0 * delta);
        J_num.block<3, 1>(3, i) = omega;
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
    Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd q_dot(2);
    q_dot << 1.0, 0.0;

    data d(m);
    compute_jacobian(m, q, d);

    const Eigen::Vector3d v = d.J.topRows(3) * q_dot;
    const Eigen::Vector3d omega = d.J.bottomRows(3) * q_dot;

    BOOST_CHECK_SMALL((v - Eigen::Vector3d(0.0, 2.0, 0.0)).norm(), 1e-10);
    BOOST_CHECK_SMALL((omega - Eigen::Vector3d(0.0, 0.0, 1.0)).norm(), 1e-10);
}

// q=[pi/2, 0], q_dot=[1,0]: base spinning, arm along +Y, EE at (0,2,0).
// EE sweeps in -X -> v = [-2, 0, 0], w = [0, 0, 1].
BOOST_AUTO_TEST_CASE(twolink_base_spin_rotated) {
    const model m = twolink();
    Eigen::VectorXd q(2);
    q << std::numbers::pi / 2.0, 0.0;
    Eigen::VectorXd q_dot(2);
    q_dot << 1.0, 0.0;

    data d(m);
    compute_jacobian(m, q, d);

    const Eigen::Vector3d v = d.J.topRows(3) * q_dot;
    const Eigen::Vector3d omega = d.J.bottomRows(3) * q_dot;

    BOOST_CHECK_SMALL((v - Eigen::Vector3d(-2.0, 0.0, 0.0)).norm(), 1e-10);
    BOOST_CHECK_SMALL((omega - Eigen::Vector3d(0.0, 0.0, 1.0)).norm(), 1e-10);
}

// q=[0, pi/2], q_dot=[1,0]: base spinning, link2 bent 90deg, EE at (1,1,0).
// -> v = [-1, 1, 0], w = [0, 0, 1].
BOOST_AUTO_TEST_CASE(twolink_base_spin_bent) {
    const model m = twolink();
    Eigen::VectorXd q(2);
    q << 0.0, std::numbers::pi / 2.0;
    Eigen::VectorXd q_dot(2);
    q_dot << 1.0, 0.0;

    data d(m);
    compute_jacobian(m, q, d);

    const Eigen::Vector3d v = d.J.topRows(3) * q_dot;
    const Eigen::Vector3d omega = d.J.bottomRows(3) * q_dot;

    BOOST_CHECK_SMALL((v - Eigen::Vector3d(-1.0, 1.0, 0.0)).norm(), 1e-10);
    BOOST_CHECK_SMALL((omega - Eigen::Vector3d(0.0, 0.0, 1.0)).norm(), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(jacobian_ground_truth_tests)

// q=[0,0]: links along X, EE at (2,0,0).
BOOST_AUTO_TEST_CASE(twolink_zero) {
    const model m = twolink();
    Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
    data d(m);
    compute_jacobian(m, q, d);

    Eigen::MatrixXd J_expected(6, 2);
    J_expected << 0.0, 0.0,
                  2.0, 1.0,
                  0.0, 0.0,
                  0.0, 0.0,
                  0.0, 0.0,
                  1.0, 1.0;

    BOOST_CHECK_SMALL((d.J - J_expected).norm(), 1e-10);
}

// q=[pi/2, 0]: links along Y, EE at (0,2,0).
BOOST_AUTO_TEST_CASE(twolink_q1_ninety) {
    const model m = twolink();
    Eigen::VectorXd q(2);
    q << std::numbers::pi / 2.0, 0.0;
    data d(m);
    compute_jacobian(m, q, d);

    Eigen::MatrixXd J_expected(6, 2);
    J_expected << -2.0, -1.0,
                   0.0,  0.0,
                   0.0,  0.0,
                   0.0,  0.0,
                   0.0,  0.0,
                   1.0,  1.0;

    BOOST_CHECK_SMALL((d.J - J_expected).norm(), 1e-10);
}

// q=[0, pi/2]: link1 along X, link2 along Y, EE at (1,1,0).
BOOST_AUTO_TEST_CASE(twolink_q2_ninety) {
    const model m = twolink();
    Eigen::VectorXd q(2);
    q << 0.0, std::numbers::pi / 2.0;
    data d(m);
    compute_jacobian(m, q, d);

    Eigen::MatrixXd J_expected(6, 2);
    J_expected << -1.0, -1.0,
                   1.0,  0.0,
                   0.0,  0.0,
                   0.0,  0.0,
                   0.0,  0.0,
                   1.0,  1.0;

    BOOST_CHECK_SMALL((d.J - J_expected).norm(), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(jacobian_numerical_consistency_tests)

namespace {

void check_matches_numerical(const model& m, const Eigen::VectorXd& q, double tol = 1e-6) {
    data d(m);
    compute_jacobian(m, q, d);
    const Eigen::MatrixXd J_num = numerical_jacobian(m, q);
    BOOST_CHECK_SMALL((d.J - J_num).norm(), tol);
}

}  // namespace

BOOST_AUTO_TEST_CASE(twolink_zero) {
    check_matches_numerical(twolink(), Eigen::VectorXd::Zero(2));
}

BOOST_AUTO_TEST_CASE(twolink_q1_ninety) {
    Eigen::VectorXd q(2);
    q << std::numbers::pi / 2.0, 0.0;
    check_matches_numerical(twolink(), q);
}

BOOST_AUTO_TEST_CASE(twolink_q1_q2_forty_five) {
    Eigen::VectorXd q(2);
    q << std::numbers::pi / 4.0, std::numbers::pi / 4.0;
    check_matches_numerical(twolink(), q);
}

BOOST_AUTO_TEST_CASE(twolink_folded_back) {
    Eigen::VectorXd q(2);
    q << 0.0, std::numbers::pi;
    check_matches_numerical(twolink(), q);
}

BOOST_AUTO_TEST_CASE(ur20_zero) {
    check_matches_numerical(ur20(), Eigen::VectorXd::Zero(6));
}

BOOST_AUTO_TEST_CASE(ur20_small_angles) {
    Eigen::VectorXd q(6);
    q << 0.1, 0.2, -0.1, 0.15, -0.05, 0.1;
    check_matches_numerical(ur20(), q);
}

BOOST_AUTO_TEST_CASE(ur20_typical) {
    Eigen::VectorXd q(6);
    q << 0.0, -std::numbers::pi / 2.0, std::numbers::pi / 2.0, 0.0, std::numbers::pi / 2.0, 0.0;
    check_matches_numerical(ur20(), q);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(jacobian_error_tests)

BOOST_AUTO_TEST_CASE(fk_rejects_wrong_q_size) {
    const model m = twolink();
    data d(m);
    Eigen::VectorXd q_bad(3);
    q_bad << 0.0, 0.0, 0.0;
    BOOST_CHECK_THROW(compute_forward_kinematics(m, q_bad, d), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
