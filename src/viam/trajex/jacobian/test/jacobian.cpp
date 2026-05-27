// Tests for the simplified jacobian module (Option B: plain-return API,
// model table consumed as an xt::xarray<double> tensor in the
// viam::sdk::ModelTable format).

#include <viam/trajex/jacobian/jacobian.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <boost/test/unit_test.hpp>

#include <viam/sdk/common/exception.hpp>

namespace {

using viam::trajex::jacobian::compute_jacobian;
using viam::trajex::jacobian::forward_kinematics;

// Joint-type encodings for column 9 of the model-table tensor.
// These match viam::sdk::JointType: revolute=0, continuous=1,
// prismatic=2, fixed=3.
constexpr double kRev = 0.0;
constexpr double kCont = 1.0;
constexpr double kPris = 2.0;
constexpr double kFix = 3.0;

xt::xarray<double> make_table(std::initializer_list<std::initializer_list<double>> rows) {
    const std::size_t n = rows.size();
    xt::xarray<double> t = xt::zeros<double>({n, std::size_t{10}});
    std::size_t i = 0;
    for (const auto& r : rows) {
        std::size_t j = 0;
        for (double v : r) {
            t(i, j++) = v;
        }
        ++i;
    }
    return t;
}

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

// Two 1m planar links rotating about z, ending in a 1m fixed flange.
xt::xarray<double> twolink_table() {
    return xt::xarray<double>{
        {0, 0, 0, 0, 0, 0, 0, 0, 1, kRev},
        {1, 0, 0, 0, 0, 0, 0, 0, 1, kRev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, kFix},
    };
}

// 3 revolute joints separated by fixed 1m spacers, ending in a 1m flange.
xt::xarray<double> threelink_with_spacers_table() {
    return xt::xarray<double>{
        {0, 0, 0, 0, 0, 0, 0, 0, 1, kRev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, kFix},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, kRev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, kFix},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, kRev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, kFix},
    };
}

// 6-revolute spatial chain mimicking a UR-like structure.
xt::xarray<double> sixdof_arm_table() {
    return xt::xarray<double>{
        {0,   0,    0.10, 0,    0,    0,    0,   0,   1,   kRev},
        {0,   0,    0.15, 0,    0,    0,    0,   1,   0,   kRev},
        {0.4, 0,    0,    0,    0,    0,    0,   1,   0,   kRev},
        {0.4, 0,    0,    0,    0,    0,    0,   1,   0,   kRev},
        {0,   0,    0.10, 0,    0,    0,    1,   0,   0,   kRev},
        {0,   0,    0.10, 0,    0,    0,    0,   0,   1,   kRev},
        {0,   0,    0.05, 0,    0,    0,    0,   0,   0,   kFix},
    };
}

// Numerical geometric Jacobian via central differences on forward_kinematics.
xt::xarray<double> numerical_jacobian(const xt::xarray<double>& table,
                                       const xt::xarray<double>& q,
                                       double delta = 1e-7) {
    const std::size_t n_actuated = q.size();
    xt::xarray<double> J_num = xt::zeros<double>({std::size_t{6}, n_actuated});

    for (std::size_t i = 0; i < n_actuated; ++i) {
        xt::xarray<double> q_plus = q;
        xt::xarray<double> q_minus = q;
        q_plus(i) += delta;
        q_minus(i) -= delta;

        const Eigen::Matrix4d Tp = forward_kinematics(table, q_plus);
        const Eigen::Matrix4d Tm = forward_kinematics(table, q_minus);

        for (std::size_t r = 0; r < 3; ++r) {
            J_num(r, i) = (Tp(r, 3) - Tm(r, 3)) / (2.0 * delta);
        }

        // dR = R_plus * R_minus^T, extract omega from skew-symmetric part.
        const Eigen::Matrix3d dR =
            Tp.block<3, 3>(0, 0) * Tm.block<3, 3>(0, 0).transpose();
        const double scale = 1.0 / (2.0 * delta);
        J_num(3, i) = 0.5 * (dR(2, 1) - dR(1, 2)) * scale;
        J_num(4, i) = 0.5 * (dR(0, 2) - dR(2, 0)) * scale;
        J_num(5, i) = 0.5 * (dR(1, 0) - dR(0, 1)) * scale;
    }
    return J_num;
}

void check_matches_numerical(const xt::xarray<double>& table,
                              const xt::xarray<double>& q,
                              double tol = 1e-6) {
    const auto J = compute_jacobian(table, q);
    const auto J_num = numerical_jacobian(table, q);
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_num), tol);
}

}  // namespace

// ============================================================================
// Velocity tests: J * q_dot produces the expected Cartesian velocity.
// ============================================================================

BOOST_AUTO_TEST_SUITE(jacobian_velocity_tests)

BOOST_AUTO_TEST_CASE(twolink_base_spin_extended) {
    const auto table = twolink_table();
    const xt::xarray<double> q = xt::zeros<double>({std::size_t{2}});
    const xt::xarray<double> q_dot = {1.0, 0.0};

    const auto J = compute_jacobian(table, q);
    const auto t = J_times_qdot(J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {0.0, 2.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_base_spin_rotated) {
    const auto table = twolink_table();
    const xt::xarray<double> q = {std::numbers::pi / 2.0, 0.0};
    const xt::xarray<double> q_dot = {1.0, 0.0};

    const auto J = compute_jacobian(table, q);
    const auto t = J_times_qdot(J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {-2.0, 0.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_base_spin_bent) {
    const auto table = twolink_table();
    const xt::xarray<double> q = {0.0, std::numbers::pi / 2.0};
    const xt::xarray<double> q_dot = {1.0, 0.0};

    const auto J = compute_jacobian(table, q);
    const auto t = J_times_qdot(J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {-1.0, 1.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()


// ============================================================================
// Ground-truth tests: hand-computed Jacobian values.
// ============================================================================

BOOST_AUTO_TEST_SUITE(jacobian_ground_truth_tests)

BOOST_AUTO_TEST_CASE(twolink_zero) {
    const auto table = twolink_table();
    const xt::xarray<double> q = xt::zeros<double>({std::size_t{2}});
    const auto J = compute_jacobian(table, q);

    const xt::xarray<double> J_expected = {
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
    const auto table = twolink_table();
    const xt::xarray<double> q = {std::numbers::pi / 2.0, 0.0};
    const auto J = compute_jacobian(table, q);

    const xt::xarray<double> J_expected = {
        {-2.0, -1.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 1.0,  1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_q2_ninety) {
    const auto table = twolink_table();
    const xt::xarray<double> q = {0.0, std::numbers::pi / 2.0};
    const auto J = compute_jacobian(table, q);

    const xt::xarray<double> J_expected = {
        {-1.0, -1.0},
        { 1.0,  0.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 0.0,  0.0},
        { 1.0,  1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()


// ============================================================================
// FK basic checks. The numerical-consistency suite depends on FK being
// correct, so verify a few easy cases directly.
// ============================================================================

BOOST_AUTO_TEST_SUITE(fk_tests)

BOOST_AUTO_TEST_CASE(twolink_zero_ee_at_2_0_0) {
    const auto table = twolink_table();
    const Eigen::Matrix4d T = forward_kinematics(table, xt::zeros<double>({std::size_t{2}}));
    BOOST_CHECK_CLOSE(T(0, 3), 2.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(1, 3)), 1e-12);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(twolink_q1_pi_over_2_ee_at_0_2_0) {
    const auto table = twolink_table();
    const xt::xarray<double> q = {std::numbers::pi / 2.0, 0.0};
    const Eigen::Matrix4d T = forward_kinematics(table, q);
    BOOST_CHECK_SMALL(std::abs(T(0, 3)), 1e-9);
    BOOST_CHECK_CLOSE(T(1, 3), 2.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(twolink_q2_pi_over_2_ee_at_1_1_0) {
    const auto table = twolink_table();
    const xt::xarray<double> q = {0.0, std::numbers::pi / 2.0};
    const Eigen::Matrix4d T = forward_kinematics(table, q);
    BOOST_CHECK_CLOSE(T(0, 3), 1.0, 1e-9);
    BOOST_CHECK_CLOSE(T(1, 3), 1.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()


// ============================================================================
// Numerical consistency: analytical Jacobian == central-difference Jacobian.
// ============================================================================

BOOST_AUTO_TEST_SUITE(jacobian_numerical_consistency_tests)

BOOST_AUTO_TEST_CASE(twolink_zero) {
    check_matches_numerical(twolink_table(), xt::zeros<double>({std::size_t{2}}));
}

BOOST_AUTO_TEST_CASE(twolink_q1_q2_forty_five) {
    const xt::xarray<double> q = {std::numbers::pi / 4.0, std::numbers::pi / 4.0};
    check_matches_numerical(twolink_table(), q);
}

BOOST_AUTO_TEST_CASE(twolink_folded_back) {
    const xt::xarray<double> q = {0.0, std::numbers::pi};
    check_matches_numerical(twolink_table(), q);
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_zero) {
    check_matches_numerical(threelink_with_spacers_table(),
                            xt::zeros<double>({std::size_t{3}}));
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_typical) {
    const xt::xarray<double> q = {0.3, -0.5, 0.8};
    check_matches_numerical(threelink_with_spacers_table(), q);
}

BOOST_AUTO_TEST_CASE(sixdof_zero) {
    check_matches_numerical(sixdof_arm_table(), xt::zeros<double>({std::size_t{6}}));
}

BOOST_AUTO_TEST_CASE(sixdof_typical) {
    const xt::xarray<double> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};
    check_matches_numerical(sixdof_arm_table(), q);
}

BOOST_AUTO_TEST_SUITE_END()


// ============================================================================
// Error handling.
// ============================================================================

BOOST_AUTO_TEST_SUITE(jacobian_error_tests)

BOOST_AUTO_TEST_CASE(rejects_wrong_q_size) {
    const auto table = twolink_table();
    const xt::xarray<double> q_bad = {0.0, 0.0, 0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q_bad), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_continuous_joint) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, kCont}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_prismatic_joint) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, kPris}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_zero_axis_for_revolute) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 0, kRev}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

// The SDK throws viam::sdk::Exception on malformed tensor input. Verify
// we don't swallow or wrap it.
BOOST_AUTO_TEST_CASE(propagates_sdk_exception_on_bad_shape) {
    xt::xarray<double> bad = xt::zeros<double>({std::size_t{1}, std::size_t{9}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(bad, q), viam::sdk::Exception);
}

BOOST_AUTO_TEST_SUITE_END()
