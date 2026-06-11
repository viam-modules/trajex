#include <viam/trajex/jacobian/jacobian.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <boost/test/unit_test.hpp>

namespace {

using viam::trajex::jacobian::kinematic_chain;

// One-shot convenience for tests: parse the tensor and evaluate at q.
xt::xarray<double> compute_jacobian(const xt::xarray<double>& table, const xt::xarray<double>& q) {
    return kinematic_chain::from(table).jacobian(q);
}

// Joint-type encodings for column 9 of the model-table tensor; values match
// viam::sdk::ModelTable::JointType.
constexpr double k_rev = 0.0;
constexpr double k_cont = 1.0;
constexpr double k_pris = 2.0;
constexpr double k_fix = 3.0;

xt::xarray<double> make_table(std::initializer_list<std::initializer_list<double>> rows) {
    const std::size_t n = rows.size();
    xt::xarray<double> t = xt::zeros<double>({n, std::size_t{10}});
    std::size_t i = 0;
    for (const auto& r : rows) {
        std::size_t j = 0;
        for (const double v : r) {
            t(i, j++) = v;
        }
        ++i;
    }
    return t;
}

double matrix_diff_norm(const xt::xarray<double>& A, const xt::xarray<double>& B) {
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
twist J_times_qdot(const xt::xarray<double>& J, const xt::xarray<double>& q_dot) {
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

// Two 1m planar links rotating about z, ending in a 1m fixed flange.
xt::xarray<double> twolink_table() {
    return xt::xarray<double>{
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
    };
}

// 3 revolute joints separated by fixed 1m spacers, ending in a 1m flange.
xt::xarray<double> threelink_with_spacers_table() {
    return xt::xarray<double>{
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
    };
}

// 6-revolute spatial chain mimicking a UR-like structure.
xt::xarray<double> sixdof_arm_table() {
    return xt::xarray<double>{
        {0, 0, 0.10, 0, 0, 0, 0, 0, 1, k_rev},
        {0, 0, 0.15, 0, 0, 0, 0, 1, 0, k_rev},
        {0.4, 0, 0, 0, 0, 0, 0, 1, 0, k_rev},
        {0.4, 0, 0, 0, 0, 0, 0, 1, 0, k_rev},
        {0, 0, 0.10, 0, 0, 0, 1, 0, 0, k_rev},
        {0, 0, 0.10, 0, 0, 0, 0, 0, 1, k_rev},
        {0, 0, 0.05, 0, 0, 0, 0, 0, 0, k_fix},
    };
}

// The analytic Jacobian in kinematic_chain is the being tested, so it
// cannot be its own evidence. The Jacobian's defining property is that it is
// the derivative of the forward kinematics; the numerical-consistency suite
// verifies exactly that property by finite-differencing the reference forward
// kinematics below, which is written independently of kinematic_chain.

xt::xarray<double> reference_identity4() {
    xt::xarray<double> t = xt::zeros<double>({std::size_t{4}, std::size_t{4}});
    for (std::size_t i = 0; i < 4; ++i) {
        t(i, i) = 1.0;
    }
    return t;
}

xt::xarray<double> reference_matmul(const xt::xarray<double>& a, const xt::xarray<double>& b) {
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

// 4x4 rotation about a unit axis by angle radians (Rodrigues).
xt::xarray<double> reference_axis_rotation(double x, double y, double z, double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double t = 1.0 - c;
    xt::xarray<double> r = reference_identity4();
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

// Tensor columns: 0..2 xyz, 3..5 rpy (fixed-axis XYZ), 6..8 axis, 9 joint
// type.
xt::xarray<double> forward_transform(const xt::xarray<double>& table, const xt::xarray<double>& q) {
    xt::xarray<double> T = reference_identity4();
    std::size_t qi = 0;
    const std::size_t n = table.shape()[0];
    for (std::size_t r = 0; r < n; ++r) {
        xt::xarray<double> link = reference_matmul(
            reference_matmul(reference_axis_rotation(0.0, 0.0, 1.0, table(r, 5)), reference_axis_rotation(0.0, 1.0, 0.0, table(r, 4))),
            reference_axis_rotation(1.0, 0.0, 0.0, table(r, 3)));
        link(0, 3) = table(r, 0);
        link(1, 3) = table(r, 1);
        link(2, 3) = table(r, 2);
        T = reference_matmul(T, link);

        if (table(r, 9) == k_rev) {
            const double ax = table(r, 6);
            const double ay = table(r, 7);
            const double az = table(r, 8);
            const double an = std::sqrt((ax * ax) + (ay * ay) + (az * az));
            T = reference_matmul(T, reference_axis_rotation(ax / an, ay / an, az / an, q(qi)));
            ++qi;
        }
    }
    return T;
}

// Numerical geometric Jacobian via central differences on the reference
// forward_transform.
xt::xarray<double> numerical_jacobian(const xt::xarray<double>& table, const xt::xarray<double>& q, double delta = 1e-7) {
    const std::size_t n_actuated = q.size();
    xt::xarray<double> J_num = xt::zeros<double>({std::size_t{6}, n_actuated});

    for (std::size_t i = 0; i < n_actuated; ++i) {
        xt::xarray<double> q_plus = q;
        xt::xarray<double> q_minus = q;
        q_plus(i) += delta;
        q_minus(i) -= delta;

        const xt::xarray<double> Tp = forward_transform(table, q_plus);
        const xt::xarray<double> Tm = forward_transform(table, q_minus);

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

void check_matches_numerical(const xt::xarray<double>& table, const xt::xarray<double>& q, double tol = 1e-6) {
    const auto J = compute_jacobian(table, q);
    const auto J_num = numerical_jacobian(table, q);
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_num), tol);
}

}  // namespace

// The tests in this suite validate that `J * q_dot` produces the expected
// Cartesian velocity.
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

// The tests in this suite check kinematic_chain::jacobian against hand-computed
// Jacobian values.
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
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_q2_ninety) {
    const auto table = twolink_table();
    const xt::xarray<double> q = {0.0, std::numbers::pi / 2.0};
    const auto J = compute_jacobian(table, q);

    const xt::xarray<double> J_expected = {
        {-1.0, -1.0},
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()

// The numerical-consistency suite depends on the reference forward kinematics
// being correct, so the tests in this suite verify a few easy cases directly.
BOOST_AUTO_TEST_SUITE(fk_tests)

BOOST_AUTO_TEST_CASE(twolink_zero_ee_at_2_0_0) {
    const auto table = twolink_table();
    const xt::xarray<double> T = forward_transform(table, xt::zeros<double>({std::size_t{2}}));
    BOOST_CHECK_CLOSE(T(0, 3), 2.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(1, 3)), 1e-12);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(twolink_q1_pi_over_2_ee_at_0_2_0) {
    const auto table = twolink_table();
    const xt::xarray<double> q = {std::numbers::pi / 2.0, 0.0};
    const xt::xarray<double> T = forward_transform(table, q);
    BOOST_CHECK_SMALL(std::abs(T(0, 3)), 1e-9);
    BOOST_CHECK_CLOSE(T(1, 3), 2.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(twolink_q2_pi_over_2_ee_at_1_1_0) {
    const auto table = twolink_table();
    const xt::xarray<double> q = {0.0, std::numbers::pi / 2.0};
    const xt::xarray<double> T = forward_transform(table, q);
    BOOST_CHECK_CLOSE(T(0, 3), 1.0, 1e-9);
    BOOST_CHECK_CLOSE(T(1, 3), 1.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite check the analytical Jacobian against a
// central-difference numerical Jacobian.
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
    check_matches_numerical(threelink_with_spacers_table(), xt::zeros<double>({std::size_t{3}}));
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

// The tests in this suite cover kinematic_chain itself: reuse of one parsed
// chain across evaluations and agreement between the two Jacobian entry
// points.
BOOST_AUTO_TEST_SUITE(kinematic_chain_tests)

BOOST_AUTO_TEST_CASE(parsed_chain_is_reusable_across_evaluations) {
    const auto chain = kinematic_chain::from(twolink_table());
    const xt::xarray<double> q1 = {0.3, -0.7};
    const xt::xarray<double> q2 = {-1.1, 0.4};

    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q1), compute_jacobian(twolink_table(), q1)), 1e-15);
    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q2), compute_jacobian(twolink_table(), q2)), 1e-15);
}

BOOST_AUTO_TEST_CASE(linear_jacobian_matches_jacobian_linear_block) {
    const auto chain = kinematic_chain::from(sixdof_arm_table());
    const xt::xarray<double> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};

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

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite verify error handling for invalid inputs.
BOOST_AUTO_TEST_SUITE(jacobian_error_tests)

BOOST_AUTO_TEST_CASE(rejects_wrong_q_size) {
    const auto table = twolink_table();
    const xt::xarray<double> q_bad = {0.0, 0.0, 0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q_bad), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(linear_jacobian_rejects_wrong_q_size) {
    const auto chain = kinematic_chain::from(twolink_table());
    const xt::xarray<double> q_bad = {0.0, 0.0, 0.0};
    BOOST_CHECK_THROW(static_cast<void>(chain.linear_jacobian(q_bad)), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_continuous_joint) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, k_cont}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_prismatic_joint) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, k_pris}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_zero_axis_for_revolute) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 0, k_rev}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_wrong_column_count) {
    const xt::xarray<double> bad = xt::zeros<double>({std::size_t{1}, std::size_t{9}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(bad, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_non_2d_tensor) {
    const xt::xarray<double> bad = xt::zeros<double>({std::size_t{10}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(bad, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_empty_table) {
    const xt::xarray<double> bad = xt::zeros<double>({std::size_t{0}, std::size_t{10}});
    const xt::xarray<double> q = xt::zeros<double>({std::size_t{0}});
    BOOST_CHECK_THROW(compute_jacobian(bad, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_non_integer_joint_type) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, 0.5}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_unknown_joint_type) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, 7.0}});
    const xt::xarray<double> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
