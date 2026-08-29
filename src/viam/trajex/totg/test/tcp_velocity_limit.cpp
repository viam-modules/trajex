#include <boost/test/unit_test.hpp>

#include <boost/test/tools/floating_point_comparison.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <json/json.h>

#include <viam/trajex/totg/observers.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/tools/json_serialization.hpp>
#include <viam/trajex/totg/trajectory.hpp>

#include "test_utils.hpp"

namespace viam::trajex::totg {

namespace {

// A simple 2-DOF path for TCP tests (planar 2-link joint space).
inline path make_2dof_path() {
    const xt::xarray<double> waypoints = {{0.0, 0.0}, {1.0, 0.5}, {2.0, 0.0}};
    return path::create(waypoints);
}

// Baseline options for a 2-DOF path (joints effectively unconstrained unless overridden).
inline trajectory::options base_2dof_options() {
    return trajectory::options{.max_velocity = xt::xarray<double>{100.0, 100.0}, .max_acceleration = xt::xarray<double>{1000.0, 1000.0}};
}

}  // namespace

BOOST_AUTO_TEST_SUITE(tcp_velocity_limit_tests)

BOOST_AUTO_TEST_CASE(create_rejects_bad_tcp_limit) {
    const path p = make_2dof_path();
    auto opt = base_2dof_options();

    // non-positive max_velocity
    opt.tcp = trajectory::tcp_limits{.max_linear_velocity = 0.0,
                                    .linear_jacobian = [](const xt::xarray<double>& q) { return test::planar_2link_jacobian(1.0, 1.0, q); },
                                    .linear_velocity_gain = {}};
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);

    // missing jacobian
    opt.tcp = trajectory::tcp_limits{.max_linear_velocity = 0.5, .linear_jacobian = {}, .linear_velocity_gain = {}};
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);

    // default-initialized limit with only the callbacks filled in: max_velocity is
    // zero-initialized, so validation must reject it deterministically
    trajectory::tcp_limits default_init;
    default_init.linear_jacobian = [](const xt::xarray<double>& q) { return test::planar_2link_jacobian(1.0, 1.0, q); };
    default_init.linear_velocity_gain = [](const xt::xarray<double>& q, const xt::xarray<double>& qp, const xt::xarray<double>& qpp) {
        return test::planar_2link_linear_velocity_gain(1.0, 1.0, q, qp, qpp);
    };
    opt.tcp = default_init;
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);

    // valid tcp must NOT throw on construction
    opt.tcp = test::planar_2link_tcp_limits(0.5);
    BOOST_CHECK_NO_THROW(static_cast<void>(trajectory::create(p, opt)));
}

// The TCP velocity component max_velocity / ||J(q) * tangent|| is exposed per cursor by
// get_velocity_limit_components. A straight joint-space path has a constant tangent, so placing a
// known (q, tangent) pair under the cursor at the path midpoint lets us read the raw component.
BOOST_AUTO_TEST_CASE(tcp_velocity_component_basic_and_singularity) {
    // planar_2link_jacobian(1,1,[0,0]) = [[0,0],[2,1],[0,0]]. Straight path along [1,0] through
    // q=[0,0]: J*tangent = [0,2,0], norm 2 -> limit = 0.5/2 = 0.25.
    {
        const path p = path::create(xt::xarray<double>{{-1.0, 0.0}, {1.0, 0.0}});
        auto opt = base_2dof_options();
        opt.tcp = test::planar_2link_tcp_limits(0.5);
        const auto traj = trajectory::create(p, opt);
        const auto c = traj.path().create_cursor(arc_length{static_cast<double>(traj.path().length()) * 0.5});
        BOOST_TEST(static_cast<double>(traj.get_velocity_limit_components(c).tcp) == 0.25, boost::test_tools::tolerance(1e-9));
    }

    // Straight path along [1,-2] through q=[0,0]: that tangent is in the null space of J at [0,0]
    // (2*1 + 1*(-2) = 0), so ||J*tangent|| = 0 and the TCP component is +inf (singularity).
    {
        const path p = path::create(xt::xarray<double>{{-1.0, 2.0}, {1.0, -2.0}});
        auto opt = base_2dof_options();
        opt.tcp = test::planar_2link_tcp_limits(0.5);
        const auto traj = trajectory::create(p, opt);
        const auto c = traj.path().create_cursor(arc_length{static_cast<double>(traj.path().length()) * 0.5});
        BOOST_TEST(std::isinf(static_cast<double>(traj.get_velocity_limit_components(c).tcp)));
    }
}

// A joint<->TCP crossing of the combined min(joint, TCP) ceiling is always a concave corner
// (the active curve's slope drops through it), so it is never a velocity switching point; the
// trajectory rides over it. This path is a single straight joint-space segment {{0,0},{1,2}}:
// the joint limit is constant (f' is constant) while the TCP limit varies because J(q) changes
// along it (||J*f'||^2 = a^2 + (a+b)^2 + 2a(a+b)*cos(q2)), so the two curves genuinely cross
// interior to the segment (verified here by the combined curve's active-flag flipping). The
// crossing is benign (the ceiling does not fall faster than the arm can decelerate), and the
// realized TCP speed stays under the cap.
BOOST_AUTO_TEST_CASE(benign_crossover_respects_tcp_limit) {
    const xt::xarray<double> waypoints = {{0.0, 0.0}, {1.0, 2.0}};
    const path p = path::create(waypoints);
    auto opt = trajectory::options{.max_velocity = xt::xarray<double>{0.6, 0.6}, .max_acceleration = xt::xarray<double>{1000.0, 1000.0}};
    opt.tcp = test::planar_2link_tcp_limits(1.0);
    const auto traj = trajectory::create(p, opt);

    // Sanity: the combined curve really does switch active constraint along this segment. The
    // public per-cursor joint and TCP components expose which is binding (TCP active where it is
    // the smaller of the two), so a flip in that flag is a genuine joint<->TCP crossing.
    const double L = static_cast<double>(traj.path().length());
    bool first = true;
    bool prev_active = false;
    int flips = 0;
    for (int i = 0; i <= 60; ++i) {
        const auto c = traj.path().create_cursor(arc_length{L * static_cast<double>(i) / 60.0});
        const auto comp = traj.get_velocity_limit_components(c);
        const bool tcp_active = static_cast<double>(comp.tcp) < static_cast<double>(comp.joint);
        if (!first && tcp_active != prev_active) {
            ++flips;
        }
        prev_active = tcp_active;
        first = false;
    }
    BOOST_REQUIRE_MESSAGE(flips >= 1, "test path must contain a genuine joint<->TCP crossing");

    // The benign crossing yields a feasible trajectory whose realized TCP speed respects the cap.
    BOOST_TEST(test::max_realized_tcp_speed(traj, 1.0, 1.0) <= 1.0 + 1e-3);
}

// Backward integration must survive a TCP dip-and-recovery: a path that drives q2 through a
// region where ||J*f'|| peaks (the TCP ceiling dips) then recovers cuts a notch into the
// combined curve, forcing a switching point from which backward integration must reach the
// forward pass. Every case below must integrate without throwing, and the realized TCP speed
// must respect the cap. The integrator also carries a defensive TCP-guarded clamp for backward
// overshoot; none of these paths currently trigger it.
BOOST_AUTO_TEST_CASE(tcp_dip_and_recovery_produces_feasible_trajectory) {
    struct dip_case {
        xt::xarray<double> wp;
        double V;
        double vt;
        double A;
    };
    const std::vector<dip_case> cases = {
        {{{0.0, 1.5}, {6.76, -1.5}}, 100.0, 0.05, 5.0},              // wide dip
        {{{0.0, 0.6}, {0.3, -0.6}}, 100.0, 0.05, 0.05},              // near-singular graze, tiny accel
        {{{0.0, 0.5}, {0.2, -0.5}}, 0.5, 0.04, 0.05},                // singular + tight V + tight A
        {{{-0.4, 0.5}, {0.0, 0.02}, {0.4, 0.5}}, 100.0, 0.05, 0.1},  // 3wp blend through singularity
        {{{0.0, 1.0}, {0.4, -1.0}}, 0.3, 0.02, 0.03},                // all tight
        {{{0.0, 0.4}, {0.2, -0.4}}, 100.0, 0.04, 0.01},              // extreme graze, tiny accel
    };

    for (const auto& tc : cases) {
        const path p = path::create(tc.wp);
        auto opt = trajectory::options{.max_velocity = xt::xarray<double>{tc.V, tc.V}, .max_acceleration = xt::xarray<double>{tc.A, tc.A}};
        opt.tcp = test::planar_2link_tcp_limits(tc.vt);

        // create() must not throw on the dip-and-recovery (uncaught throw fails the case).
        const auto traj = trajectory::create(p, opt);
        const double peak = test::max_realized_tcp_speed(traj, 1.0, 1.0);
        BOOST_TEST(peak <= tc.vt + 1e-3);
    }
}

// End-to-end: with joints unconstrained and a TCP limit, the realized Cartesian TCP speed
// (||J(q)*q_dot||) along the generated trajectory must respect the cap. Here a long
// q1-dominant approach lets the trajectory accelerate up to the TCP ceiling, which then
// falls steeply as q2 sweeps toward 0 (||J*f'|| rises) under a tiny accel budget.
BOOST_AUTO_TEST_CASE(tcp_limit_respected_end_to_end) {
    const path p = path::create(xt::xarray<double>{{0.0, 2.0}, {10.0, 0.0}});
    auto opt = trajectory::options{.max_velocity = xt::xarray<double>{100.0, 100.0}, .max_acceleration = xt::xarray<double>{0.05, 0.05}};
    opt.tcp = test::planar_2link_tcp_limits(0.3);

    const auto traj = trajectory::create(p, opt);
    BOOST_TEST(test::max_realized_tcp_speed(traj, 1.0, 1.0, 4000) <= 0.3 + 1e-3);
}

// A TCP limit so loose it never binds must reproduce the joint-only baseline exactly: the
// combined curve min(joint, huge) == joint, so the TCP machinery is a no-op when not binding.
BOOST_AUTO_TEST_CASE(loose_tcp_matches_joint_only_baseline) {
    const path p = make_2dof_path();
    auto base = base_2dof_options();
    const auto baseline = trajectory::create(p, base);

    auto with_loose = base;
    with_loose.tcp = test::planar_2link_tcp_limits(1.0e6);
    const auto loose = trajectory::create(p, with_loose);

    BOOST_TEST(loose.duration().count() == baseline.duration().count(), boost::test_tools::tolerance(1e-12));
}

// This path has a genuine joint<->TCP crossing with TCP active and falling steeply after it.
// The crossing itself is a concave corner, not a switching point; the cap is held because the
// breach-handling step rides the combined min(joint, TCP) curve (its slope comes from
// compute_velocity_limit_derivative_with_tcp) and forces the trajectory to follow the steep
// drop. So the steep-falling crossing is respected end-to-end with no crossover-specific
// handling.
BOOST_AUTO_TEST_CASE(steep_falling_crossing_respects_tcp_limit) {
    const path p = path::create(xt::xarray<double>{{0.0, 2.5}, {8.0, 0.3}});
    // joint limit (0.35/max|f'|) sits inside the TCP curve's range along the path, so joint is
    // active early (TCP above it) and TCP active later (TCP falls below), a genuine crossing.
    auto opt = trajectory::options{.max_velocity = xt::xarray<double>{0.35, 0.35}, .max_acceleration = xt::xarray<double>{0.02, 0.02}};
    opt.tcp = test::planar_2link_tcp_limits(0.3);
    const auto traj = trajectory::create(p, opt);

    // Confirm the path actually contains a joint<->TCP crossing: the smaller of the public joint
    // and TCP components (the active constraint) flips along the path.
    const double L = static_cast<double>(traj.path().length());
    bool first = true;
    bool prev_active = false;
    int flips = 0;
    for (int i = 0; i <= 80; ++i) {
        const auto c = traj.path().create_cursor(arc_length{L * static_cast<double>(i) / 80.0});
        const auto comp = traj.get_velocity_limit_components(c);
        const bool tcp_active = static_cast<double>(comp.tcp) < static_cast<double>(comp.joint);
        if (!first && tcp_active != prev_active) {
            ++flips;
        }
        prev_active = tcp_active;
        first = false;
    }
    BOOST_REQUIRE_MESSAGE(flips >= 1, "arbiter path must contain a genuine joint<->TCP crossing");

    // The realized TCP speed is held at the cap end-to-end.
    BOOST_TEST(test::max_realized_tcp_speed(traj, 1.0, 1.0, 4000) <= 0.3 + 1e-3);
}

// Regression (RSDK-13338): generation used to leave the terminal integration point with a NaN
// acceleration when the trajectory terminated at a path-end switching point that carried no
// forward_accel (the switching-point cache's path-end sentinel, and forward integration landing
// exactly on the path end). Sampling at exactly t == duration interpolates from that point with
// dt == 0; the NaN poisons the interpolated arc length, the path cursor seeks to its sentinel,
// and the sample query throws "Cannot query cursor at sentinel position". The terminal
// acceleration must be finite and the trajectory sampleable at its exact duration.
BOOST_AUTO_TEST_CASE(terminal_acceleration_finite_and_sampleable_at_exact_duration) {
    for (const double cap : {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0}) {
        const path p = path::create(xt::xarray<double>{{0.0, 2.0}, {10.0, 0.0}});
        auto opt =
            trajectory::options{.max_velocity = xt::xarray<double>{100.0, 100.0}, .max_acceleration = xt::xarray<double>{0.05, 0.05}};
        opt.tcp = test::planar_2link_tcp_limits(cap);

        const auto traj = trajectory::create(p, opt);
        BOOST_TEST_CONTEXT("cap=" << cap) {
            BOOST_CHECK(std::isfinite(static_cast<double>(traj.get_integration_points().back().s_ddot)));
            BOOST_CHECK_NO_THROW(static_cast<void>(traj.create_cursor().seek(traj.duration()).sample()));
        }
    }
}

// The TCP velocity limit requires the jacobian callback to return a 3xN matrix (3 Cartesian
// rows, one column per joint DOF). A callback that violates that contract must be rejected;
// trajectory::create evaluates the limit during generation, so the bad shape surfaces there.
BOOST_AUTO_TEST_CASE(tcp_jacobian_callback_wrong_shape_throws) {
    const path p = make_2dof_path();
    auto opt = base_2dof_options();

    // Fewer than 3 rows.
    opt.tcp = trajectory::tcp_limits{.max_linear_velocity = 0.5,
                                    .linear_jacobian = [](const xt::xarray<double>&) { return xt::xarray<double>{{1.0, 0.0}, {0.0, 1.0}}; },
                                    .linear_velocity_gain = {}};
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);

    // Right number of rows, wrong column count (3 columns for a 2-DOF tangent).
    opt.tcp = trajectory::tcp_limits{
        .max_linear_velocity = 0.5,
        .linear_jacobian = [](const xt::xarray<double>&) { return xt::xarray<double>{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}; },
        .linear_velocity_gain = {}};
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);
}

// A 1-D array (dimension 1) whose length happens to be 3 must be rejected cleanly. Without a
// dimension check, validating the column count reads shape(1) out of bounds on a rank-1 shape.
BOOST_AUTO_TEST_CASE(tcp_jacobian_callback_wrong_dimension_throws) {
    const path p = make_2dof_path();
    auto opt = base_2dof_options();
    opt.tcp = trajectory::tcp_limits{.max_linear_velocity = 0.5,
                                    .linear_jacobian = [](const xt::xarray<double>&) { return xt::xarray<double>{1.0, 2.0, 3.0}; },
                                    .linear_velocity_gain = {}};
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);
}

// get_velocity_limit_components exposes the joint and TCP curves separately; the combined
// velocity limit from get_velocity_limits is their minimum.
BOOST_AUTO_TEST_CASE(velocity_limit_components_separate_joint_and_tcp) {
    const path p = make_2dof_path();
    auto opt = base_2dof_options();  // joints effectively unconstrained (max_velocity 100)
    opt.tcp = test::planar_2link_tcp_limits(0.1);
    const auto traj = trajectory::create(p, opt);
    auto cursor = traj.path().create_cursor(arc_length{static_cast<double>(traj.path().length()) * 0.5});

    const auto comp = traj.get_velocity_limit_components(cursor);
    // With the joints unconstrained and a tight TCP cap, TCP is the binding component.
    BOOST_TEST(static_cast<double>(comp.tcp) < static_cast<double>(comp.joint));
    // The combined curve equals the smaller (TCP) component at this cursor.
    BOOST_TEST(static_cast<double>(traj.get_velocity_limits(cursor).s_dot_max_vel) == static_cast<double>(comp.tcp),
               boost::test_tools::tolerance(1e-12));

    // With no TCP limit, the TCP component is non-constraining (+inf) and the combined curve
    // equals the joint component.
    const auto traj_joint = trajectory::create(make_2dof_path(), base_2dof_options());
    auto cursor_joint = traj_joint.path().create_cursor(arc_length{static_cast<double>(traj_joint.path().length()) * 0.5});
    const auto comp_joint = traj_joint.get_velocity_limit_components(cursor_joint);
    BOOST_TEST(std::isinf(static_cast<double>(comp_joint.tcp)));
    BOOST_TEST(static_cast<double>(traj_joint.get_velocity_limits(cursor_joint).s_dot_max_vel) == static_cast<double>(comp_joint.joint),
               boost::test_tools::tolerance(1e-12));
}

// The diagnostic JSON exposes the joint and TCP limit curves as separate series so the phase
// plane can be visualized with the TCP curve, and the combined curve is their minimum.
BOOST_AUTO_TEST_CASE(serialized_trajectory_exposes_tcp_limit_curve) {
    const path p = make_2dof_path();
    auto opt = base_2dof_options();
    opt.tcp = test::planar_2link_tcp_limits(0.1);
    trajectory_integration_event_collector collector;
    opt.observer = &collector;
    const auto traj = trajectory::create(p, opt);

    const std::string json = serialize_trajectory_to_json(collector, &traj);
    Json::Value root;
    const Json::CharReaderBuilder reader;
    std::string errs;
    std::istringstream in(json);
    BOOST_REQUIRE_MESSAGE(Json::parseFromStream(reader, in, &root, &errs), errs);

    const auto& ip = root["integration_points"];
    BOOST_REQUIRE(ip.isMember("s_dot_max_vel_joint"));
    BOOST_REQUIRE(ip.isMember("s_dot_max_vel_tcp"));
    const auto& joint = ip["s_dot_max_vel_joint"];
    const auto& tcp = ip["s_dot_max_vel_tcp"];
    const auto& combined = ip["s_dot_max_vel"];
    BOOST_REQUIRE(!combined.empty());

    bool saw_binding = false;
    for (Json::ArrayIndex i = 0; i < combined.size(); ++i) {
        if (combined[i].isNull() || tcp[i].isNull() || joint[i].isNull()) {
            continue;
        }
        // The combined curve is the minimum, which is the TCP component here.
        BOOST_TEST(combined[i].asDouble() == tcp[i].asDouble(), boost::test_tools::tolerance(1e-9));
        BOOST_TEST(tcp[i].asDouble() <= joint[i].asDouble() + 1e-9);
        if (tcp[i].asDouble() < joint[i].asDouble() - 1e-9) {
            saw_binding = true;
        }
    }
    BOOST_TEST(saw_binding);
}

// Every committed integration point must respect the combined velocity limit curve. Breach
// handling commits escape points clamped onto a limit curve, and clamping to the wrong curve
// (the acceleration curve at a breach where the TCP curve dips below it) would stamp a phase
// point the sampled trajectory cannot honor.
BOOST_AUTO_TEST_CASE(integration_points_respect_combined_velocity_limit) {
    struct limit_case {
        xt::xarray<double> wp;
        double V;
        double vt;
        double A;
    };
    const std::vector<limit_case> cases = {
        {{{0.0, 2.0}, {10.0, 0.0}}, 100.0, 0.3, 0.05},   // long TCP-limited approach
        {{{0.0, 1.5}, {6.76, -1.5}}, 100.0, 0.05, 5.0},  // wide TCP dip
        {{{0.0, 1.0}, {0.4, -1.0}}, 0.3, 0.02, 0.03},    // joint and TCP both tight
        {{{0.0, 2.5}, {8.0, 0.3}}, 0.35, 0.3, 0.02},     // joint<->TCP crossing
    };

    for (const auto& tc : cases) {
        const path p = path::create(tc.wp);
        auto opt = trajectory::options{.max_velocity = xt::xarray<double>{tc.V, tc.V}, .max_acceleration = xt::xarray<double>{tc.A, tc.A}};
        opt.tcp = test::planar_2link_tcp_limits(tc.vt);
        const auto traj = trajectory::create(p, opt);

        for (const auto& ip : traj.get_integration_points()) {
            auto c = traj.path().create_cursor(ip.s);
            const auto limit = traj.get_velocity_limits(c).s_dot_max_vel;
            BOOST_TEST_CONTEXT("V=" << tc.V << " vt=" << tc.vt << " A=" << tc.A << " s=" << static_cast<double>(ip.s)) {
                BOOST_TEST(static_cast<double>(ip.s_dot) <= static_cast<double>(limit) + 1e-9);
            }
        }
    }
}

// A velocity_derivative that contradicts the jacobian (zero gain at a TCP-binding point) makes
// the Eq. 25 slope divide by zero and go non-finite. NaN compares false against everything, so
// an unguarded non-finite slope silently corrupts every curve-following decision downstream
// (trap detection, tangent following); the integrator must reject it loudly instead.
BOOST_AUTO_TEST_CASE(tcp_non_finite_limit_slope_throws) {
    const path p = path::create(xt::xarray<double>{{0.0, 2.0}, {10.0, 0.0}});
    auto opt = trajectory::options{.max_velocity = xt::xarray<double>{100.0, 100.0}, .max_acceleration = xt::xarray<double>{0.05, 0.05}};
    opt.tcp = trajectory::tcp_limits{
        .max_linear_velocity = 0.3,
        .linear_jacobian = [](const xt::xarray<double>& q) { return test::planar_2link_jacobian(1.0, 1.0, q); },
        .linear_velocity_gain = [](const xt::xarray<double>&,
                                  const xt::xarray<double>&,
                                  const xt::xarray<double>&) { return jacobian::kinematic_chain::linear_velocity_gain{0.0, 0.0}; },
    };
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(tcp_limit_without_velocity_derivative_throws) {
    const path p = make_2dof_path();
    auto opt = base_2dof_options();
    opt.tcp = trajectory::tcp_limits{.max_linear_velocity = 0.5,
                                    .linear_jacobian = [](const xt::xarray<double>& q) { return test::planar_2link_jacobian(1.0, 1.0, q); },
                                    .linear_velocity_gain = {}};
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace viam::trajex::totg
