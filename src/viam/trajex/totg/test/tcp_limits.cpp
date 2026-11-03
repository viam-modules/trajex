#include <boost/test/unit_test.hpp>

#include <boost/test/tools/floating_point_comparison.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <json/json.h>

#include <viam/trajex/totg/observers.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/test/test_utils.hpp>
#include <viam/trajex/totg/tools/json_serialization.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg {

namespace {

// A simple 2-DOF path for TCP tests (planar 2-link joint space).
inline path make_2dof_path() {
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.5}, {2.0, 0.0}};
    return path::create(waypoints);
}

// Baseline options for a 2-DOF path (joints effectively unconstrained unless overridden).
inline trajectory::options base_2dof_options() {
    return trajectory::options{.max_velocity = xvector<>{100.0, 100.0}, .max_acceleration = xvector<>{1000.0, 1000.0}};
}

}  // namespace

BOOST_AUTO_TEST_SUITE(tcp_limits_tests)

BOOST_AUTO_TEST_CASE(create_rejects_bad_tcp_limit) {
    const path p = make_2dof_path();
    auto opt = base_2dof_options();

    // non-positive or non-finite max_linear_speed
    for (const double bad : {0.0, -0.5, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        BOOST_TEST_CONTEXT("max_linear_speed=" << bad) {
            opt.tcp = test::planar_2link_tcp_limits(bad);
            BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);
        }
    }

    // missing evaluator
    opt.tcp = trajectory::tcp_limits{.max_linear_speed = 0.5, .evaluator = {}};
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);

    // default-initialized limit with only the evaluator filled in: max_linear_speed is
    // zero-initialized, so validation must reject it deterministically
    trajectory::tcp_limits default_init;
    default_init.evaluator = [](const xvector<>& q, const xvector<>& qp, const xvector<>& qpp) {
        return test::planar_2link_linear_speed_per_arc_velocity(1.0, 1.0, q, qp, qpp);
    };
    opt.tcp = default_init;
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::invalid_argument);

    // valid tcp must NOT throw on construction
    opt.tcp = test::planar_2link_tcp_limits(0.5);
    BOOST_CHECK_NO_THROW(static_cast<void>(trajectory::create(p, opt)));
}

// The TCP velocity component max_linear_speed / ||J(q) * tangent|| is exposed per cursor by
// get_velocity_limit_components. A straight joint-space path has a constant tangent, so placing a
// known (q, tangent) pair under the cursor at the path midpoint lets us read the raw component.
BOOST_AUTO_TEST_CASE(tcp_velocity_component_basic_and_singularity) {
    // planar_2link_jacobian(1,1,[0,0]) = [[0,0],[2,1],[0,0]]. Straight path along [1,0] through
    // q=[0,0]: J*tangent = [0,2,0], norm 2 -> limit = 0.5/2 = 0.25.
    {
        const path p = path::create(xmatrix<>{{-1.0, 0.0}, {1.0, 0.0}});
        auto opt = base_2dof_options();
        opt.tcp = test::planar_2link_tcp_limits(0.5);
        const auto traj = trajectory::create(p, opt);
        const auto c = traj.path().create_cursor(arc_length{static_cast<double>(traj.path().length()) * 0.5});
        BOOST_TEST(static_cast<double>(traj.get_velocity_limit_components(c).tcp) == 0.25, boost::test_tools::tolerance(1e-9));
    }

    // Straight path along [1,-2] through q=[0,0]: that tangent is in the null space of J at [0,0]
    // (2*1 + 1*(-2) = 0), so ||J*tangent|| = 0 and the TCP component is +inf (singularity).
    {
        const path p = path::create(xmatrix<>{{-1.0, 2.0}, {1.0, -2.0}});
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
    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 2.0}};
    const path p = path::create(waypoints);
    auto opt = trajectory::options{.max_velocity = xvector<>{0.6, 0.6}, .max_acceleration = xvector<>{1000.0, 1000.0}};
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
        xmatrix<> wp;
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
        auto opt = trajectory::options{.max_velocity = xvector<>{tc.V, tc.V}, .max_acceleration = xvector<>{tc.A, tc.A}};
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
    const path p = path::create(xmatrix<>{{0.0, 2.0}, {10.0, 0.0}});
    auto opt = trajectory::options{.max_velocity = xvector<>{100.0, 100.0}, .max_acceleration = xvector<>{0.05, 0.05}};
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
// compute_velocity_limit_slope) and forces the trajectory to follow the steep
// drop. So the steep-falling crossing is respected end-to-end with no crossover-specific
// handling.
BOOST_AUTO_TEST_CASE(steep_falling_crossing_respects_tcp_limit) {
    const path p = path::create(xmatrix<>{{0.0, 2.5}, {8.0, 0.3}});
    // joint limit (0.35/max|f'|) sits inside the TCP curve's range along the path, so joint is
    // active early (TCP above it) and TCP active later (TCP falls below), a genuine crossing.
    auto opt = trajectory::options{.max_velocity = xvector<>{0.35, 0.35}, .max_acceleration = xvector<>{0.02, 0.02}};
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

// The terminal integration point must carry a finite acceleration, and the trajectory must be
// sampleable at exactly its duration, which interpolates from that point with dt == 0. A NaN there
// would seek the path cursor to its sentinel and make the sample throw. This was written for
// RSDK-13338, which reported exactly that on a TCP-capped path, but the failure does not reproduce
// at any commit of the branch that added it, with or without the fixes that came with it. It is
// kept as a check of the property across a sweep of caps, not as a regression test.
BOOST_AUTO_TEST_CASE(terminal_acceleration_finite_and_sampleable_at_exact_duration) {
    for (const double cap : {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0}) {
        const path p = path::create(xmatrix<>{{0.0, 2.0}, {10.0, 0.0}});
        auto opt = trajectory::options{.max_velocity = xvector<>{100.0, 100.0}, .max_acceleration = xvector<>{0.05, 0.05}};
        opt.tcp = test::planar_2link_tcp_limits(cap);

        const auto traj = trajectory::create(p, opt);
        BOOST_TEST_CONTEXT("cap=" << cap) {
            BOOST_CHECK(std::isfinite(static_cast<double>(traj.get_integration_points().back().s_ddot)));
            BOOST_CHECK_NO_THROW(static_cast<void>(traj.create_cursor().seek(traj.duration()).sample()));
        }
    }
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

// The diagnostic JSON records the TCP cap among the limits the trajectory was generated under, and
// only when there is one, since a visualizer takes its presence to mean the run was TCP limited.
BOOST_AUTO_TEST_CASE(serialized_trajectory_records_tcp_cap) {
    const path p = make_2dof_path();
    const auto metadata_of = [&](const trajectory::options& opt) {
        trajectory_integration_event_collector collector;
        auto observed = opt;
        observed.observer = &collector;
        const auto traj = trajectory::create(p, observed);
        Json::Value root;
        const Json::CharReaderBuilder reader;
        std::string errs;
        std::istringstream in(serialize_trajectory_to_json(collector, &traj));
        BOOST_REQUIRE_MESSAGE(Json::parseFromStream(reader, in, &root, &errs), errs);
        return root["metadata"];
    };

    auto capped = base_2dof_options();
    capped.tcp = test::planar_2link_tcp_limits(0.1);
    const auto capped_metadata = metadata_of(capped);
    BOOST_REQUIRE(capped_metadata.isMember("tcp_max_linear_speed"));
    BOOST_TEST(capped_metadata["tcp_max_linear_speed"].asDouble() == 0.1);

    BOOST_TEST(!metadata_of(base_2dof_options()).isMember("tcp_max_linear_speed"));
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
        xmatrix<> wp;
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
        auto opt = trajectory::options{.max_velocity = xvector<>{tc.V, tc.V}, .max_acceleration = xvector<>{tc.A, tc.A}};
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

// An evaluator that returns a NaN arc length derivative where the TCP limit is the lower one makes
// the TCP slope non-finite.
// NaN compares false against everything, so an unguarded non-finite slope silently corrupts every
// curve-following decision downstream (trap detection, tangent following); the integrator must
// reject it loudly instead.
BOOST_AUTO_TEST_CASE(tcp_non_finite_limit_slope_throws) {
    const path p = path::create(xmatrix<>{{0.0, 2.0}, {10.0, 0.0}});
    auto opt = trajectory::options{.max_velocity = xvector<>{100.0, 100.0}, .max_acceleration = xvector<>{0.05, 0.05}};
    opt.tcp = trajectory::tcp_limits{
        .max_linear_speed = 0.3,
        .evaluator =
            [](const xvector<>& q, const xvector<>& q_prime, const xvector<>& q_double_prime) {
                auto r = test::planar_2link_linear_speed_per_arc_velocity(1.0, 1.0, q, q_prime, q_double_prime);
                if (r) {
                    r->arc_length_derivative = std::numeric_limits<double>::quiet_NaN();
                }
                return r;
            },
    };
    BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::runtime_error);
}

// An evaluator says the TCP is not moving relative to the path by returning nothing, so a ratio
// it does return must be a real one. A NaN, an infinity or a negative ratio is a broken evaluator
// rather than a singularity, and the integrator refuses it rather than reading it as one.
BOOST_AUTO_TEST_CASE(tcp_non_finite_or_negative_ratio_throws) {
    const path p = make_2dof_path();
    for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0}) {
        BOOST_TEST_CONTEXT("value=" << bad) {
            auto opt = base_2dof_options();
            opt.tcp = trajectory::tcp_limits{
                .max_linear_speed = 0.5,
                .evaluator = [bad](const xvector<>&,
                                   const xvector<>&,
                                   const xvector<>&) -> std::optional<trajectory::tcp_limits::linear_speed_per_arc_velocity> {
                    return trajectory::tcp_limits::linear_speed_per_arc_velocity{.value = bad, .arc_length_derivative = 0.0};
                },
            };
            BOOST_CHECK_THROW(static_cast<void>(trajectory::create(p, opt)), std::runtime_error);
        }
    }
}

// An evaluator that returns nothing everywhere describes a TCP that never moves relative to the
// path, so the TCP never constrains, and the trajectory must be exactly the joint-only one.
BOOST_AUTO_TEST_CASE(tcp_evaluator_returning_nothing_is_joint_only) {
    const path p = make_2dof_path();
    const auto joint_only = trajectory::create(p, base_2dof_options());

    auto opt = base_2dof_options();
    opt.tcp = trajectory::tcp_limits{
        .max_linear_speed = 0.5,
        .evaluator = [](const xvector<>&,
                        const xvector<>&,
                        const xvector<>&) -> std::optional<trajectory::tcp_limits::linear_speed_per_arc_velocity> { return std::nullopt; },
    };
    const auto never_moving = trajectory::create(p, opt);

    BOOST_TEST(never_moving.duration().count() == joint_only.duration().count());
    BOOST_TEST(never_moving.get_integration_points().size() == joint_only.get_integration_points().size());
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace viam::trajex::totg
