#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>

#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>

#include <viam/trajex/totg/observers.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/test/test_utils.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/angles.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/xt.hpp>

namespace {

using namespace viam::trajex::totg;
using namespace viam::trajex::totg::test;
using viam::trajex::arc_length;
using viam::trajex::arc_velocity;
using viam::trajex::degrees_to_radians;
using viam::trajex::xmatrix;
using viam::trajex::xvector;

// Expected number of reversals and colinears per random run. The per-step injection
// probability is derived as p = k_target / k_waypoints_per_run, so the expected count
// is fixed regardless of run length.
constexpr int k_target_special_waypoints_per_type = 10;

// Number of steps (and thus waypoints, including the zero start) per random run.
constexpr int k_waypoints_per_run = 1000;

// TODO(RSDK-12981): Permissive validation tolerance for random tests. Backward integration has known
// acceleration-bound violations; tighten after that is resolved.
constexpr double k_random_test_initial_tolerance = 1000.0;

// Draws from log-uniform distribution over [lo, hi]. Each decade is sampled equally,
// so small values get as much coverage as large ones.
double log_uniform(std::mt19937_64& rng, double lo, double hi) {
    std::uniform_real_distribution<double> dist(std::log(lo), std::log(hi));
    return std::exp(dist(rng));
}

// Returns xarray of shape {dof} with independent uniform draws from [lo, hi].
xvector<> random_xtensor(std::mt19937_64& rng, size_t dof, double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    xvector<> result = xt::empty<double>({dof});
    for (size_t i = 0; i < dof; ++i) {
        result(i) = dist(rng);
    }
    return result;
}

// Returns xarray of shape {dof} with independent log-uniform draws from [lo, hi].
xvector<> random_xtensor_log_uniform(std::mt19937_64& rng, size_t dof, double lo, double hi) {
    xvector<> result = xt::empty<double>({dof});
    for (size_t i = 0; i < dof; ++i) {
        result(i) = log_uniform(rng, lo, hi);
    }
    return result;
}

// Emits a failing seed to stderr and BOOST_TEST_MESSAGE for easy recovery and pinning.
void announce_seed(uint64_t seed) {
    std::ostringstream msg;
    msg << "FAILING SEED: 0x" << std::hex << seed;
    std::cerr << msg.str() << "\n";
    BOOST_TEST_MESSAGE(msg.str());
}

// Driver for random trajectory tests. The seed fully determines the run; the same seed
// always produces the same inputs and the same result. On failure, announces the seed
// before rethrowing so it can be added as a pinned regression test.
//
// The optional setup_fn is invoked after the fixture is configured from the seed but
// before create_and_validate(). Pinned regression tests use it to register event
// expectations or tighten tolerances.
trajectory_test_fixture make_random_fixture(uint64_t seed,
                                            std::optional<std::function<void(trajectory_test_fixture&)>> setup_fn = std::nullopt) {
    std::mt19937_64 rng{seed};

    // Draw all parameters from their specified ranges.
    std::uniform_int_distribution<size_t> dof_dist(2, 8);
    const size_t dof = dof_dist(rng);
    const auto max_velocity = random_xtensor(rng, dof, 0.1, 10.0);
    const auto max_acceleration = random_xtensor(rng, dof, 0.1, 10.0);
    const double delta = log_uniform(rng, 1e-4, 1e-1);
    const double eps = log_uniform(rng, 1e-7, 1e-5);
    const double max_blend_deviation = log_uniform(rng, 1e-4, 1.0);

    // max_linear_deviation: disabled 50% of the time via Bernoulli flip; when enabled,
    // expressed as a multiplier on max_blend_deviation so the relationship between the
    // two is stressed rather than the absolute value.
    std::bernoulli_distribution linear_enabled_dist(0.5);
    const bool linear_deviation_enabled = linear_enabled_dist(rng);
    const double max_linear_deviation = linear_deviation_enabled ? max_blend_deviation * log_uniform(rng, 0.1, 10.0) : 0.0;

    // Blend curvature bounds: within two decades of their defaults (1e-5, 1e5), keeping
    // at least six orders of magnitude of separation between them.
    const double min_blend_curvature = log_uniform(rng, 1e-6, 1e-3);
    const double max_blend_curvature = log_uniform(rng, 1e3, 1e6);

    const auto step_scales = random_xtensor_log_uniform(rng, dof, 0.01, 1.0);

    // Generate waypoints via a random walk. The walk is required because reversal and
    // colinear injection are defined in terms of the incoming step direction; independent
    // draws would give near-orthogonal angles rather than a rich variety of turning angles.
    const double p_special = static_cast<double>(k_target_special_waypoints_per_type) / k_waypoints_per_run;
    std::bernoulli_distribution reversal_dist(p_special);
    std::bernoulli_distribution colinear_dist(p_special);
    std::uniform_real_distribution<double> unit_dist(-1.0, 1.0);

    // Shape: (k_waypoints_per_run + 1) positions, row 0 is the zero start.
    xmatrix<> waypoints = xt::zeros<double>({static_cast<size_t>(k_waypoints_per_run + 1), dof});
    xvector<> current = xt::zeros<double>({dof});
    xvector<> last_step = xt::zeros<double>({dof});

    for (int i = 0; i < k_waypoints_per_run; ++i) {
        // Record current position as waypoint i.
        for (size_t j = 0; j < dof; ++j) {
            waypoints(static_cast<size_t>(i), j) = current(j);
        }

        // Both dice are rolled independently; reversal wins if both fire.
        const bool do_reversal = i > 0 && reversal_dist(rng);
        const bool do_colinear = i > 0 && colinear_dist(rng);
        xvector<> step = xt::zeros<double>({dof});
        if (do_reversal) {
            step = -last_step;
        } else if (do_colinear) {
            step = last_step;
        } else {
            for (size_t j = 0; j < dof; ++j) {
                step(j) = unit_dist(rng) * step_scales(j);
            }
        }

        current += step;
        last_step = step;
    }

    // Record final position.
    for (size_t j = 0; j < dof; ++j) {
        waypoints(static_cast<size_t>(k_waypoints_per_run), j) = current(j);
    }

    // Build and configure the fixture.
    trajectory_test_fixture fixture{dof};
    fixture.set_waypoints_rad(waypoints)
        .set_max_velocity(max_velocity)
        .set_max_acceleration(max_acceleration)
        .set_max_blend_deviation(max_blend_deviation)
        .allow_any_events()
        .disable_joint_kinematics_validation();
    fixture.traj_opts.delta = trajectory::seconds{delta};
    fixture.traj_opts.epsilon = viam::trajex::epsilon{eps};
    if (linear_deviation_enabled) {
        fixture.path_opts.set_max_linear_deviation(max_linear_deviation);
    }
    fixture.path_opts.set_min_blend_curvature(min_blend_curvature).set_max_blend_curvature(max_blend_curvature);

    // TODO(RSDK-12981): add joint kinematic validation here once backward integration correctness is resolved.
    fixture.validation_tolerance_percent = k_random_test_initial_tolerance;

    if (setup_fn.has_value()) {
        (*setup_fn)(fixture);
    }

    return fixture;
}

void run_random_trajectory(uint64_t seed, std::optional<std::function<void(trajectory_test_fixture&)>> setup_fn = std::nullopt) {
    auto fixture = make_random_fixture(seed, std::move(setup_fn));
    try {
        fixture.create_and_validate();
    } catch (...) {
        announce_seed(seed);
        throw;
    }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(random_trajectory_tests)

// Ensure that if we run the random trajectory twice on the same seed,
// that we get deterministic outcomes.
BOOST_AUTO_TEST_CASE(random_trajectory_determinism) {
    const uint64_t seed = std::random_device{}();
    auto f1 = make_random_fixture(seed);
    auto f2 = make_random_fixture(seed);
    f1.enable_event_collection().suppress_create_failure_assertions();
    f2.enable_event_collection().suppress_create_failure_assertions();

    // Keep trajectories alive through the event comparison. splice_event::pruned is a
    // subrange into the trajectory's frosts_, so the trajectory must outlive the comparison.
    // On failure, on_failed() stores the invalid trajectory in the collector, so that
    // path is already safe; on success we must capture it here.
    std::optional<trajectory> t1;
    try {
        t1 = f1.create_and_validate();
    } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch)
    }

    std::optional<trajectory> t2;
    try {
        t2 = f2.create_and_validate();
    } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch)
    }

    BOOST_REQUIRE(f1.collector_ && f2.collector_);
    BOOST_CHECK(f1.collector_->events().size() == f2.collector_->events().size());
    BOOST_CHECK(f1.collector_->events() == f2.collector_->events());

    // Validation inside each run can fail as well as the comparison, and either way the seed is
    // what's needed to reproduce it.
    if (boost::unit_test::results_collector.results(boost::unit_test::framework::current_test_case().p_id).p_assertions_failed > 0) {
        announce_seed(seed);
    }
}

// Found by random_trajectory_determinism: both runs agree, but the trajectory has
// negative s_dot.
BOOST_AUTO_TEST_CASE(random_trajectory_seed_0xc29579d4) {
    run_random_trajectory(0xc29579d4);
}

// Found by random_trajectory_determinism. Forward integration leaves an eq38 Case B switching
// point whose mandated acceleration is wrong, decelerates toward zero s_dot under the
// acceleration limit curve, and reaches zero just short of a segment boundary, so the snap to
// that boundary cannot be honored and trajectory generation throws.
//
// TODO(RSDK-12850): Re-enable once switching point slopes and eq38 mandates are computed
// analytically; with an exact switching point, the trajectory may not approach zero here.
BOOST_AUTO_TEST_CASE(random_trajectory_seed_0x3b9b05d8, *boost::unit_test::disabled()) {
    run_random_trajectory(0x3b9b05d8);
}

// Disabled for now, as it currently finds issues every time it is run
BOOST_AUTO_TEST_CASE(random_trajectory_sweep, *boost::unit_test::disabled()) {
    constexpr int k_max_iterations = 20;

    std::random_device rd;
    std::array<std::seed_seq::result_type, std::mt19937::state_size> seed_data;
    std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937_64 rng{seq};

    for (size_t i = 0; i != k_max_iterations; ++i) {
        run_random_trajectory(rng(), [](auto& fixture [[maybe_unused]]) {
            // fixture.enable_json_output_on_failure(...);
        });
    }
}

BOOST_AUTO_TEST_SUITE_END()  // random_trajectory_tests
