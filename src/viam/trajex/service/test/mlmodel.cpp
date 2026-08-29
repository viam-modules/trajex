#define BOOST_TEST_MODULE trajex_service_test

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#include <boost/test/included/unit_test.hpp>
#pragma GCC diagnostic pop

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <boost/variant/get.hpp>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/sdk/config/resource.hpp>
#include <viam/sdk/services/mlmodel.hpp>

#include <viam/trajex/service/mlmodel.hpp>

namespace {

namespace vsdk = ::viam::sdk;
using viam::trajex::service::mlmodel;

vsdk::ResourceConfig make_config(const vsdk::ProtoStruct& attrs = {}) {
    return vsdk::ResourceConfig{
        "mlmodel",      // type (must match api subtype)
        "trajex_test",  // name
        "rdk",          // namespace (must match api namespace)
        attrs,
        "rdk:service:mlmodel",
        vsdk::Model{"viam", "mlmodelservice", "trajex"},
    };
}

// Build a named_tensor_views map with a simple two-point 3-DOF trajectory input
mlmodel::named_tensor_views make_simple_inputs() {
    // We need the data to outlive the views, so use static storage for test convenience
    static const std::vector<double> waypoints_data = {
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    };
    static const std::vector<double> vel_limits = {1.0, 1.0, 1.0};
    static const std::vector<double> acc_limits = {1.0, 1.0, 1.0};
    static const std::vector<double> path_tolerance = {0.001};
    static const std::vector<double> colinearization_ratio = {0.0};  // disabled
    static const std::vector<double> dedup_tolerance = {1e-6};
    static const std::vector<std::int64_t> sampling_freq = {100};

    mlmodel::named_tensor_views inputs;

    inputs.emplace("waypoints_rads", mlmodel::make_tensor_view(waypoints_data.data(), 6, {2, 3}));
    inputs.emplace("velocity_limits_rads_per_sec", mlmodel::make_tensor_view(vel_limits.data(), 3, {3}));
    inputs.emplace("acceleration_limits_rads_per_sec2", mlmodel::make_tensor_view(acc_limits.data(), 3, {3}));
    inputs.emplace("path_tolerance_delta_rads", mlmodel::make_tensor_view(path_tolerance.data(), 1, {1}));
    inputs.emplace("path_colinearization_ratio", mlmodel::make_tensor_view(colinearization_ratio.data(), 1, {1}));
    inputs.emplace("waypoint_deduplication_tolerance_rads", mlmodel::make_tensor_view(dedup_tolerance.data(), 1, {1}));
    inputs.emplace("trajectory_sampling_freq_hz", mlmodel::make_tensor_view(sampling_freq.data(), 1, {1}));

    return inputs;
}

// Build a 2-DOF input set on a single straight segment where the TCP limit binds (joints loose,
// TCP tight, small accel). With `with_tcp`, adds the TCP scalar + a planar 2-link (l1=l2=1)
// model table (2 revolute about z + a fixed flange), matching test::planar_2link_jacobian(1,1,.).
mlmodel::named_tensor_views make_2dof_tcp_inputs(bool with_tcp) {
    static const std::vector<double> waypoints = {0.0, 2.0, 10.0, 0.0};  // 2 waypoints x 2 dof
    static const std::vector<double> vel_limits = {100.0, 100.0};
    static const std::vector<double> acc_limits = {0.05, 0.05};
    static const std::vector<double> path_tolerance = {0.001};
    static const std::vector<double> colinearization_ratio = {0.0};
    static const std::vector<double> dedup_tolerance = {1e-6};
    static const std::vector<std::int64_t> sampling_freq = {100};
    static const std::vector<double> tcp_max = {0.3};
    static const std::vector<double> model_table = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // joint0 at origin, axis z, revolute
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // joint1 +1x,      axis z, revolute
        1, 0, 0, 0, 0, 0, 0, 0, 0, 3,  // flange +1x,      fixed
    };

    mlmodel::named_tensor_views inputs;
    inputs.emplace("waypoints_rads", mlmodel::make_tensor_view(waypoints.data(), 4, {2, 2}));
    inputs.emplace("velocity_limits_rads_per_sec", mlmodel::make_tensor_view(vel_limits.data(), 2, {2}));
    inputs.emplace("acceleration_limits_rads_per_sec2", mlmodel::make_tensor_view(acc_limits.data(), 2, {2}));
    inputs.emplace("path_tolerance_delta_rads", mlmodel::make_tensor_view(path_tolerance.data(), 1, {1}));
    inputs.emplace("path_colinearization_ratio", mlmodel::make_tensor_view(colinearization_ratio.data(), 1, {1}));
    inputs.emplace("waypoint_deduplication_tolerance_rads", mlmodel::make_tensor_view(dedup_tolerance.data(), 1, {1}));
    inputs.emplace("trajectory_sampling_freq_hz", mlmodel::make_tensor_view(sampling_freq.data(), 1, {1}));
    if (with_tcp) {
        inputs.emplace("tcp_max_linear_velocity", mlmodel::make_tensor_view(tcp_max.data(), 1, {1}));
        inputs.emplace("kinematics_model_table", mlmodel::make_tensor_view(model_table.data(), 30, {3, 10}));
    }
    return inputs;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(trajex_service_config_tests)

BOOST_AUTO_TEST_CASE(default_config) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    // Should not throw -- default config is ["totg", "legacy"]
    auto result = service->infer(make_simple_inputs(), {});
    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_CASE(totg_only_config) {
    vsdk::ProtoStruct attrs;
    attrs.emplace("generator_sequence", std::vector<vsdk::ProtoValue>{vsdk::ProtoValue{"totg"}});

    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config(attrs));
    auto result = service->infer(make_simple_inputs(), {});
    BOOST_CHECK(result);

    // Should have accelerations (trajex produces them)
    BOOST_CHECK(result->count("accelerations_rads_per_sec2") > 0);
}

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
BOOST_AUTO_TEST_CASE(legacy_only_config) {
    vsdk::ProtoStruct attrs;
    attrs.emplace("generator_sequence", std::vector<vsdk::ProtoValue>{vsdk::ProtoValue{"legacy"}});

    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config(attrs));
    auto result = service->infer(make_simple_inputs(), {});
    BOOST_CHECK(result);

    // Should NOT have accelerations (legacy doesn't produce them)
    BOOST_CHECK(result->count("accelerations_rads_per_sec2") == 0);
}
#endif

BOOST_AUTO_TEST_CASE(unknown_algorithm_rejects) {
    vsdk::ProtoStruct attrs;
    attrs.emplace("generator_sequence", std::vector<vsdk::ProtoValue>{vsdk::ProtoValue{"unknown"}});

    BOOST_CHECK_THROW(std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config(attrs)), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(empty_sequence_rejects) {
    vsdk::ProtoStruct attrs;
    attrs.emplace("generator_sequence", std::vector<vsdk::ProtoValue>{});

    BOOST_CHECK_THROW(std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config(attrs)), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(trajex_service_input_validation_tests)

BOOST_AUTO_TEST_CASE(missing_tensor_throws) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    const mlmodel::named_tensor_views empty_inputs;
    BOOST_CHECK_THROW(service->infer(empty_inputs, {}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(dof_mismatch_throws) {
    static const std::vector<double> waypoints_data = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    static const std::vector<double> vel_limits = {1.0, 1.0};  // 2-DOF, waypoints are 3-DOF
    static const std::vector<double> acc_limits = {1.0, 1.0};
    static const std::vector<double> path_tolerance = {0.001};
    static const std::vector<double> colinearization_ratio = {0.0};
    static const std::vector<double> dedup_tolerance = {1e-6};
    static const std::vector<std::int64_t> sampling_freq = {100};

    mlmodel::named_tensor_views inputs;
    inputs.emplace("waypoints_rads", mlmodel::make_tensor_view(waypoints_data.data(), 6, {2, 3}));
    inputs.emplace("velocity_limits_rads_per_sec", mlmodel::make_tensor_view(vel_limits.data(), 2, {2}));
    inputs.emplace("acceleration_limits_rads_per_sec2", mlmodel::make_tensor_view(acc_limits.data(), 2, {2}));
    inputs.emplace("path_tolerance_delta_rads", mlmodel::make_tensor_view(path_tolerance.data(), 1, {1}));
    inputs.emplace("path_colinearization_ratio", mlmodel::make_tensor_view(colinearization_ratio.data(), 1, {1}));
    inputs.emplace("waypoint_deduplication_tolerance_rads", mlmodel::make_tensor_view(dedup_tolerance.data(), 1, {1}));
    inputs.emplace("trajectory_sampling_freq_hz", mlmodel::make_tensor_view(sampling_freq.data(), 1, {1}));

    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    BOOST_CHECK_THROW(service->infer(inputs, {}), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(trajex_service_inference_tests)

BOOST_AUTO_TEST_CASE(simple_trajectory_produces_valid_output) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    auto result = service->infer(make_simple_inputs(), {});

    BOOST_REQUIRE(result);
    BOOST_CHECK(result->count("sample_times_sec") > 0);
    BOOST_CHECK(result->count("configurations_rads") > 0);
    BOOST_CHECK(result->count("velocities_rads_per_sec") > 0);

    // Verify output types are float64
    BOOST_CHECK_EQUAL(vsdk::MLModelService::tensor_info::tensor_views_to_data_type(result->at("sample_times_sec")),
                      vsdk::MLModelService::tensor_info::data_types::k_float64);
    BOOST_CHECK_EQUAL(vsdk::MLModelService::tensor_info::tensor_views_to_data_type(result->at("configurations_rads")),
                      vsdk::MLModelService::tensor_info::data_types::k_float64);
}

BOOST_AUTO_TEST_CASE(single_waypoint_returns_empty) {
    static const std::vector<double> single_wp = {0.0, 0.0, 0.0};
    static const std::vector<double> vel_limits = {1.0, 1.0, 1.0};
    static const std::vector<double> acc_limits = {1.0, 1.0, 1.0};
    static const std::vector<double> path_tolerance = {0.001};
    static const std::vector<double> colinearization_ratio = {0.0};
    static const std::vector<double> dedup_tolerance = {1e-6};
    static const std::vector<std::int64_t> sampling_freq = {100};

    mlmodel::named_tensor_views inputs;
    inputs.emplace("waypoints_rads", mlmodel::make_tensor_view(single_wp.data(), 3, {1, 3}));
    inputs.emplace("velocity_limits_rads_per_sec", mlmodel::make_tensor_view(vel_limits.data(), 3, {3}));
    inputs.emplace("acceleration_limits_rads_per_sec2", mlmodel::make_tensor_view(acc_limits.data(), 3, {3}));
    inputs.emplace("path_tolerance_delta_rads", mlmodel::make_tensor_view(path_tolerance.data(), 1, {1}));
    inputs.emplace("path_colinearization_ratio", mlmodel::make_tensor_view(colinearization_ratio.data(), 1, {1}));
    inputs.emplace("waypoint_deduplication_tolerance_rads", mlmodel::make_tensor_view(dedup_tolerance.data(), 1, {1}));
    inputs.emplace("trajectory_sampling_freq_hz", mlmodel::make_tensor_view(sampling_freq.data(), 1, {1}));

    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    auto result = service->infer(inputs, {});

    // Single waypoint: nothing to do, should return empty views
    BOOST_REQUIRE(result);
    BOOST_CHECK(result->empty());
}

BOOST_AUTO_TEST_CASE(totg_trajectory_has_samples_and_accelerations) {
    vsdk::ProtoStruct attrs;
    attrs.emplace("generator_sequence", std::vector<vsdk::ProtoValue>{vsdk::ProtoValue{"totg"}});

    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config(attrs));
    auto result = service->infer(make_simple_inputs(), {});

    BOOST_REQUIRE(result);
    BOOST_CHECK(result->count("sample_times_sec") > 0);
    BOOST_CHECK(result->count("configurations_rads") > 0);
    BOOST_CHECK(result->count("velocities_rads_per_sec") > 0);
    BOOST_CHECK(result->count("accelerations_rads_per_sec2") > 0);

    // All output tensors should be float64
    for (const auto& key : {"sample_times_sec", "configurations_rads", "velocities_rads_per_sec", "accelerations_rads_per_sec2"}) {
        BOOST_CHECK_EQUAL(vsdk::MLModelService::tensor_info::tensor_views_to_data_type(result->at(key)),
                          vsdk::MLModelService::tensor_info::data_types::k_float64);
    }
}

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
BOOST_AUTO_TEST_CASE(legacy_trajectory_has_samples_but_no_accelerations) {
    vsdk::ProtoStruct attrs;
    attrs.emplace("generator_sequence", std::vector<vsdk::ProtoValue>{vsdk::ProtoValue{"legacy"}});

    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config(attrs));
    auto result = service->infer(make_simple_inputs(), {});

    BOOST_REQUIRE(result);
    BOOST_CHECK(result->count("sample_times_sec") > 0);
    BOOST_CHECK(result->count("configurations_rads") > 0);
    BOOST_CHECK(result->count("velocities_rads_per_sec") > 0);
    BOOST_CHECK_EQUAL(result->count("accelerations_rads_per_sec2"), 0U);
}
#endif

BOOST_AUTO_TEST_CASE(dual_algorithm_prefers_totg) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    auto result = service->infer(make_simple_inputs(), {});

    BOOST_REQUIRE(result);
    // Default config runs both; totg result should be preferred (has accelerations)
    BOOST_CHECK(result->count("accelerations_rads_per_sec2") > 0);
}

// End-to-end through the service: a tcp_max_velocity + model table yields a trajectory whose
// realized Cartesian TCP speed (||J(q)*q_dot||, reconstructed from the output samples) respects the
// cap, and the limit is actually binding (peak close to the cap, not trivially under).
BOOST_AUTO_TEST_CASE(tcp_limit_respected_through_service) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    auto result = service->infer(make_2dof_tcp_inputs(true), {});

    BOOST_REQUIRE(result);
    const auto& cfg = boost::get<mlmodel::tensor_view<double>>(result->at("configurations_rads"));
    const auto& vel = boost::get<mlmodel::tensor_view<double>>(result->at("velocities_rads_per_sec"));
    const std::size_t n = cfg.shape(0);
    BOOST_REQUIRE(n > 0U);

    constexpr double cap = 0.3;
    double peak = 0.0;
    for (std::size_t s = 0; s < n; ++s) {
        const double q1 = cfg.flat((s * 2) + 0);
        const double q2 = cfg.flat((s * 2) + 1);
        const double v1 = vel.flat((s * 2) + 0);
        const double v2 = vel.flat((s * 2) + 1);
        // Cartesian TCP velocity = J(q) * q_dot for a planar 2-link arm (l1 = l2 = 1).
        const double s1 = std::sin(q1);
        const double c1 = std::cos(q1);
        const double s12 = std::sin(q1 + q2);
        const double c12 = std::cos(q1 + q2);
        const double vx = ((-s1 - s12) * v1) + ((-s12) * v2);
        const double vy = ((c1 + c12) * v1) + ((c12)*v2);
        const double speed = std::sqrt((vx * vx) + (vy * vy));
        peak = std::max(speed, peak);
    }
    BOOST_TEST(peak <= cap + 1e-3);  // respected
    BOOST_TEST(peak > cap * 0.5);    // and actually binding
}

BOOST_AUTO_TEST_CASE(tcp_velocity_without_model_table_throws) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    auto inputs = make_2dof_tcp_inputs(false);
    static const std::vector<double> tcp_max = {0.3};
    inputs.emplace("tcp_max_linear_velocity", mlmodel::make_tensor_view(tcp_max.data(), 1, {1}));
    BOOST_CHECK_THROW(service->infer(inputs, {}), std::invalid_argument);
}

// A model table supplied without the velocity cap is a half-specified TCP limit: the caller
// clearly intended one but would silently get an unconstrained trajectory, so it must throw.
BOOST_AUTO_TEST_CASE(model_table_without_tcp_velocity_throws) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    auto inputs = make_2dof_tcp_inputs(false);
    static const std::vector<double> model_table = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // joint0 at origin, axis z, revolute
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // joint1 +1x,      axis z, revolute
        1, 0, 0, 0, 0, 0, 0, 0, 0, 3,  // flange +1x,      fixed
    };
    inputs.emplace("kinematics_model_table", mlmodel::make_tensor_view(model_table.data(), 30, {3, 10}));
    BOOST_CHECK_THROW(service->infer(inputs, {}), std::invalid_argument);
}

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
// A TCP speed cap is a safety limit only the totg generator can enforce. When totg fails,
// the service must surface the error rather than fall back to a legacy trajectory that
// silently ignores the cap.
BOOST_AUTO_TEST_CASE(tcp_with_totg_failure_does_not_fall_back_to_legacy) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    auto inputs = make_2dof_tcp_inputs(false);
    static const std::vector<double> tcp_max = {0.3};
    // Three revolute joints yield a 3-column jacobian, which totg rejects against the
    // 2-DOF waypoints during trajectory creation. The legacy generator ignores TCP
    // inputs entirely and would succeed.
    static const std::vector<double> model_table = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // joint0 at origin, axis z, revolute
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // joint1 +1x,      axis z, revolute
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // joint2 +1x,      axis z, revolute
    };
    inputs.emplace("tcp_max_linear_velocity", mlmodel::make_tensor_view(tcp_max.data(), 1, {1}));
    inputs.emplace("kinematics_model_table", mlmodel::make_tensor_view(model_table.data(), 30, {3, 10}));
    BOOST_CHECK_THROW(service->infer(inputs, {}), std::invalid_argument);
}
#endif

// A generator sequence with no TCP-capable generator cannot honor a requested cap; the
// request must be rejected instead of returning an unconstrained trajectory.
BOOST_AUTO_TEST_CASE(tcp_with_legacy_only_generator_throws) {
    vsdk::ProtoStruct attrs;
    attrs.emplace("generator_sequence", std::vector<vsdk::ProtoValue>{vsdk::ProtoValue{"legacy"}});

    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config(attrs));
    BOOST_CHECK_THROW(service->infer(make_2dof_tcp_inputs(true), {}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(tcp_non_positive_velocity_throws) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    auto inputs = make_2dof_tcp_inputs(true);
    static const std::vector<double> bad = {0.0};
    inputs.erase("tcp_max_linear_velocity");
    inputs.emplace("tcp_max_linear_velocity", mlmodel::make_tensor_view(bad.data(), 1, {1}));
    BOOST_CHECK_THROW(service->infer(inputs, {}), std::invalid_argument);
}

// The optional TCP inputs are part of the input schema; clients discover inputs through
// metadata(), so both tensors must be listed there.
BOOST_AUTO_TEST_CASE(metadata_lists_tcp_inputs) {
    auto service = std::make_shared<mlmodel>(vsdk::Dependencies{}, make_config());
    const auto md = service->metadata({});
    const auto has_input = [&](std::string_view name) {
        return std::any_of(md.inputs.begin(), md.inputs.end(), [&](const auto& info) { return info.name == name; });
    };
    BOOST_CHECK(has_input("tcp_max_linear_velocity"));
    BOOST_CHECK(has_input("kinematics_model_table"));
}

BOOST_AUTO_TEST_SUITE_END()
