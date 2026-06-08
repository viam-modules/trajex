// trajex C ABI boost.test suite.
//
// Scope: the shim layer (marshaling, lifetime, key/dtype/shape
// validation, error reporting). The underlying TOTG algorithm has its
// own test coverage; this suite intentionally treats `totg_generate` as
// a glue function and validates only sanity-level invariants on its
// output -- not golden values.

#define BOOST_TEST_MODULE viam_trajex_capi
#include <boost/test/included/unit_test.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <viam/trajex/capi/capi.h>

namespace {

// RAII handle for viam_trajex_tensor_map_t. unique_ptr with a custom
// deleter is the standard pattern; the alias keeps test bodies readable.
struct tensor_map_deleter {
    void operator()(viam_trajex_tensor_map_t* p) const noexcept {
        viam_trajex_tensor_map_destroy(p);
    }
};
using map_owner = std::unique_ptr<viam_trajex_tensor_map_t, tensor_map_deleter>;

map_owner make_map() {
    map_owner m{viam_trajex_tensor_map_create()};
    BOOST_TEST_REQUIRE(m);
    return m;
}

// RAII handle for the heap-allocated diagnostic string returned by
// viam_trajex_totg_generate.
struct error_string_deleter {
    void operator()(const char* p) const noexcept {
        viam_trajex_string_destroy(p);
    }
};
using error_string = std::unique_ptr<const char, error_string_deleter>;

// Pair of (status, owned-error) returned by run_totg(). Lets callers
// inspect both fields without worrying about the const char* lifetime.
struct totg_result {
    int status = 0;
    error_string error;
};

// Wrap viam_trajex_totg_generate so the diagnostic-string lifetime is
// always RAII-managed even on success (where error.get() is nullptr,
// which the deleter no-ops).
totg_result run_totg(const viam_trajex_tensor_map_t* inputs, viam_trajex_tensor_map_t* outputs) {
    const char* raw = nullptr;
    const int status = viam_trajex_totg_generate(inputs, outputs, &raw);
    return {status, error_string{raw}};
}

// Aggregate the four view-out fields into one struct so test bodies can
// query a key cleanly in a single line.
struct view_fields {
    viam_trajex_dtype_t dtype = VIAM_TRAJEX_DTYPE_F64;
    std::size_t rank = 0;
    const std::size_t* dims = nullptr;
    const void* data = nullptr;
};

view_fields view_required(const viam_trajex_tensor_map_t* m, const char* key) {
    view_fields v;
    const int ret = viam_trajex_tensor_map_view(m, key, &v.dtype, &v.rank, &v.dims, &v.data);
    BOOST_TEST_REQUIRE(ret == 0);
    return v;
}

// Populate the given input map with a minimal valid TOTG input set: 3
// waypoints in 2-DOF space, modest limits, modest tolerance. Used as
// the baseline for the end-to-end test and as the starting point for
// error-path tests that override a single key.
void populate_simple_totg_inputs(viam_trajex_tensor_map_t* inputs) {
    // clang-format off
    const std::vector<double> waypoints = {
        0.0, 0.0,
        1.0, 0.5,
        2.0, 1.0,
    };
    // clang-format on
    const std::vector<std::size_t> waypoints_dims = {3, 2};
    BOOST_TEST_REQUIRE(
        viam_trajex_tensor_map_insert_f64(inputs, viam_trajex_totg_key_waypoints_rads, 2, waypoints_dims.data(), waypoints.data()) == 0);

    const std::vector<double> velocity_limits = {1.0, 1.0};
    const std::vector<double> acceleration_limits = {1.0, 1.0};
    const std::vector<std::size_t> dof_dims = {2};
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_f64(
                           inputs, viam_trajex_totg_key_velocity_limits_rads_per_sec, 1, dof_dims.data(), velocity_limits.data()) == 0);
    BOOST_TEST_REQUIRE(
        viam_trajex_tensor_map_insert_f64(
            inputs, viam_trajex_totg_key_acceleration_limits_rads_per_sec2, 1, dof_dims.data(), acceleration_limits.data()) == 0);

    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(inputs, viam_trajex_totg_key_path_tolerance_delta_rads, 0.01) == 0);
}

}  // namespace

// ============================================================================
// Tensor map round-trip
// ============================================================================

BOOST_AUTO_TEST_CASE(scalar_f64_roundtrip) {
    const auto m = make_map();
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(m.get(), "k", 3.14) == 0);

    const auto v = view_required(m.get(), "k");
    BOOST_TEST(v.dtype == VIAM_TRAJEX_DTYPE_F64);
    BOOST_TEST(v.rank == 1U);
    BOOST_TEST_REQUIRE(v.dims != nullptr);
    BOOST_TEST(v.dims[0] == 1U);
    BOOST_TEST_REQUIRE(v.data != nullptr);
    BOOST_TEST(*static_cast<const double*>(v.data) == 3.14);
}

BOOST_AUTO_TEST_CASE(scalar_i64_roundtrip) {
    const auto m = make_map();
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_i64(m.get(), "k", std::int64_t{42}) == 0);

    const auto v = view_required(m.get(), "k");
    BOOST_TEST(v.dtype == VIAM_TRAJEX_DTYPE_I64);
    BOOST_TEST(v.rank == 1U);
    BOOST_TEST_REQUIRE(v.dims != nullptr);
    BOOST_TEST(v.dims[0] == 1U);
    BOOST_TEST_REQUIRE(v.data != nullptr);
    BOOST_TEST(*static_cast<const std::int64_t*>(v.data) == 42);
}

BOOST_AUTO_TEST_CASE(array_f64_2d_roundtrip) {
    const auto m = make_map();
    const std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::vector<std::size_t> dims = {2, 3};

    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_f64(m.get(), "k", 2, dims.data(), data.data()) == 0);

    const auto v = view_required(m.get(), "k");
    BOOST_TEST(v.dtype == VIAM_TRAJEX_DTYPE_F64);
    BOOST_TEST(v.rank == 2U);
    BOOST_TEST(v.dims[0] == 2U);
    BOOST_TEST(v.dims[1] == 3U);

    const auto* out = static_cast<const double*>(v.data);
    for (std::size_t i = 0; i < data.size(); ++i) {
        BOOST_TEST(out[i] == data[i]);
    }
}

BOOST_AUTO_TEST_CASE(array_i64_1d_roundtrip) {
    const auto m = make_map();
    const std::vector<std::int64_t> data = {10, 20, 30, 40};
    const std::vector<std::size_t> dims = {4};

    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_i64(m.get(), "k", 1, dims.data(), data.data()) == 0);

    const auto v = view_required(m.get(), "k");
    BOOST_TEST(v.dtype == VIAM_TRAJEX_DTYPE_I64);
    BOOST_TEST(v.rank == 1U);
    BOOST_TEST(v.dims[0] == 4U);

    const auto* out = static_cast<const std::int64_t*>(v.data);
    for (std::size_t i = 0; i < data.size(); ++i) {
        BOOST_TEST(out[i] == data[i]);
    }
}

// ============================================================================
// Replace-on-duplicate
// ============================================================================

BOOST_AUTO_TEST_CASE(replace_on_duplicate_changes_value) {
    const auto m = make_map();
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(m.get(), "k", 1.0) == 0);
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(m.get(), "k", 2.0) == 0);

    const auto v = view_required(m.get(), "k");
    BOOST_TEST(v.dtype == VIAM_TRAJEX_DTYPE_F64);
    BOOST_TEST(*static_cast<const double*>(v.data) == 2.0);
}

BOOST_AUTO_TEST_CASE(replace_on_duplicate_changes_dtype) {
    const auto m = make_map();
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(m.get(), "k", 1.0) == 0);
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_i64(m.get(), "k", std::int64_t{99}) == 0);

    const auto v = view_required(m.get(), "k");
    BOOST_TEST(v.dtype == VIAM_TRAJEX_DTYPE_I64);
    BOOST_TEST(*static_cast<const std::int64_t*>(v.data) == 99);
}

// ============================================================================
// View pointer stability
// ============================================================================

// Per the documented contract, view-returned pointers remain valid until
// the map is destroyed or the corresponding key is replaced. Insertions
// of *other* keys must not invalidate the pointers. This depends on
// std::unordered_map's element-pointer-stability guarantee; if a future
// maintainer swaps the container, this test catches the regression.
BOOST_AUTO_TEST_CASE(view_pointer_stable_across_other_inserts) {
    const auto m = make_map();
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(m.get(), "a", 1.0) == 0);
    const auto v1 = view_required(m.get(), "a");

    // Insert several other keys, exercising any rehash that might happen.
    for (int i = 0; i < 32; ++i) {
        const std::string key = "filler_" + std::to_string(i);
        BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(m.get(), key.c_str(), static_cast<double>(i)) == 0);
    }

    const auto v2 = view_required(m.get(), "a");
    BOOST_TEST(v1.data == v2.data);
    BOOST_TEST(v1.dims == v2.dims);
    BOOST_TEST(*static_cast<const double*>(v2.data) == 1.0);
}

// Positive-direction check on the replacement contract: viewing after a
// replace returns the new data. We deliberately don't test the
// invalidated-pointer side -- dereferencing the old pointer would be
// undefined behavior, not a portable testable condition.
BOOST_AUTO_TEST_CASE(view_after_key_replacement_returns_new_data) {
    const auto m = make_map();
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(m.get(), "k", 1.0) == 0);
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(m.get(), "k", 7.0) == 0);

    const auto v = view_required(m.get(), "k");
    BOOST_TEST(*static_cast<const double*>(v.data) == 7.0);
}

// ============================================================================
// Tensor map error paths
// ============================================================================

BOOST_AUTO_TEST_CASE(view_missing_key_returns_one) {
    const auto m = make_map();
    viam_trajex_dtype_t dtype = VIAM_TRAJEX_DTYPE_F64;
    std::size_t rank = 0;
    const std::size_t* dims = nullptr;
    const void* data = nullptr;
    const int ret = viam_trajex_tensor_map_view(m.get(), "missing", &dtype, &rank, &dims, &data);
    BOOST_TEST(ret == 1);
}

BOOST_AUTO_TEST_CASE(insert_rejects_null_handle) {
    BOOST_TEST(viam_trajex_tensor_map_insert_scalar_f64(nullptr, "k", 1.0) == -1);
}

BOOST_AUTO_TEST_CASE(insert_rejects_null_key) {
    const auto m = make_map();
    BOOST_TEST(viam_trajex_tensor_map_insert_scalar_f64(m.get(), nullptr, 1.0) == -1);
}

BOOST_AUTO_TEST_CASE(insert_rejects_zero_rank) {
    const auto m = make_map();
    BOOST_TEST(viam_trajex_tensor_map_insert(m.get(), "k", VIAM_TRAJEX_DTYPE_F64, 0, nullptr, nullptr) == -1);
}

BOOST_AUTO_TEST_CASE(view_rejects_null_handle) {
    viam_trajex_dtype_t dtype = VIAM_TRAJEX_DTYPE_F64;
    std::size_t rank = 0;
    const std::size_t* dims = nullptr;
    const void* data = nullptr;
    BOOST_TEST(viam_trajex_tensor_map_view(nullptr, "k", &dtype, &rank, &dims, &data) == -1);
}

BOOST_AUTO_TEST_CASE(destroy_null_is_noop) {
    // Documented as no-op; should not crash.
    viam_trajex_tensor_map_destroy(nullptr);
    viam_trajex_string_destroy(nullptr);
}

// ============================================================================
// totg_generate end-to-end sanity
// ============================================================================

BOOST_AUTO_TEST_CASE(totg_generate_basic_sanity) {
    const auto inputs = make_map();
    const auto outputs = make_map();
    populate_simple_totg_inputs(inputs.get());

    const auto result = run_totg(inputs.get(), outputs.get());
    BOOST_TEST_REQUIRE(result.status == 0);
    BOOST_TEST_REQUIRE(!result.error);

    const auto times = view_required(outputs.get(), viam_trajex_totg_key_sample_times_sec);
    const auto configurations = view_required(outputs.get(), viam_trajex_totg_key_configurations_rads);
    const auto velocities = view_required(outputs.get(), viam_trajex_totg_key_velocities_rads_per_sec);
    const auto accelerations = view_required(outputs.get(), viam_trajex_totg_key_accelerations_rads_per_sec2);

    // All outputs are F64 per the schema.
    BOOST_TEST(times.dtype == VIAM_TRAJEX_DTYPE_F64);
    BOOST_TEST(configurations.dtype == VIAM_TRAJEX_DTYPE_F64);
    BOOST_TEST(velocities.dtype == VIAM_TRAJEX_DTYPE_F64);
    BOOST_TEST(accelerations.dtype == VIAM_TRAJEX_DTYPE_F64);

    // Ranks per the schema: times is 1D, the rest are 2D.
    BOOST_TEST_REQUIRE(times.rank == 1U);
    BOOST_TEST_REQUIRE(configurations.rank == 2U);
    BOOST_TEST_REQUIRE(velocities.rank == 2U);
    BOOST_TEST_REQUIRE(accelerations.rank == 2U);

    // n_samples is consistent across all four streams; n_dof matches the
    // input's velocity_limits.size() (here, 2).
    const std::size_t n_samples = times.dims[0];
    BOOST_TEST(n_samples > 0U);
    BOOST_TEST(configurations.dims[0] == n_samples);
    BOOST_TEST(configurations.dims[1] == 2U);
    BOOST_TEST(velocities.dims[0] == n_samples);
    BOOST_TEST(velocities.dims[1] == 2U);
    BOOST_TEST(accelerations.dims[0] == n_samples);
    BOOST_TEST(accelerations.dims[1] == 2U);

    // Sample times are monotonically increasing and start at t=0.
    const auto* t = static_cast<const double*>(times.data);
    BOOST_TEST(t[0] == 0.0);
    for (std::size_t i = 1; i < n_samples; ++i) {
        BOOST_TEST(t[i] > t[i - 1]);
    }

    // Endpoint configurations match the first and last waypoints
    // exactly (the trajectory must start and end at the requested
    // waypoints; if a glue bug transposed or scrambled the joint axes,
    // these break).
    const auto* c = static_cast<const double*>(configurations.data);
    BOOST_TEST(c[0] == 0.0);
    BOOST_TEST(c[1] == 0.0);
    const std::size_t last = (n_samples - 1) * 2;
    BOOST_TEST(c[last + 0] == 2.0);
    BOOST_TEST(c[last + 1] == 1.0);

    // No NaN or Inf anywhere.
    const auto all_finite = [](const double* arr, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(arr[i])) {
                return false;
            }
        }
        return true;
    };
    BOOST_TEST(all_finite(t, n_samples));
    BOOST_TEST(all_finite(c, n_samples * 2));
    BOOST_TEST(all_finite(static_cast<const double*>(velocities.data), n_samples * 2));
    BOOST_TEST(all_finite(static_cast<const double*>(accelerations.data), n_samples * 2));

    // Velocities are within the per-joint limits (small absolute
    // tolerance for floating-point slack). Acceleration is intentionally
    // not validated against limits -- known accel boundary violations in
    // trajex would force tolerance-tuning we don't want to be on the
    // hook for here.
    constexpr double k_velocity_tolerance = 1e-6;
    const auto* v = static_cast<const double*>(velocities.data);
    for (std::size_t s = 0; s < n_samples; ++s) {
        BOOST_TEST(std::abs(v[(s * 2) + 0]) <= 1.0 + k_velocity_tolerance);
        BOOST_TEST(std::abs(v[(s * 2) + 1]) <= 1.0 + k_velocity_tolerance);
    }
}

// ============================================================================
// totg_generate error paths
// ============================================================================

BOOST_AUTO_TEST_CASE(totg_generate_rejects_null_inputs) {
    const auto outputs = make_map();
    const auto result = run_totg(nullptr, outputs.get());
    BOOST_TEST(result.status == -1);
    BOOST_TEST_REQUIRE(result.error);
    BOOST_TEST(std::string(result.error.get()).size() > 0U);
}

BOOST_AUTO_TEST_CASE(totg_generate_rejects_null_outputs) {
    const auto inputs = make_map();
    populate_simple_totg_inputs(inputs.get());
    const auto result = run_totg(inputs.get(), nullptr);
    BOOST_TEST(result.status == -1);
    BOOST_TEST_REQUIRE(result.error);
    BOOST_TEST(std::string(result.error.get()).size() > 0U);
}

BOOST_AUTO_TEST_CASE(totg_generate_missing_required_input) {
    // Construct an input map with only one of the four required keys
    // populated; the missing waypoints_rads should be the diagnostic.
    const auto inputs = make_map();
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_scalar_f64(inputs.get(), viam_trajex_totg_key_path_tolerance_delta_rads, 0.01) == 0);

    const auto outputs = make_map();
    const auto result = run_totg(inputs.get(), outputs.get());
    BOOST_TEST(result.status == -1);
    BOOST_TEST_REQUIRE(result.error);
    BOOST_TEST(std::string(result.error.get()).find("waypoints_rads") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(totg_generate_wrong_dtype_on_required_input) {
    // Build the simple input set, then overwrite waypoints_rads with an
    // I64 tensor instead of F64. The schema check should reject.
    const auto inputs = make_map();
    populate_simple_totg_inputs(inputs.get());
    const std::vector<std::int64_t> bad = {0, 1, 2, 3};
    const std::vector<std::size_t> dims = {2, 2};
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_i64(inputs.get(), viam_trajex_totg_key_waypoints_rads, 2, dims.data(), bad.data()) ==
                       0);

    const auto outputs = make_map();
    const auto result = run_totg(inputs.get(), outputs.get());
    BOOST_TEST(result.status == -1);
    BOOST_TEST_REQUIRE(result.error);
    BOOST_TEST(std::string(result.error.get()).find("waypoints_rads") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(totg_generate_wrong_shape_on_velocity_limits) {
    // Build the simple input set, then overwrite velocity_limits with a
    // 1D [3] tensor instead of 1D [2]. n_dof is derived from
    // velocity_limits, so the mismatch should surface against
    // waypoints_rads.shape[1] or acceleration_limits.size().
    const auto inputs = make_map();
    populate_simple_totg_inputs(inputs.get());
    const std::vector<double> bad = {1.0, 1.0, 1.0};
    const std::vector<std::size_t> dims = {3};
    BOOST_TEST_REQUIRE(viam_trajex_tensor_map_insert_f64(
                           inputs.get(), viam_trajex_totg_key_velocity_limits_rads_per_sec, 1, dims.data(), bad.data()) == 0);

    const auto outputs = make_map();
    const auto result = run_totg(inputs.get(), outputs.get());
    BOOST_TEST(result.status == -1);
    BOOST_TEST_REQUIRE(result.error);
    BOOST_TEST(std::string(result.error.get()).size() > 0U);
}
