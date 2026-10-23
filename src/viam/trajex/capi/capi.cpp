#include <viam/trajex/capi/capi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <version>

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>
#endif

#include <viam/trajex/kinematics/chain.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/streaming/session.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/totg/waypoint_utils.hpp>
#include <viam/trajex/types/hertz.hpp>
#include <viam/trajex/types/xt.hpp>

// The opaque tensor-map type. Forward-declared in the C header; the full
// definition lives here so callers see only an opaque pointer.
//
// Storage is a variant over element type and rank. The alternatives are trajex's
// own types, so hand-off is a std::get rather than a copy.
//
// Rank is capped at 2 because that is all the API carries, and enumerating the
// ranks also puts the UB conversion out of reach.
struct viam_trajex_tensor_map {
    using tensor_value = std::variant<viam::trajex::xvector<double>,
                                      viam::trajex::xmatrix<double>,
                                      viam::trajex::xvector<std::int64_t>,
                                      viam::trajex::xmatrix<std::int64_t>>;

    // Transparent hasher for heterogeneous lookup. std::unordered_map's
    // C++20 heterogeneous lookup requires a hash type with `is_transparent`;
    // the standard provides std::equal_to<> (transparent since C++14) for the
    // equality side but no transparent hash. One templated operator() suffices
    // here because std::string, std::string_view, and const char* all convert
    // to std::string_view (which std::hash specializes on).
    struct transparent_hash {
        using is_transparent = void;
        template <typename T>
        std::size_t operator()(const T& key) const noexcept {
            return std::hash<std::string_view>{}(key);
        }
    };

    std::unordered_map<std::string, tensor_value, transparent_hash, std::equal_to<>> tensors;
};

// The opaque streaming session type. Wraps an in-place
// `viam::trajex::totg::streaming::session` plus a captured DOF count so empty-sample
// outputs (shape-zero tensors) still carry the correct second-dimension extent for
// the configuration / velocity / acceleration arrays.
struct viam_trajex_totg_streaming_session {
    viam::trajex::totg::streaming::session sess;
    std::size_t n_dof;

    viam_trajex_totg_streaming_session(viam::trajex::totg::path::options popts,
                                       viam::trajex::totg::trajectory::options topts,
                                       viam::trajex::types::hertz sample_rate,
                                       std::size_t dof)
        : sess(std::move(popts), std::move(topts), sample_rate), n_dof(dof) {}
};

// The opaque kinematic chain builder type. Building consumes the C++ builder, so it is held
// in an optional that build empties whether or not it succeeds; an empty one is what the
// header calls a builder that has already been built.
struct viam_trajex_kinematics_chain_builder {
    std::optional<viam::trajex::kinematics::chain::builder> builder{std::in_place};
};

// The opaque kinematic chain type.
struct viam_trajex_kinematics_chain {
    viam::trajex::kinematics::chain chain;
};

namespace {

// Default values for optional inputs to viam_trajex_totg_generate, per the
// schema documented in capi.h.
constexpr double k_default_colinearization_ratio = 0.0;
constexpr double k_default_dedup_tolerance = 1e-5;
constexpr double k_default_sampling_freq_hz = 100.0;

bool dtype_is_valid(viam_trajex_dtype_t dtype) noexcept {
    switch (dtype) {
        case VIAM_TRAJEX_DTYPE_F64:
        case VIAM_TRAJEX_DTYPE_I64:
            return true;
    }
    return false;
}

std::size_t dtype_size_bytes(viam_trajex_dtype_t dtype) {
    switch (dtype) {
        case VIAM_TRAJEX_DTYPE_F64:
            return sizeof(double);
        case VIAM_TRAJEX_DTYPE_I64:
            return sizeof(std::int64_t);
    }
    std::ostringstream oss;
    oss << "unknown dtype value: " << static_cast<int>(dtype);
    throw std::invalid_argument(oss.str());
}

// Map a variant alternative type to its dtype enum value. Constexpr so
// the std::visit dispatch in tensor_map_view can resolve at compile time.
template <typename T>
constexpr viam_trajex_dtype_t dtype_for() noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return VIAM_TRAJEX_DTYPE_F64;
    } else {
        static_assert(std::is_same_v<T, std::int64_t>);
        return VIAM_TRAJEX_DTYPE_I64;
    }
}

// Heterogeneous lookup on std::unordered_map (find by string_view against a
// std::string key, via the transparent hasher and equality) only reached
// libstdc++ in GCC 11 (P0919, __cpp_lib_generic_unordered_lookup). On the GCC 10
// (Focal) floor it is absent, so there we materialize a std::string key -- the
// transparent hasher/equality still route it correctly, at the cost of one
// allocation per lookup on that path only. Remove once the floor moves past GCC 10.
template <typename Map>
auto find_tensor(const Map& map, std::string_view key) {
#if defined(__cpp_lib_generic_unordered_lookup) && __cpp_lib_generic_unordered_lookup >= 201811L
    return map.find(key);
#else
    return map.find(std::string(key));
#endif
}

// Describes what the caller actually stored, for diagnostics when the
// alternative they asked for is not the one present.
std::string describe_tensor(const viam_trajex_tensor_map::tensor_value& value) {
    return std::visit(
        [](const auto& tensor) {
            using element_t = typename std::decay_t<decltype(tensor)>::value_type;
            std::ostringstream oss;
            oss << (std::is_same_v<element_t, double> ? "f64" : "i64") << " rank-" << tensor.dimension();
            return oss.str();
        },
        value);
}

// Names the alternative a call site asked for, to pair with describe_tensor.
template <typename A>
std::string describe_requested() {
    std::ostringstream oss;
    oss << (std::is_same_v<typename A::value_type, double> ? "f64" : "i64") << " rank-" << A::rank;
    return oss.str();
}

// `A` names both the element type and the rank, so a call site asking for a
// matrix cannot be handed a vector. Throws std::invalid_argument if the key is
// missing or holds a different alternative.
template <typename A>
const A& require_tensor(const viam_trajex_tensor_map& inputs, std::string_view key) {
    const auto it = find_tensor(inputs.tensors, key);
    if (it == inputs.tensors.end()) {
        std::ostringstream oss;
        oss << "missing required input: " << key;
        throw std::invalid_argument(oss.str());
    }
    const auto* ptr = std::get_if<A>(&it->second);
    if (ptr == nullptr) {
        std::ostringstream oss;
        oss << "input '" << key << "' is " << describe_tensor(it->second) << ", expected " << describe_requested<A>();
        throw std::invalid_argument(oss.str());
    }
    return *ptr;
}

// Resolve an optional input by key. Returns nullptr if the key is missing.
// Throws std::invalid_argument if the key is present but holds a different
// alternative than `A`.
template <typename A>
const A* find_optional_tensor(const viam_trajex_tensor_map& inputs, std::string_view key) {
    const auto it = find_tensor(inputs.tensors, key);
    if (it == inputs.tensors.end()) {
        return nullptr;
    }
    const auto* ptr = std::get_if<A>(&it->second);
    if (ptr == nullptr) {
        std::ostringstream oss;
        oss << "input '" << key << "' is " << describe_tensor(it->second) << ", expected " << describe_requested<A>();
        throw std::invalid_argument(oss.str());
    }
    return ptr;
}

// Rank is carried by the type the accessors return, so these check extent only.
template <typename T>
void require_size(const viam::trajex::xvector<T>& vector, std::string_view key, std::size_t expected_size) {
    if (vector.shape(0) != expected_size) {
        std::ostringstream oss;
        oss << "input '" << key << "' has wrong shape (expected 1D [" << expected_size << "])";
        throw std::invalid_argument(oss.str());
    }
}

template <typename T>
void require_scalar(const viam::trajex::xvector<T>& vector, std::string_view key) {
    if (vector.shape(0) != 1) {
        std::ostringstream oss;
        oss << "input '" << key << "' must be a scalar (shape [1])";
        throw std::invalid_argument(oss.str());
    }
}

// The C ABI passes points, quaternions and axes as bare arrays; these give them their C++ types.
std::array<double, 3> vector3_from(const double* values) {
    return {values[0], values[1], values[2]};
}

viam::trajex::kinematics::chain::joint_origin origin_from(const double* xyz, const double* wxyz) {
    return {.translation = vector3_from(xyz), .rotation = {wxyz[0], wxyz[1], wxyz[2], wxyz[3]}};
}

// The TCP limit a call asks for, if any. A chain must match the DOF even when no cap is set, so
// that a mismatch is reported by the call that made it rather than by the first one that caps.
// A cap has nothing to apply to without a chain, and is refused rather than ignored.
std::optional<viam::trajex::totg::trajectory::tcp_limits> tcp_limits_for(const viam_trajex_tensor_map& inputs,
                                                                         const viam_trajex_kinematics_chain* chain,
                                                                         std::size_t n_dof) {
    if (chain && chain->chain.dof() != n_dof) {
        std::ostringstream oss;
        oss << "kinematic chain has " << chain->chain.dof() << " rotating or translating joints, but the inputs have " << n_dof
            << " degrees of freedom";
        throw std::invalid_argument(oss.str());
    }
    const auto* cap = find_optional_tensor<viam::trajex::xvector<double>>(inputs, viam_trajex_totg_key_tcp_max_linear_speed);
    if (!cap) {
        return std::nullopt;
    }
    require_scalar(*cap, viam_trajex_totg_key_tcp_max_linear_speed);
    if (!chain) {
        std::ostringstream oss;
        oss << "input '" << viam_trajex_totg_key_tcp_max_linear_speed << "' requires a kinematic chain";
        throw std::invalid_argument(oss.str());
    }
    return viam::trajex::totg::trajectory::tcp_limits::from(chain->chain, (*cap)(0));
}

// Build a fresh output map from validated inputs. Calls directly into the
// trajex TOTG primitives (no planner abstraction: there is no algorithm
// fallback to orchestrate, and TOTG handles reversals natively). Throws on
// any failure; the extern "C" wrapper converts exceptions into the
// documented error contract.
viam_trajex_tensor_map totg_generate_impl(const viam_trajex_tensor_map& inputs, const viam_trajex_kinematics_chain* chain) {
    namespace totg = viam::trajex::totg;

    // Required inputs

    const auto& waypoints = require_tensor<viam::trajex::xmatrix<double>>(inputs, viam_trajex_totg_key_waypoints_rads);
    const auto n_dof = waypoints.shape(1);

    const auto& velocity_limits = require_tensor<viam::trajex::xvector<double>>(inputs, viam_trajex_totg_key_velocity_limits_rads_per_sec);
    require_size(velocity_limits, viam_trajex_totg_key_velocity_limits_rads_per_sec, n_dof);

    const auto& acceleration_limits =
        require_tensor<viam::trajex::xvector<double>>(inputs, viam_trajex_totg_key_acceleration_limits_rads_per_sec2);
    require_size(acceleration_limits, viam_trajex_totg_key_acceleration_limits_rads_per_sec2, n_dof);

    const auto& path_tolerance_xa = require_tensor<viam::trajex::xvector<double>>(inputs, viam_trajex_totg_key_path_tolerance_delta_rads);
    require_scalar(path_tolerance_xa, viam_trajex_totg_key_path_tolerance_delta_rads);
    const auto path_tolerance = path_tolerance_xa(0);

    // Optional inputs

    auto colinearization_ratio = k_default_colinearization_ratio;
    if (const auto* xa = find_optional_tensor<viam::trajex::xvector<double>>(inputs, viam_trajex_totg_key_path_colinearization_ratio)) {
        require_scalar(*xa, viam_trajex_totg_key_path_colinearization_ratio);
        colinearization_ratio = (*xa)(0);
    }

    auto dedup_tolerance = k_default_dedup_tolerance;
    if (const auto* xa =
            find_optional_tensor<viam::trajex::xvector<double>>(inputs, viam_trajex_totg_key_waypoint_deduplication_tolerance_rads)) {
        require_scalar(*xa, viam_trajex_totg_key_waypoint_deduplication_tolerance_rads);
        dedup_tolerance = (*xa)(0);
    }

    auto sampling_freq = k_default_sampling_freq_hz;
    if (const auto* xa = find_optional_tensor<viam::trajex::xvector<double>>(inputs, viam_trajex_totg_key_trajectory_sampling_freq_hz)) {
        require_scalar(*xa, viam_trajex_totg_key_trajectory_sampling_freq_hz);
        sampling_freq = (*xa)(0);
    }

    // Hand the waypoints xarray directly to the accumulator: it borrows
    // (the input map owns and outlives this function), no copy.
    auto accumulator = totg::deduplicate_waypoints(totg::waypoint_accumulator{waypoints}, dedup_tolerance);

    auto path_opts = totg::path::options{}.set_max_blend_deviation(path_tolerance);
    if (colinearization_ratio > 0.0) {
        path_opts.set_max_linear_deviation(path_tolerance * colinearization_ratio);
    }
    auto p = totg::path::create(accumulator, path_opts);

    totg::trajectory::options topts;
    topts.max_velocity = velocity_limits;
    topts.max_acceleration = acceleration_limits;
    topts.tcp = tcp_limits_for(inputs, chain, n_dof);
    auto traj = totg::trajectory::create(std::move(p), std::move(topts));

    auto sampler = totg::uniform_sampler::quantized_for_trajectory(traj, viam::trajex::types::hertz{sampling_freq});

    // Preallocate the output xarrays at their final shape.
    // `calculate_quantized_samples` is the same calculation the sampler
    // factory used internally, so the counts agree.
    const auto n_samples = totg::uniform_sampler::calculate_quantized_samples(traj.duration().count(), sampling_freq);
    if (n_samples == 0) {
        throw std::logic_error("internal: sampler produced no samples");
    }

    viam::trajex::xvector<double> times(viam::trajex::xvector<double>::shape_type{n_samples});
    viam::trajex::xmatrix<double> configurations(viam::trajex::xmatrix<double>::shape_type{n_samples, n_dof});
    viam::trajex::xmatrix<double> velocities(viam::trajex::xmatrix<double>::shape_type{n_samples, n_dof});
    viam::trajex::xmatrix<double> accelerations(viam::trajex::xmatrix<double>::shape_type{n_samples, n_dof});

    std::size_t idx = 0;
    for (const auto& sample : traj.samples(sampler)) {
        if (idx >= n_samples) {
            throw std::logic_error("internal: sampler produced more samples than expected");
        }
        times(idx) = sample.time.count();
        xt::view(configurations, idx, xt::all()) = sample.configuration;
        xt::view(velocities, idx, xt::all()) = sample.velocity;
        xt::view(accelerations, idx, xt::all()) = sample.acceleration;
        ++idx;
    }
    if (idx != n_samples) {
        throw std::logic_error("internal: sampler produced fewer samples than expected");
    }

    // Move the freshly-built xarrays into the output map. No copy; the map
    // now owns them. The boundary wrapper move-assigns the whole map into
    // the caller's outputs only on a fully successful build.
    viam_trajex_tensor_map out;
    out.tensors.emplace(viam_trajex_totg_key_sample_times_sec, std::move(times));
    out.tensors.emplace(viam_trajex_totg_key_configurations_rads, std::move(configurations));
    out.tensors.emplace(viam_trajex_totg_key_velocities_rads_per_sec, std::move(velocities));
    out.tensors.emplace(viam_trajex_totg_key_accelerations_rads_per_sec2, std::move(accelerations));

    return out;
}

// Build a streaming session from a config tensor map. Throws on missing required keys,
// dtype mismatches, or shape mismatches; the extern "C" wrapper converts exceptions to
// NULL with diagnostic.
std::unique_ptr<viam_trajex_totg_streaming_session> build_streaming_session(const viam_trajex_tensor_map& options,
                                                                            const viam_trajex_kinematics_chain* chain) {
    namespace totg = viam::trajex::totg;

    const auto& velocity_limits = require_tensor<viam::trajex::xvector<double>>(options, viam_trajex_totg_key_velocity_limits_rads_per_sec);
    const auto n_dof = velocity_limits.shape(0);

    const auto& acceleration_limits =
        require_tensor<viam::trajex::xvector<double>>(options, viam_trajex_totg_key_acceleration_limits_rads_per_sec2);
    require_size(acceleration_limits, viam_trajex_totg_key_acceleration_limits_rads_per_sec2, n_dof);

    const auto& path_tolerance_xa = require_tensor<viam::trajex::xvector<double>>(options, viam_trajex_totg_key_path_tolerance_delta_rads);
    require_scalar(path_tolerance_xa, viam_trajex_totg_key_path_tolerance_delta_rads);
    const auto path_tolerance = path_tolerance_xa(0);

    const auto& sample_rate_xa = require_tensor<viam::trajex::xvector<double>>(options, viam_trajex_totg_key_trajectory_sampling_freq_hz);
    require_scalar(sample_rate_xa, viam_trajex_totg_key_trajectory_sampling_freq_hz);
    const auto sample_rate_hz = sample_rate_xa(0);

    auto colinearization_ratio = k_default_colinearization_ratio;
    if (const auto* xa = find_optional_tensor<viam::trajex::xvector<double>>(options, viam_trajex_totg_key_path_colinearization_ratio)) {
        require_scalar(*xa, viam_trajex_totg_key_path_colinearization_ratio);
        colinearization_ratio = (*xa)(0);
    }

    auto path_opts = totg::path::options{}.set_max_blend_deviation(path_tolerance);
    if (colinearization_ratio > 0.0) {
        path_opts.set_max_linear_deviation(path_tolerance * colinearization_ratio);
    }

    // The limit holds its own copy of the chain, and the session holds the options, which is what
    // lets the caller destroy the chain as soon as construction returns.
    totg::trajectory::options trajectory_opts;
    trajectory_opts.max_velocity = velocity_limits;
    trajectory_opts.max_acceleration = acceleration_limits;
    trajectory_opts.tcp = tcp_limits_for(options, chain, n_dof);

    return std::make_unique<viam_trajex_totg_streaming_session>(
        std::move(path_opts), std::move(trajectory_opts), viam::trajex::types::hertz{sample_rate_hz}, n_dof);
}

// Materialize a vector of trajectory samples into the four output tensors of the
// streaming sample schema. The shape-zero case (no samples) still emits all four keys
// with the right DOF-dimension extent, so the caller can read the shape uniformly.
viam_trajex_tensor_map materialize_samples(const std::vector<struct viam::trajex::totg::trajectory::sample>& samples, std::size_t n_dof) {
    const std::size_t n = samples.size();

    viam::trajex::xvector<double> times = xt::zeros<double>(viam::trajex::xvector<double>::shape_type{n});
    viam::trajex::xmatrix<double> configurations = xt::zeros<double>(viam::trajex::xmatrix<double>::shape_type{n, n_dof});
    viam::trajex::xmatrix<double> velocities = xt::zeros<double>(viam::trajex::xmatrix<double>::shape_type{n, n_dof});
    viam::trajex::xmatrix<double> accelerations = xt::zeros<double>(viam::trajex::xmatrix<double>::shape_type{n, n_dof});

    for (std::size_t i = 0; i < n; ++i) {
        times(i) = samples[i].time.count();
        xt::view(configurations, i, xt::all()) = samples[i].configuration;
        xt::view(velocities, i, xt::all()) = samples[i].velocity;
        xt::view(accelerations, i, xt::all()) = samples[i].acceleration;
    }

    viam_trajex_tensor_map out;
    out.tensors.emplace(viam_trajex_totg_key_sample_times_sec, std::move(times));
    out.tensors.emplace(viam_trajex_totg_key_configurations_rads, std::move(configurations));
    out.tensors.emplace(viam_trajex_totg_key_velocities_rads_per_sec, std::move(velocities));
    out.tensors.emplace(viam_trajex_totg_key_accelerations_rads_per_sec2, std::move(accelerations));
    return out;
}

// Allocate a NUL-terminated copy of `msg` via new[]. Bad-alloc inside a
// boundary catch handler is intentional fail-fast (per the design doc):
// cascading bad_alloc during exception handling triggers std::terminate
// per language rules, and v0 makes no attempt to recover. Callers must
// not wrap this in their own try.
const char* duplicate_error_string(const char* msg) {
    const auto len = std::strlen(msg);
    auto* buf = new char[len + 1];
    std::memcpy(buf, msg, len + 1);
    return buf;
}

// Translate an extend kind into its C ABI counterpart. Both enumerations pin the same
// integer values, so a cast would work today, but writing the mapping out means the compiler
// flags the omission if a kind is ever added on the C++ side alone.
viam_trajex_totg_streaming_session_extend_kind_t to_capi_extend_kind(viam::trajex::totg::streaming::session::extend_result::kinds kind) {
    using kinds = viam::trajex::totg::streaming::session::extend_result::kinds;
    switch (kind) {
        case kinds::k_first_build:
            return VIAM_TRAJEX_TOTG_STREAMING_SESSION_EXTEND_FIRST_BUILD;
        case kinds::k_pivot:
            return VIAM_TRAJEX_TOTG_STREAMING_SESSION_EXTEND_PIVOT;
        case kinds::k_staged_branch_sampled:
            return VIAM_TRAJEX_TOTG_STREAMING_SESSION_EXTEND_STAGED_BRANCH_SAMPLED;
        case kinds::k_staged_unsamplable:
            return VIAM_TRAJEX_TOTG_STREAMING_SESSION_EXTEND_STAGED_UNSAMPLABLE;
        case kinds::k_staged_again:
            return VIAM_TRAJEX_TOTG_STREAMING_SESSION_EXTEND_STAGED_AGAIN;
        case kinds::k_noop:
            return VIAM_TRAJEX_TOTG_STREAMING_SESSION_EXTEND_NOOP;
    }
    throw std::logic_error("unmapped streaming extend kind");
}

// A time the session did not compute crosses the ABI as NaN. Zero would be indistinguishable
// from a branch that landed exactly on the last emitted sample, which is reachable, and
// leaving the parameter alone would oblige every caller to initialize it first.
double seconds_to_capi(const std::optional<viam::trajex::totg::trajectory::seconds>& value) {
    return value ? value->count() : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

extern "C" {

// Schema key constants. Defined once here; the strings are part of the
// stable ABI and match `service::mlmodel::infer`'s schema 1:1.
const char viam_trajex_totg_key_acceleration_limits_rads_per_sec2[] = "acceleration_limits_rads_per_sec2";
const char viam_trajex_totg_key_accelerations_rads_per_sec2[] = "accelerations_rads_per_sec2";
const char viam_trajex_totg_key_configurations_rads[] = "configurations_rads";
const char viam_trajex_totg_key_path_colinearization_ratio[] = "path_colinearization_ratio";
const char viam_trajex_totg_key_path_tolerance_delta_rads[] = "path_tolerance_delta_rads";
const char viam_trajex_totg_key_sample_times_sec[] = "sample_times_sec";
const char viam_trajex_totg_key_tcp_max_linear_speed[] = "tcp_max_linear_speed";
const char viam_trajex_totg_key_trajectory_sampling_freq_hz[] = "trajectory_sampling_freq_hz";
const char viam_trajex_totg_key_velocities_rads_per_sec[] = "velocities_rads_per_sec";
const char viam_trajex_totg_key_velocity_limits_rads_per_sec[] = "velocity_limits_rads_per_sec";
const char viam_trajex_totg_key_waypoint_deduplication_tolerance_rads[] = "waypoint_deduplication_tolerance_rads";
const char viam_trajex_totg_key_waypoints_rads[] = "waypoints_rads";

viam_trajex_tensor_map_t* viam_trajex_tensor_map_create(void) {
    try {
        return new viam_trajex_tensor_map;
    } catch (...) {
        return nullptr;
    }
}

void viam_trajex_tensor_map_destroy(viam_trajex_tensor_map_t* tensor_map) {
    try {
        delete tensor_map;
    } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Variant of xtensor destruction is noexcept under our element
        // types; this catch is purely a boundary discipline guarantee.
    }
}

int viam_trajex_tensor_map_insert(viam_trajex_tensor_map_t* tensor_map,
                                  const char* key,
                                  viam_trajex_dtype_t dtype,
                                  std::size_t rank,
                                  const std::size_t* dims,
                                  const void* data) {
    try {
        // Validated here and nowhere else. Past this point rank is part of the
        // stored type.
        if (!tensor_map || !key || rank < 1 || rank > 2 || !dims || !data) {
            return -1;
        }
        if (!dtype_is_valid(dtype)) {
            return -1;
        }
        std::size_t total_elements = 1;
        for (std::size_t i = 0; i < rank; ++i) {
            if (dims[i] < 1) {
                return -1;
            }
            total_elements *= dims[i];
        }
        const auto total_bytes = total_elements * dtype_size_bytes(dtype);

        // Build the replacement array fully before touching the map, so a
        // mid-call allocation failure leaves any prior entry intact.
        // Move-assignment into the map slot is noexcept under our element
        // types (xtensor move is noexcept).
        const auto store = [&]<typename A>(std::type_identity<A>) {
            typename A::shape_type shape{};
            std::copy_n(dims, A::rank, shape.begin());
            A tensor(shape);
            std::memcpy(tensor.data(), data, total_bytes);
            tensor_map->tensors.insert_or_assign(std::string(key), std::move(tensor));
        };

        switch (dtype) {
            case VIAM_TRAJEX_DTYPE_F64:
                if (rank == 1) {
                    store(std::type_identity<viam::trajex::xvector<double>>{});
                } else {
                    store(std::type_identity<viam::trajex::xmatrix<double>>{});
                }
                break;
            case VIAM_TRAJEX_DTYPE_I64:
                if (rank == 1) {
                    store(std::type_identity<viam::trajex::xvector<std::int64_t>>{});
                } else {
                    store(std::type_identity<viam::trajex::xmatrix<std::int64_t>>{});
                }
                break;
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

int viam_trajex_tensor_map_insert_f64(
    viam_trajex_tensor_map_t* tensor_map, const char* key, std::size_t rank, const std::size_t* dims, const double* data) {
    return viam_trajex_tensor_map_insert(tensor_map, key, VIAM_TRAJEX_DTYPE_F64, rank, dims, static_cast<const void*>(data));
}

int viam_trajex_tensor_map_insert_i64(
    viam_trajex_tensor_map_t* tensor_map, const char* key, std::size_t rank, const std::size_t* dims, const std::int64_t* data) {
    return viam_trajex_tensor_map_insert(tensor_map, key, VIAM_TRAJEX_DTYPE_I64, rank, dims, static_cast<const void*>(data));
}

int viam_trajex_tensor_map_insert_scalar_f64(viam_trajex_tensor_map_t* tensor_map, const char* key, double value) {
    const std::size_t dim = 1;
    return viam_trajex_tensor_map_insert_f64(tensor_map, key, 1, &dim, &value);
}

int viam_trajex_tensor_map_insert_scalar_i64(viam_trajex_tensor_map_t* tensor_map, const char* key, std::int64_t value) {
    const std::size_t dim = 1;
    return viam_trajex_tensor_map_insert_i64(tensor_map, key, 1, &dim, &value);
}

int viam_trajex_tensor_map_view(const viam_trajex_tensor_map_t* tensor_map,
                                const char* key,
                                viam_trajex_dtype_t* dtype_out,
                                std::size_t* rank_out,
                                const std::size_t** dims_out,
                                const void** data_out) {
    try {
        if (!tensor_map || !key || !dtype_out || !rank_out || !dims_out || !data_out) {
            return -1;
        }
        const auto it = find_tensor(tensor_map->tensors, key);
        if (it == tensor_map->tensors.end()) {
            return 1;
        }
        std::visit(
            [&](const auto& xarray) {
                using element_t = typename std::decay_t<decltype(xarray)>::value_type;
                *dtype_out = dtype_for<element_t>();
                *rank_out = xarray.dimension();
                *dims_out = xarray.shape().data();
                *data_out = static_cast<const void*>(xarray.data());
            },
            it->second);
        return 0;
    } catch (...) {
        return -1;
    }
}

void viam_trajex_string_destroy(const char* string) {
    try {
        // Cast away const to pair with the new[] allocation in
        // duplicate_error_string. The const on the parameter is a
        // read-only contract for the caller; the destroyer knows the
        // allocation pattern and owns the buffer.
        delete[] const_cast<char*>(string);
    } catch (...) {  // NOLINT(bugprone-empty-catch)
        // char has trivial destruction; this catch is purely a boundary
        // discipline guarantee.
    }
}

viam_trajex_kinematics_chain_builder_t* viam_trajex_kinematics_chain_builder_create(void) {
    try {
        return new viam_trajex_kinematics_chain_builder;
    } catch (...) {
        return nullptr;
    }
}

void viam_trajex_kinematics_chain_builder_destroy(viam_trajex_kinematics_chain_builder_t* builder) {
    try {
        delete builder;
    } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Defensive boundary; destroying a vector of joints does not throw.
    }
}

int viam_trajex_kinematics_chain_builder_extend_fixed(viam_trajex_kinematics_chain_builder_t* builder,
                                                      const double xyz[3],
                                                      const double wxyz[4]) {
    try {
        if (!builder || !builder->builder || !xyz || !wxyz) {
            return -1;
        }
        builder->builder->extend(origin_from(xyz, wxyz));
        return 0;
    } catch (...) {
        return -1;
    }
}

int viam_trajex_kinematics_chain_builder_extend_rotational(viam_trajex_kinematics_chain_builder_t* builder,
                                                           const double xyz[3],
                                                           const double wxyz[4],
                                                           const double axis[3]) {
    try {
        if (!builder || !builder->builder || !xyz || !wxyz || !axis) {
            return -1;
        }
        builder->builder->extend(origin_from(xyz, wxyz), viam::trajex::kinematics::chain::rotation_axis{vector3_from(axis)});
        return 0;
    } catch (...) {
        return -1;
    }
}

int viam_trajex_kinematics_chain_builder_extend_translational(viam_trajex_kinematics_chain_builder_t* builder,
                                                              const double xyz[3],
                                                              const double wxyz[4],
                                                              const double axis[3]) {
    try {
        if (!builder || !builder->builder || !xyz || !wxyz || !axis) {
            return -1;
        }
        builder->builder->extend(origin_from(xyz, wxyz), viam::trajex::kinematics::chain::translation_axis{vector3_from(axis)});
        return 0;
    } catch (...) {
        return -1;
    }
}

viam_trajex_kinematics_chain_t* viam_trajex_kinematics_chain_builder_build(viam_trajex_kinematics_chain_builder_t* builder,
                                                                           const char** error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    try {
        if (!builder) {
            throw std::invalid_argument("builder is null");
        }
        if (!builder->builder) {
            throw std::invalid_argument("builder has already been built");
        }
        // Take the builder out before building, so that it is spent even if build throws.
        auto consumed = std::move(*builder->builder);
        builder->builder.reset();
        return new viam_trajex_kinematics_chain{std::move(consumed).build()};
    } catch (const std::exception& e) {
        if (error_out) {
            *error_out = duplicate_error_string(e.what());
        }
        return nullptr;
    } catch (...) {
        if (error_out) {
            *error_out = duplicate_error_string("unknown exception");
        }
        return nullptr;
    }
}

void viam_trajex_kinematics_chain_destroy(viam_trajex_kinematics_chain_t* chain) {
    try {
        delete chain;
    } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Defensive boundary; destroying a vector of joints does not throw.
    }
}

int viam_trajex_kinematics_chain_end_position(
    const viam_trajex_kinematics_chain_t* chain, const double* q, std::size_t n_q, double out[3], const char** error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    try {
        if (!chain) {
            throw std::invalid_argument("chain is null");
        }
        if (!q && n_q != 0) {
            throw std::invalid_argument("q is null");
        }
        if (!out) {
            throw std::invalid_argument("out is null");
        }
        viam::trajex::xvector<double> joints(viam::trajex::xvector<double>::shape_type{n_q});
        std::copy_n(q, n_q, joints.begin());
        const auto position = chain->chain.end_position(joints);
        std::ranges::copy(position, out);
        return 0;
    } catch (const std::exception& e) {
        if (error_out) {
            *error_out = duplicate_error_string(e.what());
        }
        return -1;
    } catch (...) {
        if (error_out) {
            *error_out = duplicate_error_string("unknown exception");
        }
        return -1;
    }
}

// The `restrict` qualifier on the declared parameters expands to nothing
// in C++ (per VIAM_TRAJEX_RESTRICT's C++ branch), so the definition
// signature is the same as the declaration's once preprocessed. The macro
// itself is not visible here because the header push/pop'd it.
int viam_trajex_totg_generate(const viam_trajex_kinematics_chain_t* chain,
                              const viam_trajex_tensor_map_t* inputs,
                              viam_trajex_tensor_map_t* outputs,
                              const char** error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    try {
        if (!inputs) {
            throw std::invalid_argument("inputs is null");
        }
        if (!outputs) {
            throw std::invalid_argument("outputs is null");
        }
        // Build the result locally first; only on success do we
        // move-assign into the caller's map. Move-assignment of the
        // underlying unordered_map is noexcept under our element types,
        // so success is observable as a single atomic replacement of the
        // caller's map contents.
        *outputs = totg_generate_impl(*inputs, chain);
        return 0;
    } catch (const std::exception& e) {
        if (error_out) {
            *error_out = duplicate_error_string(e.what());
        }
        return -1;
    } catch (...) {
        if (error_out) {
            *error_out = duplicate_error_string("unknown exception");
        }
        return -1;
    }
}

viam_trajex_totg_streaming_session_t* viam_trajex_totg_streaming_session_create(const viam_trajex_kinematics_chain_t* chain,
                                                                                const viam_trajex_tensor_map_t* options,
                                                                                const char** error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    try {
        if (!options) {
            throw std::invalid_argument("options is null");
        }
        return build_streaming_session(*options, chain).release();
    } catch (const std::exception& e) {
        if (error_out) {
            *error_out = duplicate_error_string(e.what());
        }
        return nullptr;
    } catch (...) {
        if (error_out) {
            *error_out = duplicate_error_string("unknown exception");
        }
        return nullptr;
    }
}

void viam_trajex_totg_streaming_session_destroy(viam_trajex_totg_streaming_session_t* session) {
    try {
        delete session;
    } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Defensive boundary; session destructor should be noexcept.
    }
}

int viam_trajex_totg_streaming_session_extend(viam_trajex_totg_streaming_session_t* session,
                                              const viam_trajex_tensor_map_t* batch,
                                              viam_trajex_totg_streaming_session_extend_kind_t* kind_out,
                                              double* branch_slack_sec_out,
                                              double* delta_total_duration_sec_out,
                                              const char** error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    try {
        if (!session) {
            throw std::invalid_argument("session is null");
        }
        if (!batch) {
            throw std::invalid_argument("batch is null");
        }
        const auto& waypoints = require_tensor<viam::trajex::xmatrix<double>>(*batch, viam_trajex_totg_key_waypoints_rads);
        if (waypoints.shape(1) != session->n_dof) {
            throw std::invalid_argument("waypoints_rads DOF does not match session DOF");
        }
        // Accumulator borrows from `waypoints` (owned by the input map, which the caller
        // guarantees outlives this call). The session's extend internally copies any
        // waypoints it retains, so the accumulator's lifetime ending at function exit is
        // fine.
        const viam::trajex::totg::waypoint_accumulator acc(waypoints);
        const auto result = session->sess.extend(acc);

        // Report only once the call has succeeded. A call that threw has no outcome to
        // describe, and the header promises to leave these parameters as the caller left them.
        if (kind_out) {
            *kind_out = to_capi_extend_kind(result.kind);
        }
        if (branch_slack_sec_out) {
            *branch_slack_sec_out = seconds_to_capi(result.branch_slack);
        }
        if (delta_total_duration_sec_out) {
            *delta_total_duration_sec_out = result.delta_total_duration.count();
        }
        return 0;
    } catch (const std::exception& e) {
        if (error_out) {
            *error_out = duplicate_error_string(e.what());
        }
        return -1;
    } catch (...) {
        if (error_out) {
            *error_out = duplicate_error_string("unknown exception");
        }
        return -1;
    }
}

void viam_trajex_totg_streaming_session_start_staging(viam_trajex_totg_streaming_session_t* session) {
    session->sess.start_staging();
}

int viam_trajex_totg_streaming_session_sample_next(viam_trajex_totg_streaming_session_t* session,
                                                   std::size_t n,
                                                   viam_trajex_tensor_map_t* outputs,
                                                   const char** error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    try {
        if (!session) {
            throw std::invalid_argument("session is null");
        }
        if (!outputs) {
            throw std::invalid_argument("outputs is null");
        }
        auto samples = session->sess.sample_next(n);
        *outputs = materialize_samples(samples, session->n_dof);
        return 0;
    } catch (const std::exception& e) {
        if (error_out) {
            *error_out = duplicate_error_string(e.what());
        }
        return -1;
    } catch (...) {
        if (error_out) {
            *error_out = duplicate_error_string("unknown exception");
        }
        return -1;
    }
}

int viam_trajex_totg_streaming_session_sample_at_least(viam_trajex_totg_streaming_session_t* session,
                                                       double horizon_sec,
                                                       viam_trajex_tensor_map_t* outputs,
                                                       const char** error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    try {
        if (!session) {
            throw std::invalid_argument("session is null");
        }
        if (!outputs) {
            throw std::invalid_argument("outputs is null");
        }
        auto samples = session->sess.sample_at_least(viam::trajex::totg::trajectory::seconds{horizon_sec});
        *outputs = materialize_samples(samples, session->n_dof);
        return 0;
    } catch (const std::exception& e) {
        if (error_out) {
            *error_out = duplicate_error_string(e.what());
        }
        return -1;
    } catch (...) {
        if (error_out) {
            *error_out = duplicate_error_string("unknown exception");
        }
        return -1;
    }
}

void viam_trajex_totg_streaming_session_current_time_sec(const viam_trajex_totg_streaming_session_t* session, double* out) {
    *out = session->sess.current_time().count();
}

void viam_trajex_totg_streaming_session_generation_count(const viam_trajex_totg_streaming_session_t* session, std::int64_t* out) {
    *out = static_cast<std::int64_t>(session->sess.trajectory_generation_count());
}

void viam_trajex_totg_streaming_session_has_active_trajectory(const viam_trajex_totg_streaming_session_t* session, int* out) {
    *out = session->sess.active_trajectory() != nullptr ? 1 : 0;
}

void viam_trajex_totg_streaming_session_active_duration_sec(const viam_trajex_totg_streaming_session_t* session, double* out) {
    const auto* active = session->sess.active_trajectory();
    *out = active ? active->duration().count() : 0.0;
}

void viam_trajex_totg_streaming_session_remaining_active_duration_sec(const viam_trajex_totg_streaming_session_t* session, double* out) {
    *out = session->sess.remaining_active_duration().count();
}

void viam_trajex_totg_streaming_session_remaining_total_duration_sec(const viam_trajex_totg_streaming_session_t* session, double* out) {
    *out = session->sess.remaining_total_duration().count();
}

}  // extern "C"
