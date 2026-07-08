///
/// @file
///
/// @brief trajex C ABI: stable, language-agnostic entry points for invoking trajex from non-C++ callers (cgo, FFI, etc.).
///
/// The contract documented here is the source of truth -- there is no separate IDL or schema artifact.
///
/// ## Naming
///
/// Shared infrastructure carries the prefix `viam_trajex_`. Algorithm entry points additionally carry the C++ namespace
/// segment of the algorithm they invoke, e.g. `viam_trajex_totg_*`.
///
/// ## Return values
///
/// All entry points return `int` status codes. The convention is `0` on success, `-1` on error (bad arguments,
/// allocation failure, or any caught exception), and `1` reserved for expected non-error non-success outcomes --
/// currently only `viam_trajex_tensor_map_view` uses `1` to signal a missing key. Per-function documentation lists
/// which codes each entry point produces.
///
/// ## Destruction
///
/// All `*_destroy` entry points accept NULL as a no-op, consistent with `free(NULL)`.
///
/// ## Error reporting
///
/// Only `viam_trajex_totg_generate` produces a structured diagnostic via its `error_out` parameter. The tensor-map
/// operations report failure as a coarse-grained `-1` return without diagnostic; documented failure modes are listed
/// per function.
///
/// ## Threading
///
/// The library carries no global mutable state. Calls into independent `viam_trajex_tensor_map_t` handles from
/// different threads need no external synchronization. Concurrent operations on the same handle are the caller's
/// responsibility, matching the model of `std::map`.
///

#pragma once

#include <stddef.h>
#include <stdint.h>

// Portable spelling for the C99 `restrict` qualifier. In C99+ it is a keyword; in C++ it is not standard, and we
// deliberately decline the non-standard `__restrict` extension here -- the C++ expansion is empty, so C++ callers honor
// distinctness by documentation alone. The macro is wrapped in #pragma push_macro / pop_macro so an inclusion does not
// silently clobber any prior caller-side definition.
#pragma push_macro("VIAM_TRAJEX_RESTRICT")
#ifdef VIAM_TRAJEX_RESTRICT
#undef VIAM_TRAJEX_RESTRICT
#endif
#ifdef __cplusplus
#define VIAM_TRAJEX_RESTRICT
#else
#define VIAM_TRAJEX_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

///
/// @defgroup viam_trajex_dtype Tensor element data type
/// @{
///

///
/// Element data type carried in a tensor.
///
/// Append-only; existing integer values are stable across future additions.
///
typedef enum {                  // NOLINT(performance-enum-size)
    VIAM_TRAJEX_DTYPE_F64 = 1,  ///< IEEE 754 double-precision binary floating point
    VIAM_TRAJEX_DTYPE_I64 = 2,  ///< Signed 64-bit two's-complement integer
} viam_trajex_dtype_t;

/// @}

///
/// @defgroup viam_trajex_tensor_map Named tensor map
///
/// Opaque handle to a key -> tensor map. Used to pass heterogeneous inputs into the algorithm entry points and to
/// receive their outputs.
///
/// @{
///

///
/// Opaque tensor map handle. Construct via `viam_trajex_tensor_map_create`; release via
/// `viam_trajex_tensor_map_destroy`.
///
typedef struct viam_trajex_tensor_map viam_trajex_tensor_map_t;

///
/// Construct an empty tensor map.
///
/// The caller owns the returned handle and must release it via `viam_trajex_tensor_map_destroy`.
///
/// @return Newly-allocated tensor map handle, or NULL on allocation failure.
///
viam_trajex_tensor_map_t* viam_trajex_tensor_map_create(void);

///
/// Destroy a tensor map and release all owned tensor storage.
///
/// @param tensor_map Map to destroy. NULL is a no-op.
///
void viam_trajex_tensor_map_destroy(viam_trajex_tensor_map_t* tensor_map);

///
/// Insert a tensor into the map, copying the caller's buffer.
///
/// If `key` is already present, the existing entry is replaced transactionally: the new entry is fully allocated and
/// populated before the prior entry is freed, so a mid-call allocation failure leaves the prior entry intact.
///
/// @param tensor_map Map handle. Must not be NULL.
///
/// @param key Insertion key. Must be a NUL-terminated string, not NULL.
///
/// @param dtype Element type of the tensor.
///
/// @param rank Number of dimensions. Must be >= 1.
///
/// @param dims Array of `rank` size_t values describing the shape. All values must be >= 1. Must not be NULL.
///
/// @param data Pointer to the raw tensor data of `sizeof(dtype) * prod(dims)` bytes. Must not be NULL.
///
/// @return 0 on success. -1 on bad arguments or allocation failure (no
///         diagnostic).
///
int viam_trajex_tensor_map_insert(
    viam_trajex_tensor_map_t* tensor_map, const char* key, viam_trajex_dtype_t dtype, size_t rank, const size_t* dims, const void* data);

///
/// Insert a double-precision floating-point tensor.
///
/// Convenience wrapper around `viam_trajex_tensor_map_insert` with `dtype` fixed at `VIAM_TRAJEX_DTYPE_F64` and a typed
/// data pointer in place of `const void*`. Semantics (transactional replace, error reporting) are identical to the
/// generic form.
///
/// @param tensor_map Map handle. Must not be NULL.
///
/// @param key Insertion key. Must be a NUL-terminated string, not NULL.
///
/// @param rank Number of dimensions. Must be >= 1.
///
/// @param dims Array of `rank` size_t values describing the shape. All values must be >= 1. Must not be NULL.
///
/// @param data Pointer to `prod(dims)` double values. Must not be NULL.
///
/// @return 0 on success. -1 on bad arguments or allocation failure (no diagnostic).
///
int viam_trajex_tensor_map_insert_f64(
    viam_trajex_tensor_map_t* tensor_map, const char* key, size_t rank, const size_t* dims, const double* data);

///
/// Insert a signed 64-bit integer tensor.
///
/// Convenience wrapper around `viam_trajex_tensor_map_insert` with `dtype` fixed at `VIAM_TRAJEX_DTYPE_I64` and a typed
/// data pointer in place of `const void*`. Semantics (transactional replace, error reporting) are identical to the
/// generic form.
///
/// @param tensor_map Map handle. Must not be NULL.
///
/// @param key Insertion key. Must be a NUL-terminated string, not NULL.
///
/// @param rank Number of dimensions. Must be >= 1.
///
/// @param dims Array of `rank` size_t values describing the shape. All values must be >= 1. Must not be NULL.
///
/// @param data Pointer to `prod(dims)` int64_t values. Must not be NULL.
///
/// @return 0 on success. -1 on bad arguments or allocation failure (no
///         diagnostic).
///
int viam_trajex_tensor_map_insert_i64(
    viam_trajex_tensor_map_t* tensor_map, const char* key, size_t rank, const size_t* dims, const int64_t* data);

///
/// Insert a double-precision floating-point scalar (shape `[1]`).
///
/// Convenience wrapper around `viam_trajex_tensor_map_insert_f64` that stores `value` as a rank-1 shape-`[1]` tensor --
/// the schema convention for scalar inputs. Semantics (transactional replace, error reporting) are identical to the
/// underlying form.
///
/// @param tensor_map Map handle. Must not be NULL.
///
/// @param key Insertion key. Must be a NUL-terminated string, not NULL.
///
/// @param value Scalar value to store.
///
/// @return 0 on success. -1 on bad arguments or allocation failure (no diagnostic).
///
int viam_trajex_tensor_map_insert_scalar_f64(viam_trajex_tensor_map_t* tensor_map, const char* key, double value);

///
/// Insert a signed 64-bit integer scalar (shape `[1]`).
///
/// Convenience wrapper around `viam_trajex_tensor_map_insert_i64` that stores `value` as a rank-1 shape-`[1]` tensor --
/// the schema convention for scalar inputs. Semantics (transactional replace, error reporting) are identical to the
/// underlying form.
///
/// @param tensor_map Map handle. Must not be NULL.
///
/// @param key Insertion key. Must be a NUL-terminated string, not NULL.
///
/// @param value Scalar value to store.
///
/// @return 0 on success. -1 on bad arguments or allocation failure (no diagnostic).
///
int viam_trajex_tensor_map_insert_scalar_i64(viam_trajex_tensor_map_t* tensor_map, const char* key, int64_t value);

///
/// View a tensor by key, returning pointers into the map's owned storage.
///
/// No copy is performed: `*dims_out` points at the map's internal shape buffer and `*data_out` points at the map's
/// internal data buffer. Both pointers remain valid until the map is destroyed or the corresponding key is replaced
/// (via another insert). Callers must not write through the returned pointers; the storage is conceptually read-only
/// from outside the map.
///
/// To retain the data past the map's lifetime, copy it out: the caller knows the byte count from `*rank_out`,
/// `*dims_out`, and `*dtype_out`.
///
/// @param tensor_map Map handle. Must not be NULL.
///
/// @param key Tensor key. Must be a NUL-terminated string, not NULL.
///
/// @param dtype_out Receives the element type on success. Must not be NULL.
///
/// @param rank_out Receives the rank on success (always >= 1). Must not be NULL.
///
/// @param dims_out Receives a pointer to the shape array (size `*rank_out`). Must not be NULL.
///
/// @param data_out Receives a pointer to the raw data buffer. Must not be NULL.
///
/// @return 0 on success. 1 if no tensor with the given key is present.  -1 on internal error or bad arguments.
///
int viam_trajex_tensor_map_view(const viam_trajex_tensor_map_t* tensor_map,
                                const char* key,
                                viam_trajex_dtype_t* dtype_out,
                                size_t* rank_out,
                                const size_t** dims_out,
                                const void** data_out);

/// @}

///
/// @defgroup viam_trajex_string Strings
/// @{
///

///
/// Release a heap-allocated string returned by the trajex C ABI.
///
/// @param string String to release. NULL is a no-op.
///
void viam_trajex_string_destroy(const char* string);

/// @}

///
/// @defgroup viam_trajex_totg TOTG: time-optimal trajectory generation
///
/// Stateless one-shot trajectory generation following the Kunz & Stilman TOTG algorithm.
///
/// @{
///

///
/// @name Schema key constants
///
/// String constants for the input and output keys honored by `viam_trajex_totg_generate`. Callers can reference these
/// in place of hard-coded string literals; the underlying string values are 1:1 with the existing trajex
/// `service::mlmodel::infer` schema and are part of the stable ABI.
///
/// @{
///

extern const char viam_trajex_totg_key_acceleration_limits_rads_per_sec2[];
extern const char viam_trajex_totg_key_accelerations_rads_per_sec2[];
extern const char viam_trajex_totg_key_configurations_rads[];
extern const char viam_trajex_totg_key_path_colinearization_ratio[];
extern const char viam_trajex_totg_key_path_tolerance_delta_rads[];
extern const char viam_trajex_totg_key_sample_times_sec[];
extern const char viam_trajex_totg_key_trajectory_sampling_freq_hz[];
extern const char viam_trajex_totg_key_velocities_rads_per_sec[];
extern const char viam_trajex_totg_key_velocity_limits_rads_per_sec[];
extern const char viam_trajex_totg_key_waypoint_deduplication_tolerance_rads[];
extern const char viam_trajex_totg_key_waypoints_rads[];

/// @}

///
/// Generate a time-optimal trajectory from a waypoint sequence using the Kunz & Stilman TOTG algorithm.
///
/// On success, any prior contents of `outputs` are replaced by the generated output tensors. On failure, `outputs` is
/// left unchanged.
///
/// `inputs` and `outputs` must refer to distinct map handles. The `restrict` qualifier on each parameter (active in C
/// only; expanded to nothing in C++) makes this a type-level contract in C; C++ callers honor it by documentation
/// alone.
///
/// ## Required inputs
///
/// Entry point returns nonzero with diagnostic if absent.
///
/// | Key | Dtype | Shape | Meaning |
/// |-----|-------|-------|---------|
/// | `waypoints_rads` | F64 | `[n_waypoints, n_dof]` | Joint configurations to follow, in radians |
/// | `velocity_limits_rads_per_sec` | F64 | `[n_dof]` | Per-joint maximum velocity; defines DOF |
/// | `acceleration_limits_rads_per_sec2` | F64 | `[n_dof]` | Per-joint maximum acceleration |
/// | `path_tolerance_delta_rads` | F64 | `[1]` | Path-blending tolerance |
///
/// ## Optional inputs
///
/// Default applied if absent; supplied value used verbatim otherwise (no sentinel encoding).
///
/// | Key | Dtype | Shape | Default | Meaning |
/// |-----|-------|-------|---------|---------|
/// | `path_colinearization_ratio` | F64 | `[1]` | `0.0` (off) | Colinearization aggressiveness |
/// | `waypoint_deduplication_tolerance_rads` | F64 | `[1]` | `1e-5` | Waypoints closer than this are merged |
/// | `trajectory_sampling_freq_hz` | I64 | `[1]` | `100` | Output sample rate in Hz |
///
/// ## Outputs
///
/// Always present on success.
///
/// | Key | Dtype | Shape | Meaning |
/// |-----|-------|-------|---------|
/// | `sample_times_sec` | F64 | `[n_samples]` | Sample timestamps from `t=0`, seconds |
/// | `configurations_rads` | F64 | `[n_samples, n_dof]` | Joint positions at each sample |
/// | `velocities_rads_per_sec` | F64 | `[n_samples, n_dof]` | Joint velocities at each sample |
/// | `accelerations_rads_per_sec2` | F64 | `[n_samples, n_dof]` | Joint accelerations at each sample |
///
/// ## Cross-input rules
///
/// - DOF is derived from `velocity_limits_rads_per_sec.size()`.
/// - `acceleration_limits_rads_per_sec2.size()` must equal DOF.
/// - `waypoints_rads.shape[1]` must equal DOF.
///
/// @param inputs Input tensor map carrying the required and optional keys documented above. Must not be NULL.
///
/// @param outputs Caller-allocated output tensor map (via `viam_trajex_tensor_map_create`). On success its contents are
///                replaced by the generated output tensors; on failure its contents are unchanged. Must not be NULL and
///                must be distinct from `inputs`.
///
/// @param error_out On `-1` return, receives a newly-allocated NUL-terminated diagnostic string the caller releases via
///                  `viam_trajex_string_destroy`. On success, `*error_out` is set to NULL. May be NULL if the caller
///                  does not need a diagnostic.
///
/// @return 0 on success, -1 on failure.
///
int viam_trajex_totg_generate(const viam_trajex_tensor_map_t* VIAM_TRAJEX_RESTRICT inputs,
                              viam_trajex_tensor_map_t* VIAM_TRAJEX_RESTRICT outputs,
                              const char** error_out);

/// @}

#ifdef __cplusplus
}  // extern "C"
#endif

#pragma pop_macro("VIAM_TRAJEX_RESTRICT")
