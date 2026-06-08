// Compile-smoke for the trajex C ABI header.
//
// This file is compiled by the C compiler (strict ISO C99, no GNU
// extensions) and linked into an executable. The combination validates
// two contracts:
//
//   1. capi.h parses cleanly as ISO C99 -- no C++ tokens, no
//      GCC-specific syntax, no missing standard-library includes.
//
//   2. Every entry point and key constant declared by the header is
//      actually exported by the linked viam-trajex-capi library. The
//      address-take of each symbol forces the linker to resolve it.
//
// Build failure of this target is the early-warning tripwire for a
// regression in either of those contracts.

#include <viam/trajex/capi/capi.h>

// If this fires, the smoke is meaningless: somewhere upstream the build
// arranged for this file to be compiled by a C++ compiler instead of a C
// compiler, and we've lost the "header parses as C" contract this target
// exists to enforce.
#ifdef __cplusplus
#error "capi_compile_smoke.c must be compiled as C, not C++"
#endif

int main(void) {
    // Force-resolve every exported entry point.
    (void)viam_trajex_tensor_map_create;
    (void)viam_trajex_tensor_map_destroy;
    (void)viam_trajex_tensor_map_insert;
    (void)viam_trajex_tensor_map_insert_f64;
    (void)viam_trajex_tensor_map_insert_i64;
    (void)viam_trajex_tensor_map_insert_scalar_f64;
    (void)viam_trajex_tensor_map_insert_scalar_i64;
    (void)viam_trajex_tensor_map_view;
    (void)viam_trajex_string_destroy;
    (void)viam_trajex_totg_generate;

    // Force-resolve every exported key constant.
    (void)viam_trajex_totg_key_waypoints_rads;
    (void)viam_trajex_totg_key_velocity_limits_rads_per_sec;
    (void)viam_trajex_totg_key_acceleration_limits_rads_per_sec2;
    (void)viam_trajex_totg_key_path_tolerance_delta_rads;
    (void)viam_trajex_totg_key_path_colinearization_ratio;
    (void)viam_trajex_totg_key_waypoint_deduplication_tolerance_rads;
    (void)viam_trajex_totg_key_trajectory_sampling_freq_hz;
    (void)viam_trajex_totg_key_sample_times_sec;
    (void)viam_trajex_totg_key_configurations_rads;
    (void)viam_trajex_totg_key_velocities_rads_per_sec;
    (void)viam_trajex_totg_key_accelerations_rads_per_sec2;

    return 0;
}
