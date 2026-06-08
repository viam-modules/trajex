//go:build !windows && !no_cgo

// Package totg exposes the trajex TOTG (time-optimal trajectory generation)
// algorithm as a Go API, layered on top of github.com/viam-modules/trajex's
// TensorMap.
package totg

/*
#cgo CFLAGS: -I${SRCDIR}/../../src/viam/trajex/capi

#include "capi.h"
*/
import "C"

import (
	"context"
	"unsafe"

	"github.com/pkg/errors"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/internal/capi"
)

// Schema key string constants for inputs and outputs honored by Generate.
// Initialized at package init from the cgo-exposed viam_trajex_totg_key_*
// externs in capi.h so the strings cannot drift between Go and C without a
// recompile. Use these in place of hard-coded string literals.
var (
	KeyWaypointsRads                      string //nolint:revive
	KeyVelocityLimitsRadsPerSec           string //nolint:revive
	KeyAccelerationLimitsRadsPerSec2      string //nolint:revive
	KeyPathToleranceDeltaRads             string //nolint:revive
	KeyPathColinearizationRatio           string //nolint:revive
	KeyWaypointDeduplicationToleranceRads string //nolint:revive
	KeyTrajectorySamplingFreqHz           string //nolint:revive
	KeySampleTimesSec                     string //nolint:revive
	KeyConfigurationsRads                 string //nolint:revive
	KeyVelocitiesRadsPerSec               string //nolint:revive
	KeyAccelerationsRadsPerSec2           string //nolint:revive
)

func init() {
	KeyWaypointsRads = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_waypoints_rads))
	KeyVelocityLimitsRadsPerSec = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_velocity_limits_rads_per_sec))
	KeyAccelerationLimitsRadsPerSec2 = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_acceleration_limits_rads_per_sec2))
	KeyPathToleranceDeltaRads = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_path_tolerance_delta_rads))
	KeyPathColinearizationRatio = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_path_colinearization_ratio))
	KeyWaypointDeduplicationToleranceRads = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_waypoint_deduplication_tolerance_rads))
	KeyTrajectorySamplingFreqHz = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_trajectory_sampling_freq_hz))
	KeySampleTimesSec = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_sample_times_sec))
	KeyConfigurationsRads = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_configurations_rads))
	KeyVelocitiesRadsPerSec = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_velocities_rads_per_sec))
	KeyAccelerationsRadsPerSec2 = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_accelerations_rads_per_sec2))
}

// Generate computes a time-parameterized trajectory from the named-tensor
// inputs map, writing results into outputs. See the trajex C ABI header
// for the input/output schema; the key string constants in this package
// (KeyWaypointsRads, etc.) name the keys.
//
// inputs and outputs must be distinct, non-closed TensorMap handles.
//
// Generate honors ctx at entry and exit: if ctx is already cancelled when
// Generate is called, it returns ctx.Err() without invoking the C ABI; if
// ctx is cancelled while the C call is in flight (which itself cannot be
// interrupted), the result is discarded and ctx.Err() is returned.
func Generate(ctx context.Context, inputs, outputs *trajex.TensorMap) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	inHandle := (*C.viam_trajex_tensor_map_t)(inputs.UnsafeHandle())
	outHandle := (*C.viam_trajex_tensor_map_t)(outputs.UnsafeHandle())

	var errOut *C.char
	rc := C.viam_trajex_totg_generate(inHandle, outHandle, &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return errors.Errorf("trajex/totg: generate failed: %s", msg)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}
