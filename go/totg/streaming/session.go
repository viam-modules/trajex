//go:build !windows

// Package streaming exposes the trajex TOTG streaming session as a Go API,
// layered on top of github.com/viam-modules/trajex's TensorMap. Sessions are
// stateful: callers construct a session with a fixed configuration, extend it
// with waypoint batches over time, and pull samples incrementally.
//
// The schema-key constants (KeyWaypointsRads, KeyVelocityLimitsRadsPerSec,
// etc.) are re-exported from the parent totg package so callers using both
// stateless Generate and streaming sessions need to import only one set of
// key names.
package streaming

/*
#cgo CFLAGS: -I${SRCDIR}/../../artifacts/include

#include <viam/trajex/capi/capi.h>
*/
import "C"

import (
	"context"
	"time"

	"github.com/pkg/errors"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/totg"
)

// Re-export the shared schema-key constants from totg so streaming callers
// have a single source of truth. The values are identical to totg.Key* by
// construction (both initialized from the same C externs at package init).
var (
	KeyVelocityLimitsRadsPerSec      = totg.KeyVelocityLimitsRadsPerSec      //nolint:revive
	KeyAccelerationLimitsRadsPerSec2 = totg.KeyAccelerationLimitsRadsPerSec2 //nolint:revive
	KeyPathToleranceDeltaRads        = totg.KeyPathToleranceDeltaRads        //nolint:revive
	KeyPathColinearizationRatio      = totg.KeyPathColinearizationRatio      //nolint:revive
	KeyTrajectorySamplingFreqHz      = totg.KeyTrajectorySamplingFreqHz      //nolint:revive
	KeyWaypointsRads                 = totg.KeyWaypointsRads                 //nolint:revive
	KeySampleTimesSec                = totg.KeySampleTimesSec                //nolint:revive
	KeyConfigurationsRads            = totg.KeyConfigurationsRads            //nolint:revive
	KeyVelocitiesRadsPerSec          = totg.KeyVelocitiesRadsPerSec          //nolint:revive
	KeyAccelerationsRadsPerSec2      = totg.KeyAccelerationsRadsPerSec2      //nolint:revive
)

// Session is a Go-owned handle to a CAPI streaming session. Close must be
// called (typically via defer) to release the underlying C resource. Session
// is not safe for concurrent use; concurrent operations on the same handle
// are the caller's responsibility, matching the C ABI's contract.
type Session struct {
	handle *C.viam_trajex_totg_streaming_session_t
}

// New constructs a streaming session from a configuration tensor map. The
// options map carries the velocity / acceleration limits, path tolerance,
// sample rate, and optional path colinearization ratio; see the C ABI header
// for the full schema. The options map is read-only during construction and
// may be closed by the caller as soon as New returns.
func New(options *trajex.TensorMap) (*Session, error) {
	optHandle := (*C.viam_trajex_tensor_map_t)(options.UnsafeHandle())
	var errOut *C.char
	h := C.viam_trajex_totg_streaming_session_create(optHandle, &errOut)
	if h == nil {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return nil, errors.Errorf("trajex/totg/streaming: New failed: %s", msg)
	}
	return &Session{handle: h}, nil
}

// Close releases the underlying C session. Safe to call on a nil receiver
// and idempotent: subsequent calls are no-ops.
func (s *Session) Close() {
	if s == nil || s.handle == nil {
		return
	}
	C.viam_trajex_totg_streaming_session_destroy(s.handle)
	s.handle = nil
}

// Extend appends a waypoint batch to the session. The batch must contain a
// waypoints_rads tensor of shape [n_waypoints, n_dof]; on calls after the
// first, batch[0] must compare bit-exactly equal to the session's most
// recently stored waypoint (the seam contract).
//
// Extend honors ctx at entry only: if ctx is already cancelled when Extend is
// called, it returns ctx.Err() without invoking the C ABI. Once the C call
// begins it cannot be interrupted, so a cancellation landing mid-call does not
// abort it; the operation runs to completion and its result (including any
// committed state mutation) is always reported.
func (s *Session) Extend(ctx context.Context, batch *trajex.TensorMap) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	batchHandle := (*C.viam_trajex_tensor_map_t)(batch.UnsafeHandle())
	var errOut *C.char
	rc := C.viam_trajex_totg_streaming_session_extend(s.handle, batchHandle, &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return errors.Errorf("trajex/totg/streaming: Extend failed: %s", msg)
	}
	return nil
}

// SampleNext pulls up to n samples into outputs. The output map's prior
// contents are replaced. If the session is exhausted, outputs carries
// zero-length sample tensors.
//
// Honors ctx like Extend.
func (s *Session) SampleNext(ctx context.Context, n int, outputs *trajex.TensorMap) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	outHandle := (*C.viam_trajex_tensor_map_t)(outputs.UnsafeHandle())
	var errOut *C.char
	rc := C.viam_trajex_totg_streaming_session_sample_next(s.handle, C.size_t(n), outHandle, &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return errors.Errorf("trajex/totg/streaming: SampleNext failed: %s", msg)
	}
	return nil
}

// SampleAtLeast pulls samples until the most recent sample's time is at least
// CurrentTime + horizon, writing them into outputs.
//
// Honors ctx like Extend.
func (s *Session) SampleAtLeast(ctx context.Context, horizon time.Duration, outputs *trajex.TensorMap) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	outHandle := (*C.viam_trajex_tensor_map_t)(outputs.UnsafeHandle())
	var errOut *C.char
	rc := C.viam_trajex_totg_streaming_session_sample_at_least(s.handle, C.double(horizon.Seconds()), outHandle, &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return errors.Errorf("trajex/totg/streaming: SampleAtLeast failed: %s", msg)
	}
	return nil
}

// CurrentTime returns the global time of the most recently emitted sample,
// or zero if no samples have been emitted yet.
func (s *Session) CurrentTime() time.Duration {
	var out C.double
	C.viam_trajex_totg_streaming_session_current_time_sec(s.handle, &out)
	return time.Duration(float64(out) * float64(time.Second))
}

// GenerationCount returns the cumulative number of trajectories the session
// has installed as active (first build + each pivot + each rebase). Zero for
// a fresh session before the first Extend.
func (s *Session) GenerationCount() int64 {
	var out C.int64_t
	C.viam_trajex_totg_streaming_session_generation_count(s.handle, &out)
	return int64(out)
}

// HasActiveTrajectory reports whether the session has an active trajectory.
// False iff fresh (no successful Extend has occurred yet).
func (s *Session) HasActiveTrajectory() bool {
	var out C.int
	C.viam_trajex_totg_streaming_session_has_active_trajectory(s.handle, &out)
	return out != 0
}

// ActiveDuration returns the duration of the active trajectory. Returns zero
// when no active trajectory is present.
func (s *Session) ActiveDuration() time.Duration {
	var out C.double
	C.viam_trajex_totg_streaming_session_active_duration_sec(s.handle, &out)
	return time.Duration(float64(out) * float64(time.Second))
}
