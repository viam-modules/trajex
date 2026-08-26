//go:build !windows && cgo

package totg_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"go.viam.com/test"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/totg"
)

func TestSchemaKeysMatchABI(t *testing.T) {
	for _, tc := range []struct {
		name     string
		actual   string
		expected string
	}{
		{"waypoints", totg.KeyWaypointsRads, "waypoints_rads"},
		{"vel_limits", totg.KeyVelocityLimitsRadsPerSec, "velocity_limits_rads_per_sec"},
		{"acc_limits", totg.KeyAccelerationLimitsRadsPerSec2, "acceleration_limits_rads_per_sec2"},
		{"path_tol", totg.KeyPathToleranceDeltaRads, "path_tolerance_delta_rads"},
		{"colin", totg.KeyPathColinearizationRatio, "path_colinearization_ratio"},
		{"dedup", totg.KeyWaypointDeduplicationToleranceRads, "waypoint_deduplication_tolerance_rads"},
		{"sampling_freq", totg.KeyTrajectorySamplingFreqHz, "trajectory_sampling_freq_hz"},
		{"sample_times", totg.KeySampleTimesSec, "sample_times_sec"},
		{"configs", totg.KeyConfigurationsRads, "configurations_rads"},
		{"vels", totg.KeyVelocitiesRadsPerSec, "velocities_rads_per_sec"},
		{"accs", totg.KeyAccelerationsRadsPerSec2, "accelerations_rads_per_sec2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			test.That(t, tc.actual, test.ShouldEqual, tc.expected)
		})
	}
}

// buildMinimalInputs constructs a small but well-formed input map: 3
// waypoints in 2 DOF, with conservative limits. Caller must Close it.
func buildMinimalInputs(t *testing.T) *trajex.TensorMap {
	t.Helper()
	in, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)

	waypoints := []float64{
		0.0, 0.0,
		1.0, 1.0,
		2.0, 0.0,
	}
	test.That(t, in.InsertFloat64s(totg.KeyWaypointsRads, []uint64{3, 2}, waypoints), test.ShouldBeNil)
	test.That(t, in.InsertFloat64s(totg.KeyVelocityLimitsRadsPerSec, []uint64{2}, []float64{1.0, 1.0}), test.ShouldBeNil)
	test.That(t, in.InsertFloat64s(totg.KeyAccelerationLimitsRadsPerSec2, []uint64{2}, []float64{1.0, 1.0}), test.ShouldBeNil)
	test.That(t, in.InsertScalarFloat64(totg.KeyPathToleranceDeltaRads, 0.1), test.ShouldBeNil)
	test.That(t, in.InsertScalarFloat64(totg.KeyTrajectorySamplingFreqHz, 100.0), test.ShouldBeNil)

	return in
}

func TestGenerateMinimal(t *testing.T) {
	in := buildMinimalInputs(t)
	defer in.Close()
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()

	err = totg.Generate(context.Background(), in, out)
	test.That(t, err, test.ShouldBeNil)

	// All four outputs must be present and well-shaped.
	times, timesData, ok, err := out.ViewFloat64s(totg.KeySampleTimesSec)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, len(times), test.ShouldEqual, 1)
	test.That(t, times[0], test.ShouldBeGreaterThan, uint64(0))
	nSamples := times[0]

	cfgShape, cfgData, ok, err := out.ViewFloat64s(totg.KeyConfigurationsRads)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, cfgShape, test.ShouldResemble, []uint64{nSamples, 2})

	velShape, velData, ok, err := out.ViewFloat64s(totg.KeyVelocitiesRadsPerSec)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, velShape, test.ShouldResemble, []uint64{nSamples, 2})

	accShape, accData, ok, err := out.ViewFloat64s(totg.KeyAccelerationsRadsPerSec2)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, accShape, test.ShouldResemble, []uint64{nSamples, 2})

	// First sample is t=0 (we emit it, unlike the sidecar service which drops it).
	test.That(t, timesData[0], test.ShouldEqual, 0.0)

	// Time samples are monotonically nondecreasing and finite.
	for i, t0 := range timesData {
		test.That(t, math.IsNaN(t0), test.ShouldBeFalse)
		test.That(t, math.IsInf(t0, 0), test.ShouldBeFalse)
		if i > 0 {
			test.That(t, t0, test.ShouldBeGreaterThanOrEqualTo, timesData[i-1])
		}
	}

	// Joint streams are finite throughout.
	for _, v := range cfgData {
		test.That(t, math.IsNaN(v), test.ShouldBeFalse)
		test.That(t, math.IsInf(v, 0), test.ShouldBeFalse)
	}
	for _, v := range velData {
		test.That(t, math.IsNaN(v), test.ShouldBeFalse)
		test.That(t, math.IsInf(v, 0), test.ShouldBeFalse)
	}
	for _, v := range accData {
		test.That(t, math.IsNaN(v), test.ShouldBeFalse)
		test.That(t, math.IsInf(v, 0), test.ShouldBeFalse)
	}

	// First configuration matches the first waypoint (0,0); last matches
	// the final waypoint (2,0).
	test.That(t, cfgData[0], test.ShouldEqual, 0.0)
	test.That(t, cfgData[1], test.ShouldEqual, 0.0)
	lastIdx := int(nSamples-1) * 2
	test.That(t, cfgData[lastIdx], test.ShouldAlmostEqual, 2.0, 1e-6)
	test.That(t, cfgData[lastIdx+1], test.ShouldAlmostEqual, 0.0, 1e-6)
}

func TestGenerateMissingRequiredKey(t *testing.T) {
	in, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer in.Close()
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()

	// Only waypoints; everything else missing. C side must surface a
	// diagnostic mentioning the first missing key.
	test.That(t, in.InsertFloat64s(totg.KeyWaypointsRads, []uint64{3, 2},
		[]float64{0, 0, 1, 1, 2, 0}), test.ShouldBeNil)

	err = totg.Generate(context.Background(), in, out)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "missing required input"), test.ShouldBeTrue)
}

func TestGenerateContextAlreadyCancelled(t *testing.T) {
	in := buildMinimalInputs(t)
	defer in.Close()
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = totg.Generate(ctx, in, out)
	test.That(t, err, test.ShouldEqual, context.Canceled)
}
