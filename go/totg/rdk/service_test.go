//go:build !windows && !no_cgo

package rdk_test

import (
	"context"
	"math"
	"sort"
	"strings"
	"testing"
	"time"

	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
	"go.viam.com/test"
	"gorgonia.org/tensor"

	"github.com/viam-modules/trajex/go/totg"
	trajexrdk "github.com/viam-modules/trajex/go/totg/rdk"
)

func newTestService(t *testing.T) *trajexrdk.Service {
	t.Helper()
	name := resource.NewName(mlmodel.API, "trajex-totg-test")
	return trajexrdk.NewService(name)
}

func buildInputs() ml.Tensors {
	return ml.Tensors{
		totg.KeyWaypointsRads: tensor.New(
			tensor.WithShape(3, 2),
			tensor.WithBacking([]float64{0, 0, 1, 1, 2, 0}),
		),
		totg.KeyVelocityLimitsRadsPerSec: tensor.New(
			tensor.WithShape(2),
			tensor.WithBacking([]float64{1.0, 1.0}),
		),
		totg.KeyAccelerationLimitsRadsPerSec2: tensor.New(
			tensor.WithShape(2),
			tensor.WithBacking([]float64{1.0, 1.0}),
		),
		totg.KeyPathToleranceDeltaRads: tensor.New(
			tensor.WithShape(1),
			tensor.WithBacking([]float64{0.1}),
		),
		totg.KeyTrajectorySamplingFreqHz: tensor.New(
			tensor.WithShape(1),
			tensor.WithBacking([]int64{100}),
		),
	}
}

func TestMetadata(t *testing.T) {
	s := newTestService(t)
	md, err := s.Metadata(context.Background())
	test.That(t, err, test.ShouldBeNil)
	test.That(t, md.ModelName, test.ShouldEqual, "trajex-totg")
	test.That(t, md.ModelType, test.ShouldNotBeEmpty)
	test.That(t, md.ModelDescription, test.ShouldNotBeEmpty)
	test.That(t, len(md.Inputs), test.ShouldBeGreaterThan, 0)
	test.That(t, len(md.Outputs), test.ShouldEqual, 4)

	// Spot-check a few critical entries.
	inputNames := map[string]bool{}
	for _, tinfo := range md.Inputs {
		inputNames[tinfo.Name] = true
	}
	test.That(t, inputNames[totg.KeyWaypointsRads], test.ShouldBeTrue)
	test.That(t, inputNames[totg.KeyVelocityLimitsRadsPerSec], test.ShouldBeTrue)
	test.That(t, inputNames[totg.KeyAccelerationLimitsRadsPerSec2], test.ShouldBeTrue)

	outputNames := map[string]bool{}
	for _, tinfo := range md.Outputs {
		outputNames[tinfo.Name] = true
	}
	test.That(t, outputNames[totg.KeySampleTimesSec], test.ShouldBeTrue)
	test.That(t, outputNames[totg.KeyConfigurationsRads], test.ShouldBeTrue)
	test.That(t, outputNames[totg.KeyVelocitiesRadsPerSec], test.ShouldBeTrue)
	test.That(t, outputNames[totg.KeyAccelerationsRadsPerSec2], test.ShouldBeTrue)
}

func TestInferRoundtrip(t *testing.T) {
	s := newTestService(t)
	out, err := s.Infer(context.Background(), buildInputs())
	test.That(t, err, test.ShouldBeNil)

	times, ok := out[totg.KeySampleTimesSec]
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, times, test.ShouldNotBeNil)
	test.That(t, times.Dtype(), test.ShouldResemble, tensor.Float64)
	test.That(t, times.Shape(), test.ShouldHaveLength, 1)
	nSamples := times.Shape()[0]
	test.That(t, nSamples, test.ShouldBeGreaterThan, 1)

	for _, key := range []string{
		totg.KeyConfigurationsRads,
		totg.KeyVelocitiesRadsPerSec,
		totg.KeyAccelerationsRadsPerSec2,
	} {
		entry, ok := out[key]
		test.That(t, ok, test.ShouldBeTrue)
		test.That(t, entry.Shape(), test.ShouldResemble, tensor.Shape{nSamples, 2})
	}

	// All values finite.
	for _, entry := range out {
		data, ok := entry.Data().([]float64)
		test.That(t, ok, test.ShouldBeTrue)
		for _, v := range data {
			test.That(t, math.IsNaN(v), test.ShouldBeFalse)
			test.That(t, math.IsInf(v, 0), test.ShouldBeFalse)
		}
	}

	// First sample should be t=0 (CAPI emits the start state; sidecar dropped it).
	tData, ok := times.Data().([]float64)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, tData[0], test.ShouldEqual, 0.0)
}

func TestInferContextCancelled(t *testing.T) {
	s := newTestService(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := s.Infer(ctx, buildInputs())
	test.That(t, err, test.ShouldEqual, context.Canceled)
}

func TestInferUnsupportedDtype(t *testing.T) {
	s := newTestService(t)
	bad := ml.Tensors{
		totg.KeyWaypointsRads: tensor.New(
			tensor.WithShape(3, 2),
			tensor.WithBacking([]float32{0, 0, 1, 1, 2, 0}),
		),
	}
	_, err := s.Infer(context.Background(), bad)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "unsupported tensor dtype"), test.ShouldBeTrue)
}

// TestMinimalTrajectoryLatency measures end-to-end Service.Infer latency
// for the simplest possible trajectory: 2 DOF, 2 waypoints (0,0) -> (1,1),
// velocity limits far above what the trajectory will reach, low
// acceleration limits so the binding constraint is acceleration throughout.
// One forward pass + one backward pass, no limit-curve interior switching.
// Reports cold-call (first invocation, includes any one-time cgo setup),
// then a warm distribution over N calls. Run with `go test -v` to see
// output.
func TestMinimalTrajectoryLatency(t *testing.T) {
	const nDOF = 2
	inputs := ml.Tensors{
		totg.KeyWaypointsRads: tensor.New(
			tensor.WithShape(2, nDOF),
			tensor.WithBacking([]float64{0, 0, 1, 1}),
		),
		totg.KeyVelocityLimitsRadsPerSec: tensor.New(
			tensor.WithShape(nDOF),
			tensor.WithBacking([]float64{1000.0, 1000.0}),
		),
		totg.KeyAccelerationLimitsRadsPerSec2: tensor.New(
			tensor.WithShape(nDOF),
			tensor.WithBacking([]float64{0.1, 0.1}),
		),
		totg.KeyPathToleranceDeltaRads: tensor.New(
			tensor.WithShape(1),
			tensor.WithBacking([]float64{0.01}),
		),
		totg.KeyTrajectorySamplingFreqHz: tensor.New(
			tensor.WithShape(1),
			tensor.WithBacking([]int64{100}),
		),
	}

	s := newTestService(t)

	// Cold call. Pays for any one-time cgo setup and the first dylib
	// touch by this binary.
	coldStart := time.Now()
	out, err := s.Infer(context.Background(), inputs)
	coldElapsed := time.Since(coldStart)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, out, test.ShouldNotBeNil)

	// Sanity: time monotonic and starts at zero, endpoints match the
	// waypoints. Lightweight checks; correctness is tested elsewhere.
	tData := out[totg.KeySampleTimesSec].Data().([]float64)
	cData := out[totg.KeyConfigurationsRads].Data().([]float64)
	nSamples := len(tData)
	test.That(t, nSamples, test.ShouldBeGreaterThan, 1)
	test.That(t, tData[0], test.ShouldEqual, 0.0)
	test.That(t, cData[0], test.ShouldEqual, 0.0)
	test.That(t, cData[1], test.ShouldEqual, 0.0)
	test.That(t, cData[(nSamples-1)*nDOF+0], test.ShouldAlmostEqual, 1.0, 1e-6)
	test.That(t, cData[(nSamples-1)*nDOF+1], test.ShouldAlmostEqual, 1.0, 1e-6)

	// Warm distribution.
	const N = 1000
	durations := make([]time.Duration, N)
	for i := 0; i < N; i++ {
		start := time.Now()
		_, err := s.Infer(context.Background(), inputs)
		durations[i] = time.Since(start)
		test.That(t, err, test.ShouldBeNil)
	}
	sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })

	var total time.Duration
	for _, d := range durations {
		total += d
	}
	mean := total / time.Duration(N)

	t.Logf("Minimal trajectory latency (2 DOF, 2 waypoints, %d samples per call):", nSamples)
	t.Logf("  cold:   %v", coldElapsed)
	t.Logf("  warm (n=%d):", N)
	t.Logf("    min:    %v", durations[0])
	t.Logf("    p50:    %v", durations[N/2])
	t.Logf("    mean:   %v", mean)
	t.Logf("    p90:    %v", durations[N*90/100])
	t.Logf("    p99:    %v", durations[N*99/100])
	t.Logf("    p99.9:  %v", durations[N*999/1000])
	t.Logf("    max:    %v", durations[N-1])
}

// grayCodeHypercubeWaypoints returns a flat row-major (n, 6) waypoint
// buffer where each row is a vertex of the unit-edge 6-cube, visited in
// reflected-Gray-code order. The Gray code over 6 bits has 64 vertices
// and is cyclic (vertex 63 -> vertex 0 is also a single-bit flip), so
// indexing by (i mod 64) cycles cleanly. Each consecutive pair differs
// in exactly one coordinate, so every segment is an axis-aligned move
// of length `edge` along one of the 6 axes.
func grayCodeHypercubeWaypoints(n int, edge float64) []float64 {
	const nDOF = 6
	out := make([]float64, n*nDOF)
	for i := 0; i < n; i++ {
		v := uint(i) % 64
		g := v ^ (v >> 1)
		for j := 0; j < nDOF; j++ {
			if g&(1<<j) != 0 {
				out[i*nDOF+j] = edge
			}
		}
	}
	return out
}

// TestHypercubeTortureLatency walks a 6-DOF unit hypercube via a reflected
// Gray code 64 times (4097 waypoints, 4096 axis-aligned segments). Fat
// path tolerance squircularizes the 90 degree corners; tight acceleration
// makes those corners painful; tight velocity forces vel-curve cruise on
// the linear stretches between corners. Skipped under -short.
func TestHypercubeTortureLatency(t *testing.T) {
	if testing.Short() {
		t.Skip("hypercube torture test is slow; skipped under -short")
	}

	const (
		nWaypoints = 4097
		nDOF       = 6
		edge       = 1.0
	)
	waypoints := grayCodeHypercubeWaypoints(nWaypoints, edge)

	inputs := ml.Tensors{
		totg.KeyWaypointsRads: tensor.New(
			tensor.WithShape(nWaypoints, nDOF),
			tensor.WithBacking(waypoints),
		),
		totg.KeyVelocityLimitsRadsPerSec: tensor.New(
			tensor.WithShape(nDOF),
			tensor.WithBacking([]float64{0.3, 0.3, 0.3, 0.3, 0.3, 0.3}),
		),
		totg.KeyAccelerationLimitsRadsPerSec2: tensor.New(
			tensor.WithShape(nDOF),
			tensor.WithBacking([]float64{0.4, 0.4, 0.4, 0.4, 0.4, 0.4}),
		),
		totg.KeyPathToleranceDeltaRads: tensor.New(
			tensor.WithShape(1),
			tensor.WithBacking([]float64{0.1}),
		),
		totg.KeyTrajectorySamplingFreqHz: tensor.New(
			tensor.WithShape(1),
			tensor.WithBacking([]int64{10}),
		),
	}

	s := newTestService(t)

	coldStart := time.Now()
	out, err := s.Infer(context.Background(), inputs)
	coldElapsed := time.Since(coldStart)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, out, test.ShouldNotBeNil)

	tData := out[totg.KeySampleTimesSec].Data().([]float64)
	nSamples := len(tData)
	totalSeconds := tData[nSamples-1]

	const N = 5
	durations := make([]time.Duration, N)
	for i := 0; i < N; i++ {
		start := time.Now()
		_, err := s.Infer(context.Background(), inputs)
		durations[i] = time.Since(start)
		test.That(t, err, test.ShouldBeNil)
	}
	sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })

	var total time.Duration
	for _, d := range durations {
		total += d
	}
	mean := total / time.Duration(N)

	t.Logf("Hypercube torture (6 DOF, %d waypoints, %d samples per call, %.1f s trajectory):",
		nWaypoints, nSamples, totalSeconds)
	t.Logf("  cold:   %v", coldElapsed)
	t.Logf("  warm (n=%d):", N)
	t.Logf("    min:    %v", durations[0])
	t.Logf("    median: %v", durations[N/2])
	t.Logf("    mean:   %v", mean)
	t.Logf("    max:    %v", durations[N-1])
}

// TestPhase4Gate exercises a 6-DOF, 5-waypoint trajectory through the full
// stack (ml.Tensors -> trajex.TensorMap -> totg.Generate -> trajex.TensorMap
// -> ml.Tensors) and asserts well-shapedness, finiteness, monotonic time,
// endpoint match, and per-joint velocity limit compliance. Passing this
// test closes Phase 4 of RSDK-12835.
func TestPhase4Gate(t *testing.T) {
	const nDOF = 6
	velLimits := []float64{1.0, 1.0, 1.0, 1.5, 1.5, 2.0}
	accLimits := []float64{2.0, 2.0, 2.0, 3.0, 3.0, 4.0}
	waypoints := []float64{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.5, -0.5, 1.0, 0.2, -0.2, 0.3,
		1.0, 0.5, 1.5, -0.4, 0.4, -0.5,
		0.5, 1.0, 0.5, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}
	inputs := ml.Tensors{
		totg.KeyWaypointsRads:                 tensor.New(tensor.WithShape(5, nDOF), tensor.WithBacking(waypoints)),
		totg.KeyVelocityLimitsRadsPerSec:      tensor.New(tensor.WithShape(nDOF), tensor.WithBacking(velLimits)),
		totg.KeyAccelerationLimitsRadsPerSec2: tensor.New(tensor.WithShape(nDOF), tensor.WithBacking(accLimits)),
		totg.KeyPathToleranceDeltaRads:        tensor.New(tensor.WithShape(1), tensor.WithBacking([]float64{0.05})),
		totg.KeyTrajectorySamplingFreqHz:      tensor.New(tensor.WithShape(1), tensor.WithBacking([]int64{200})),
	}

	s := newTestService(t)
	out, err := s.Infer(context.Background(), inputs)
	test.That(t, err, test.ShouldBeNil)

	times, ok := out[totg.KeySampleTimesSec]
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, times.Shape(), test.ShouldHaveLength, 1)
	nSamples := times.Shape()[0]
	test.That(t, nSamples, test.ShouldBeGreaterThan, 10)

	configs := out[totg.KeyConfigurationsRads]
	velocities := out[totg.KeyVelocitiesRadsPerSec]
	accelerations := out[totg.KeyAccelerationsRadsPerSec2]
	test.That(t, configs.Shape(), test.ShouldResemble, tensor.Shape{nSamples, nDOF})
	test.That(t, velocities.Shape(), test.ShouldResemble, tensor.Shape{nSamples, nDOF})
	test.That(t, accelerations.Shape(), test.ShouldResemble, tensor.Shape{nSamples, nDOF})

	tData := times.Data().([]float64)
	cData := configs.Data().([]float64)
	vData := velocities.Data().([]float64)
	aData := accelerations.Data().([]float64)

	// Time starts at zero and is monotonically nondecreasing.
	test.That(t, tData[0], test.ShouldEqual, 0.0)
	for i := 1; i < nSamples; i++ {
		test.That(t, tData[i], test.ShouldBeGreaterThanOrEqualTo, tData[i-1])
	}

	// All values finite.
	for _, arr := range [][]float64{tData, cData, vData, aData} {
		for _, v := range arr {
			test.That(t, math.IsNaN(v), test.ShouldBeFalse)
			test.That(t, math.IsInf(v, 0), test.ShouldBeFalse)
		}
	}

	// Endpoints match the requested first and last waypoints.
	for j := 0; j < nDOF; j++ {
		test.That(t, cData[j], test.ShouldAlmostEqual, waypoints[j], 1e-6)
		test.That(t, cData[(nSamples-1)*nDOF+j], test.ShouldAlmostEqual, waypoints[4*nDOF+j], 1e-6)
	}

	// Per-joint velocity limits respected (small tolerance for FP slack).
	const velTol = 1e-6
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nDOF; j++ {
			test.That(t, math.Abs(vData[i*nDOF+j]), test.ShouldBeLessThanOrEqualTo, velLimits[j]+velTol)
		}
	}
}
