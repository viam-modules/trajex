//go:build !windows

package streaming_test

import (
	"context"
	"testing"
	"time"

	"go.viam.com/test"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/totg/streaming"
)

// buildOptions constructs a session configuration tensor map: 2 DOF,
// conservative limits, 100 Hz sample rate. Caller must Close it.
func buildOptions(t *testing.T) *trajex.TensorMap {
	t.Helper()
	opts, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)

	test.That(t, opts.InsertFloat64s(streaming.KeyVelocityLimitsRadsPerSec, []uint64{2}, []float64{1.0, 1.0}), test.ShouldBeNil)
	test.That(t, opts.InsertFloat64s(streaming.KeyAccelerationLimitsRadsPerSec2, []uint64{2}, []float64{1.0, 1.0}), test.ShouldBeNil)
	test.That(t, opts.InsertScalarFloat64(streaming.KeyPathToleranceDeltaRads, 0.1), test.ShouldBeNil)
	test.That(t, opts.InsertScalarFloat64(streaming.KeyTrajectorySamplingFreqHz, 100.0), test.ShouldBeNil)

	return opts
}

// buildBatch constructs a waypoint batch tensor map carrying the supplied
// flat waypoint coordinates as [n, 2] (2 DOF). Caller must Close it.
func buildBatch(t *testing.T, waypoints []float64) *trajex.TensorMap {
	t.Helper()
	const dof = 2
	if len(waypoints)%dof != 0 {
		t.Fatalf("buildBatch: waypoints length %d not divisible by dof %d", len(waypoints), dof)
	}
	n := uint64(len(waypoints) / dof)

	batch, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	test.That(t, batch.InsertFloat64s(streaming.KeyWaypointsRads, []uint64{n, dof}, waypoints), test.ShouldBeNil)
	return batch
}

func TestSessionLifecycle(t *testing.T) {
	opts := buildOptions(t)
	defer opts.Close()

	sess, err := streaming.New(opts)
	test.That(t, err, test.ShouldBeNil)
	defer sess.Close()

	// Fresh session: no active trajectory, nothing sampled yet.
	test.That(t, sess.HasActiveTrajectory(), test.ShouldBeFalse)
	test.That(t, sess.GenerationCount(), test.ShouldEqual, int64(0))
	test.That(t, sess.CurrentTime(), test.ShouldEqual, time.Duration(0))
	test.That(t, sess.ActiveDuration(), test.ShouldEqual, time.Duration(0))
}

func TestSessionExtendAndSample(t *testing.T) {
	opts := buildOptions(t)
	defer opts.Close()
	sess, err := streaming.New(opts)
	test.That(t, err, test.ShouldBeNil)
	defer sess.Close()

	// Bootstrap with three 2-DOF waypoints.
	batch := buildBatch(t, []float64{
		0.0, 0.0,
		1.0, 0.0,
		1.0, 1.0,
	})
	defer batch.Close()

	test.That(t, sess.Extend(context.Background(), batch), test.ShouldBeNil)

	// Active trajectory should now exist.
	test.That(t, sess.HasActiveTrajectory(), test.ShouldBeTrue)
	test.That(t, sess.GenerationCount(), test.ShouldEqual, int64(1))
	test.That(t, sess.ActiveDuration(), test.ShouldBeGreaterThan, time.Duration(0))

	// Pull 10 samples. The output map's contents are populated with the
	// sample tensors.
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()

	test.That(t, sess.SampleNext(context.Background(), 10, out), test.ShouldBeNil)

	timesShape, timesData, ok, err := out.ViewFloat64s(streaming.KeySampleTimesSec)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, len(timesShape), test.ShouldEqual, 1)
	test.That(t, timesShape[0], test.ShouldEqual, uint64(10))
	test.That(t, len(timesData), test.ShouldEqual, 10)

	cfgShape, cfgData, ok, err := out.ViewFloat64s(streaming.KeyConfigurationsRads)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, cfgShape, test.ShouldResemble, []uint64{10, 2})
	test.That(t, len(cfgData), test.ShouldEqual, 20)

	// Timestamps strictly monotonic and CurrentTime tracks the last sample.
	for i := 1; i < len(timesData); i++ {
		test.That(t, timesData[i], test.ShouldBeGreaterThan, timesData[i-1])
	}
	expectedCurrent := time.Duration(timesData[len(timesData)-1] * float64(time.Second))
	// Allow ~1 ns slack for the float-seconds round-trip.
	delta := sess.CurrentTime() - expectedCurrent
	if delta < 0 {
		delta = -delta
	}
	test.That(t, delta < time.Microsecond, test.ShouldBeTrue)
}

func TestSessionSampleAtLeast(t *testing.T) {
	opts := buildOptions(t)
	defer opts.Close()
	sess, err := streaming.New(opts)
	test.That(t, err, test.ShouldBeNil)
	defer sess.Close()

	batch := buildBatch(t, []float64{
		0.0, 0.0,
		1.0, 0.0,
		1.0, 1.0,
	})
	defer batch.Close()
	test.That(t, sess.Extend(context.Background(), batch), test.ShouldBeNil)

	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()

	horizon := 100 * time.Millisecond
	test.That(t, sess.SampleAtLeast(context.Background(), horizon, out), test.ShouldBeNil)

	test.That(t, sess.CurrentTime(), test.ShouldBeGreaterThanOrEqualTo, horizon)
}

func TestSessionExtendCtxCancel(t *testing.T) {
	opts := buildOptions(t)
	defer opts.Close()
	sess, err := streaming.New(opts)
	test.That(t, err, test.ShouldBeNil)
	defer sess.Close()

	batch := buildBatch(t, []float64{
		0.0, 0.0,
		1.0, 1.0,
	})
	defer batch.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = sess.Extend(ctx, batch)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, "context")
}

func TestSessionCloseIdempotent(t *testing.T) {
	opts := buildOptions(t)
	defer opts.Close()
	sess, err := streaming.New(opts)
	test.That(t, err, test.ShouldBeNil)

	sess.Close()
	sess.Close()
}
