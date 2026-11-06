//go:build !windows && cgo

package rdk_test

import (
	"context"
	"math"
	"testing"

	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/rdk/utils"
	"go.viam.com/test"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/kinematics"
	"github.com/viam-modules/trajex/go/rdk"
	"github.com/viam-modules/trajex/go/totg"
)

func f64(v float64) *float64 { return &v }

// limitedArm is the planar arm with the shoulder fully bounded and the elbow
// bounded in position only, and with both properties set.
func limitedArm() *commonpb.KinematicModel {
	m := planarArm()
	m.Joints[0].Limits = &commonpb.JointLimits{Min: f64(-90), Max: f64(90), MaxVelocity: f64(60), MaxAcceleration: f64(120)}
	m.Joints[1].Limits = &commonpb.JointLimits{Min: f64(-150), Max: f64(150)}
	m.Properties = &commonpb.KinematicProperties{OperatingFreqHz: f64(100), MaxLinearTcpSpeedMmPerSec: f64(250)}
	return m
}

// fullyBoundedArm bounds both joints the same way and caps the TCP at 100
// mm/s, which the joints alone would let reach about 146 mm/s.
func fullyBoundedArm() *commonpb.KinematicModel {
	m := limitedArm()
	m.Joints[1].Limits = &commonpb.JointLimits{Min: f64(-150), Max: f64(150), MaxVelocity: f64(60), MaxAcceleration: f64(120)}
	m.Properties.MaxLinearTcpSpeedMmPerSec = f64(100)
	return m
}

// The limits come back in the chain's units and position order, which is
// RDK's rather than the message's, with every bound the message lacks
// absent rather than zero.
func TestArmCarriesLimitsAndProperties(t *testing.T) {
	for _, swapped := range []bool{false, true} {
		m := limitedArm()
		if swapped {
			m.Joints[0], m.Joints[1] = m.Joints[1], m.Joints[0]
		}
		arm, err := rdk.ArmFromKinematicModel(m, "")
		test.That(t, err, test.ShouldBeNil)
		test.That(t, len(arm.Joints), test.ShouldEqual, 3)
		test.That(t, len(arm.Limits), test.ShouldEqual, 2)

		shoulder, elbow := arm.Limits[0], arm.Limits[1]
		test.That(t, shoulder.Min, test.ShouldAlmostEqual, -math.Pi/2)
		test.That(t, shoulder.Max, test.ShouldAlmostEqual, math.Pi/2)
		test.That(t, *shoulder.MaxVelocity, test.ShouldAlmostEqual, utils.DegToRad(60))
		test.That(t, *shoulder.MaxAcceleration, test.ShouldAlmostEqual, utils.DegToRad(120))
		test.That(t, elbow.Max, test.ShouldAlmostEqual, utils.DegToRad(150))
		test.That(t, elbow.MaxVelocity, test.ShouldBeNil)
		test.That(t, elbow.MaxAcceleration, test.ShouldBeNil)
		test.That(t, arm.OperatingFreqHz, test.ShouldEqual, 100)
		test.That(t, arm.MaxTCPSpeedMMPerSec, test.ShouldEqual, 250)
	}

	bare, err := rdk.ArmFromKinematicModel(planarArm(), "")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, math.IsInf(bare.Limits[0].Min, -1), test.ShouldBeTrue)
	test.That(t, bare.Limits[0].MaxVelocity, test.ShouldBeNil)
	test.That(t, bare.OperatingFreqHz, test.ShouldEqual, 0)
	test.That(t, bare.MaxTCPSpeedMMPerSec, test.ShouldEqual, 0)
}

func TestArmKeepsPrismaticLimitsInMillimetres(t *testing.T) {
	m := &commonpb.KinematicModel{
		Links:  []*commonpb.KinematicLink{link("base", "world", 0, 0, 0, nil), link("tip", "slide", 10, 0, 0, nil)},
		Joints: []*commonpb.KinematicJoint{joint("slide", "base", prismatic, 1, 0, 0)},
	}
	m.Joints[0].Limits = &commonpb.JointLimits{Min: f64(0), Max: f64(100), MaxVelocity: f64(50)}
	arm, err := rdk.ArmFromKinematicModel(m, "")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, arm.Limits[0].Max, test.ShouldEqual, 100)
	test.That(t, *arm.Limits[0].MaxVelocity, test.ShouldEqual, 50)
}

func TestArmRefusesWhatTOTGCannotUse(t *testing.T) {
	t.Run("a zero bound", func(t *testing.T) {
		m := fullyBoundedArm()
		m.Joints[1].Limits.MaxAcceleration = f64(0)
		arm, err := rdk.ArmFromKinematicModel(m, "")
		test.That(t, err, test.ShouldBeNil)
		in, err := trajex.NewTensorMap()
		test.That(t, err, test.ShouldBeNil)
		defer in.Close()
		err = arm.Inputs(in)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "joint 1 has a acceleration bound of 0")
	})
	t.Run("a negative property", func(t *testing.T) {
		m := limitedArm()
		m.Properties.OperatingFreqHz = f64(-5)
		_, err := rdk.ArmFromKinematicModel(m, "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "negative operating frequency -5")
	})
}

func view(t *testing.T, in *trajex.TensorMap, key string) ([]float64, bool) {
	t.Helper()
	_, data, ok, err := in.ViewFloat64s(key)
	test.That(t, err, test.ShouldBeNil)
	return data, ok
}

// Inputs writes a limit key only when every joint has that bound, so a half
// bounded model leaves the key for the caller rather than inventing a value,
// and what it writes is the model's own numbers under the right keys.
func TestArmInputsWritesWhatTheModelCarries(t *testing.T) {
	newInputs := func(t *testing.T, m *commonpb.KinematicModel) *trajex.TensorMap {
		t.Helper()
		arm, err := rdk.ArmFromKinematicModel(m, "")
		test.That(t, err, test.ShouldBeNil)
		in, err := trajex.NewTensorMap()
		test.That(t, err, test.ShouldBeNil)
		t.Cleanup(in.Close)
		test.That(t, arm.Inputs(in), test.ShouldBeNil)
		return in
	}
	t.Run("fully bounded", func(t *testing.T) {
		in := newInputs(t, fullyBoundedArm())
		vel, ok := view(t, in, totg.KeyVelocityLimitsRadsPerSec)
		test.That(t, ok, test.ShouldBeTrue)
		test.That(t, vel, test.ShouldResemble, []float64{utils.DegToRad(60), utils.DegToRad(60)})
		acc, ok := view(t, in, totg.KeyAccelerationLimitsRadsPerSec2)
		test.That(t, ok, test.ShouldBeTrue)
		test.That(t, acc, test.ShouldResemble, []float64{utils.DegToRad(120), utils.DegToRad(120)})
		hz, ok := view(t, in, totg.KeyTrajectorySamplingFreqHz)
		test.That(t, ok, test.ShouldBeTrue)
		test.That(t, hz, test.ShouldResemble, []float64{100})
		cap, ok := view(t, in, totg.KeyTCPMaxLinearSpeed)
		test.That(t, ok, test.ShouldBeTrue)
		test.That(t, cap, test.ShouldResemble, []float64{100})
	})
	t.Run("half bounded", func(t *testing.T) {
		in := newInputs(t, limitedArm())
		_, ok := view(t, in, totg.KeyVelocityLimitsRadsPerSec)
		test.That(t, ok, test.ShouldBeFalse)
		_, ok = view(t, in, totg.KeyAccelerationLimitsRadsPerSec2)
		test.That(t, ok, test.ShouldBeFalse)
		_, ok = view(t, in, totg.KeyTrajectorySamplingFreqHz)
		test.That(t, ok, test.ShouldBeTrue)
		_, ok = view(t, in, totg.KeyTCPMaxLinearSpeed)
		test.That(t, ok, test.ShouldBeTrue)
	})
	t.Run("nothing carried", func(t *testing.T) {
		in := newInputs(t, planarArm())
		for _, key := range []string{totg.KeyVelocityLimitsRadsPerSec, totg.KeyAccelerationLimitsRadsPerSec2, totg.KeyTrajectorySamplingFreqHz, totg.KeyTCPMaxLinearSpeed} {
			_, ok := view(t, in, key)
			test.That(t, ok, test.ShouldBeFalse)
		}
	})
}

// plan runs TOTG on the arm's own inputs over one short move and reports the
// duration and the fastest the end moved between samples, in mm/s.
func plan(t *testing.T, arm *rdk.Arm) (duration, top float64) {
	t.Helper()
	chain, err := kinematics.NewChain(arm.Joints)
	test.That(t, err, test.ShouldBeNil)
	defer chain.Close()
	in, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer in.Close()
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()
	test.That(t, arm.Inputs(in), test.ShouldBeNil)
	test.That(t, in.InsertFloat64s(totg.KeyWaypointsRads, []uint64{2, 2}, []float64{0, 0, 1, -0.5}), test.ShouldBeNil)
	test.That(t, in.InsertScalarFloat64(totg.KeyPathToleranceDeltaRads, 0.01), test.ShouldBeNil)
	test.That(t, totg.GenerateWithKinematics(context.Background(), chain, in, out), test.ShouldBeNil)

	times, ok := view(t, out, totg.KeySampleTimesSec)
	test.That(t, ok, test.ShouldBeTrue)
	configs, ok := view(t, out, totg.KeyConfigurationsRads)
	test.That(t, ok, test.ShouldBeTrue)
	var last [3]float64
	for i := range times {
		p, err := chain.EndPosition(configs[2*i : 2*i+2])
		test.That(t, err, test.ShouldBeNil)
		if i > 0 {
			d := math.Sqrt((p[0]-last[0])*(p[0]-last[0]) + (p[1]-last[1])*(p[1]-last[1]) + (p[2]-last[2])*(p[2]-last[2]))
			top = math.Max(top, d/(times[i]-times[i-1]))
		}
		last = p
	}
	return times[len(times)-1], top
}

// A fully bounded model plans with nothing from the caller but the waypoints
// and the path tolerance. The cap it names binds, since the joints alone would
// let the end move faster, so the capped run is slower and tops out at the cap.
func TestArmInputsDriveTOTG(t *testing.T) {
	capped, err := rdk.ArmFromKinematicModel(fullyBoundedArm(), "")
	test.That(t, err, test.ShouldBeNil)
	uncapped := *capped
	uncapped.MaxTCPSpeedMMPerSec = 0

	freeDuration, freeTop := plan(t, &uncapped)
	cappedDuration, cappedTop := plan(t, capped)
	test.That(t, freeTop, test.ShouldBeGreaterThan, 120)
	test.That(t, cappedTop, test.ShouldBeLessThan, 101)
	test.That(t, cappedTop, test.ShouldBeGreaterThan, 95)
	test.That(t, cappedDuration, test.ShouldBeGreaterThan, freeDuration)
}

// The frame system flavor concatenates the parts' limits in chain order and
// takes the properties from the tip.
func TestArmFromFrameSystemConfig(t *testing.T) {
	gantryModel := &commonpb.KinematicModel{
		Links:  []*commonpb.KinematicLink{link("rail", "world", 0, 0, 0, nil), link("carriage", "slide", 0, 0, 0, nil)},
		Joints: []*commonpb.KinematicJoint{joint("slide", "rail", prismatic, 1, 0, 0)},
	}
	gantryModel.Joints[0].Limits = &commonpb.JointLimits{Min: f64(0), Max: f64(500), MaxVelocity: f64(100)}
	gantry := part("gantry", "world", 0, 0, 0, 0, gantryModel)
	arm := part("arm", "gantry", 0, 0, 50, 0, limitedArm())
	parts := []*robotpb.FrameSystemConfig{arm, gantry}

	whole, err := rdk.ArmFromFrameSystemConfig(parts, "arm")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(whole.Limits), test.ShouldEqual, 3)
	test.That(t, whole.Limits[0].Max, test.ShouldEqual, 500)
	test.That(t, *whole.Limits[0].MaxVelocity, test.ShouldEqual, 100)
	test.That(t, whole.Limits[1].Max, test.ShouldAlmostEqual, math.Pi/2)
	test.That(t, whole.OperatingFreqHz, test.ShouldEqual, 100)
	test.That(t, whole.MaxTCPSpeedMMPerSec, test.ShouldEqual, 250)

	toGantry, err := rdk.ArmFromFrameSystemConfig(parts, "gantry")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(toGantry.Limits), test.ShouldEqual, 1)
	test.That(t, toGantry.OperatingFreqHz, test.ShouldEqual, 0)
	test.That(t, toGantry.MaxTCPSpeedMMPerSec, test.ShouldEqual, 0)
}

// ArmFromModel takes the limits from the model itself, so they are exactly
// its DoF even where the message's degrees would come back a bit off.
func TestArmFromModelMatchesTheModel(t *testing.T) {
	for _, file := range []string{"components/arm/kinematics/ur5e.json", "components/arm/kinematics/lite6.json"} {
		m := loadModel(t, file, "arm")
		arm, err := rdk.ArmFromModel(m, "")
		test.That(t, err, test.ShouldBeNil)
		test.That(t, arm.Limits, test.ShouldResemble, m.DoF())
		joints, err := rdk.JointsFromModel(m, "")
		test.That(t, err, test.ShouldBeNil)
		test.That(t, len(arm.Joints), test.ShouldEqual, len(joints))
	}

	_, err := rdk.ArmFromModel(nil, "")
	test.That(t, err, test.ShouldNotBeNil)
	_, err = rdk.ArmFromKinematicModel(nil, "")
	test.That(t, err, test.ShouldNotBeNil)
}
