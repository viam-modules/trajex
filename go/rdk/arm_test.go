//go:build !windows && cgo

package rdk_test

import (
	"context"
	"math"
	"testing"

	commonpb "go.viam.com/api/common/v1"
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

// The limits come back in the chain's units and position order, which is
// RDK's, with every bound the message lacks absent rather than zero.
func TestArmCarriesLimitsAndProperties(t *testing.T) {
	arm, err := rdk.ArmFromKinematicModel(limitedArm(), "")
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
	test.That(t, arm.OperatingHz, test.ShouldEqual, 100)
	test.That(t, arm.MaxTCPSpeedMMPerSec, test.ShouldEqual, 250)

	bare, err := rdk.ArmFromKinematicModel(planarArm(), "")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, math.IsInf(bare.Limits[0].Min, -1), test.ShouldBeTrue)
	test.That(t, bare.Limits[0].MaxVelocity, test.ShouldBeNil)
	test.That(t, bare.OperatingHz, test.ShouldEqual, 0)
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

// Inputs writes a limit key only when every joint has that bound, so a half
// bounded model leaves the key for the caller rather than inventing a value.
func TestArmInputsWritesWhatTheModelCarries(t *testing.T) {
	present := func(t *testing.T, in *trajex.TensorMap, key string) bool {
		t.Helper()
		_, _, ok, err := in.ViewFloat64s(key)
		test.That(t, err, test.ShouldBeNil)
		return ok
	}
	t.Run("half bounded", func(t *testing.T) {
		arm, err := rdk.ArmFromKinematicModel(limitedArm(), "")
		test.That(t, err, test.ShouldBeNil)
		in, err := trajex.NewTensorMap()
		test.That(t, err, test.ShouldBeNil)
		defer in.Close()
		test.That(t, arm.Inputs(in), test.ShouldBeNil)
		test.That(t, present(t, in, totg.KeyVelocityLimitsRadsPerSec), test.ShouldBeFalse)
		test.That(t, present(t, in, totg.KeyAccelerationLimitsRadsPerSec2), test.ShouldBeFalse)
		test.That(t, present(t, in, totg.KeyTrajectorySamplingFreqHz), test.ShouldBeTrue)
		test.That(t, present(t, in, totg.KeyTCPMaxLinearSpeed), test.ShouldBeTrue)
	})
	t.Run("nothing carried", func(t *testing.T) {
		arm, err := rdk.ArmFromKinematicModel(planarArm(), "")
		test.That(t, err, test.ShouldBeNil)
		in, err := trajex.NewTensorMap()
		test.That(t, err, test.ShouldBeNil)
		defer in.Close()
		test.That(t, arm.Inputs(in), test.ShouldBeNil)
		for _, key := range []string{totg.KeyVelocityLimitsRadsPerSec, totg.KeyAccelerationLimitsRadsPerSec2, totg.KeyTrajectorySamplingFreqHz, totg.KeyTCPMaxLinearSpeed} {
			test.That(t, present(t, in, key), test.ShouldBeFalse)
		}
	})
}

// A fully bounded model plans with nothing from the caller but the waypoints
// and the path tolerance, and the cap the model names holds.
func TestArmInputsDriveTOTG(t *testing.T) {
	m := limitedArm()
	m.Joints[1].Limits = &commonpb.JointLimits{Min: f64(-150), Max: f64(150), MaxVelocity: f64(60), MaxAcceleration: f64(120)}
	arm, err := rdk.ArmFromKinematicModel(m, "")
	test.That(t, err, test.ShouldBeNil)
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

	_, times, ok, err := out.ViewFloat64s(totg.KeySampleTimesSec)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	shape, configs, ok, err := out.ViewFloat64s(totg.KeyConfigurationsRads)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, shape[1], test.ShouldEqual, 2)
	// the fastest the end moves between samples, against the 250 mm/s the model named
	var top float64
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
	test.That(t, top, test.ShouldBeLessThan, 250*1.01)
	test.That(t, top, test.ShouldBeGreaterThan, 100)
}

func TestArmFromModelMatchesTheModel(t *testing.T) {
	m := loadModel(t, "components/arm/kinematics/ur5e.json", "arm")
	arm, err := rdk.ArmFromModel(m, "")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, arm.Limits, test.ShouldResemble, m.DoF())
	joints, err := rdk.JointsFromModel(m, "")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(arm.Joints), test.ShouldEqual, len(joints))

	_, err = rdk.ArmFromModel(nil, "")
	test.That(t, err, test.ShouldNotBeNil)
	_, err = rdk.ArmFromKinematicModel(nil, "")
	test.That(t, err, test.ShouldNotBeNil)
}
