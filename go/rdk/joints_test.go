//go:build !windows && cgo

package rdk_test

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/golang/geo/r3"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/rdk/utils"
	"go.viam.com/test"

	"github.com/viam-modules/trajex/go/kinematics"
	"github.com/viam-modules/trajex/go/rdk"
)

// These tests compare the chain's end position with RDK's forward kinematics
// at many joint positions. One comparison covers composition order, the
// quaternion convention, the axes and the units.

func loadModel(t *testing.T, file, name string) referenceframe.Model {
	t.Helper()
	m, err := referenceframe.KinematicModelFromFile(utils.ResolveFile(file), name)
	test.That(t, err, test.ShouldBeNil)
	return m
}

func chainFromModel(t *testing.T, m referenceframe.Model, tcp spatialmath.Pose) *kinematics.Chain {
	t.Helper()
	joints, err := rdk.JointsFromModel(m, tcp)
	test.That(t, err, test.ShouldBeNil)
	return newChain(t, joints)
}

func newChain(t *testing.T, joints []kinematics.Joint) *kinematics.Chain {
	t.Helper()
	chain, err := kinematics.NewChain(joints)
	test.That(t, err, test.ShouldBeNil)
	t.Cleanup(chain.Close)
	return chain
}

// randomPositions draws joint positions within each joint's limits. A
// continuous joint has infinite limits, so for those we draw from two full
// turns either side of zero, which is enough to show that both sides wrap the
// angle the same way.
func randomPositions(m referenceframe.Model, rng *rand.Rand) []float64 {
	limits := m.DoF()
	q := make([]float64, len(limits))
	for i, limit := range limits {
		lo, hi := limit.Min, limit.Max
		if math.IsInf(lo, 0) || math.IsInf(hi, 0) {
			lo, hi = -4*math.Pi, 4*math.Pi
		}
		q[i] = lo + rng.Float64()*(hi-lo)
	}
	return q
}

func endMatches(t *testing.T, chain *kinematics.Chain, q []float64, want r3.Vector) {
	t.Helper()
	got, err := chain.EndPosition(q)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, got[0], test.ShouldAlmostEqual, want.X, 1e-6)
	test.That(t, got[1], test.ShouldAlmostEqual, want.Y, 1e-6)
	test.That(t, got[2], test.ShouldAlmostEqual, want.Z, 1e-6)
}

func assertChainMatchesModel(t *testing.T, m referenceframe.Model, seed int64) {
	t.Helper()
	chain := chainFromModel(t, m, spatialmath.NewZeroPose())
	rng := rand.New(rand.NewSource(seed))
	for range 100 {
		q := randomPositions(m, rng)
		want, err := m.Transform(q)
		test.That(t, err, test.ShouldBeNil)
		endMatches(t, chain, q, want.Point())
	}
}

// The models are every arm RDK ships plus a URDF, a DH model and a gantry
// from its test files, so each file format and joint type is covered.
func TestJointsFromModelMatchesRDKForwardKinematics(t *testing.T) {
	arms, err := filepath.Glob(utils.ResolveFile("components/arm/kinematics/*.json"))
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(arms), test.ShouldBeGreaterThan, 5)
	files := arms
	files = append(files,
		utils.ResolveFile("referenceframe/testfiles/ur5e.urdf"),
		utils.ResolveFile("referenceframe/testfiles/ur5eDH.json"),
		utils.ResolveFile("referenceframe/testfiles/example_gantry.json"),
	)
	for i, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			m, err := referenceframe.KinematicModelFromFile(file, "test")
			test.That(t, err, test.ShouldBeNil)
			assertChainMatchesModel(t, m, int64(i))
		})
	}
}

// Each run of links in this model both rotates and translates, so composing a
// run in the wrong order moves the end. The models RDK ships would not catch
// that.
const orderSensitiveModel = `{
	"name": "order_sensitive",
	"output_frames": ["tip_move"],
	"links": [
		{"id": "base_turn", "parent": "world", "translation": {"x": 0, "y": 0, "z": 0},
		 "orientation": {"type": "euler_angles", "value": {"roll": 0, "pitch": 0, "yaw": 1.2}}},
		{"id": "base_move", "parent": "base_turn", "translation": {"x": 100, "y": 0, "z": 30}},
		{"id": "mid_turn", "parent": "spin", "translation": {"x": 0, "y": 0, "z": 0},
		 "orientation": {"type": "euler_angles", "value": {"roll": 0.7, "pitch": 0, "yaw": 0}}},
		{"id": "mid_move", "parent": "mid_turn", "translation": {"x": 0, "y": 80, "z": 0}},
		{"id": "tip_turn", "parent": "slide", "translation": {"x": 0, "y": 0, "z": 0},
		 "orientation": {"type": "euler_angles", "value": {"roll": 0, "pitch": -0.9, "yaw": 0}}},
		{"id": "tip_move", "parent": "tip_turn", "translation": {"x": 40, "y": 0, "z": 60}},
		{"id": "tool", "parent": "tip_move", "translation": {"x": 10, "y": -20, "z": 150},
		 "orientation": {"type": "euler_angles", "value": {"roll": 0.3, "pitch": 0, "yaw": -0.5}}}
	],
	"joints": [
		{"id": "spin", "type": "revolute", "parent": "base_move", "axis": {"x": 0, "y": 0, "z": 1}, "min": -180, "max": 180},
		{"id": "slide", "type": "prismatic", "parent": "mid_move", "axis": {"x": 1, "y": 0, "z": 0}, "min": -50, "max": 50}
	]
}`

func TestJointsFromModelComposesLinksInOrder(t *testing.T) {
	m, err := referenceframe.UnmarshalModelJSON([]byte(orderSensitiveModel), "")
	test.That(t, err, test.ShouldBeNil)
	assertChainMatchesModel(t, m, 3)
}

// The tool center point is given relative to the output frame, at a rotated
// offset here, so composing it on the wrong side of the last link would move
// the point.
func TestJointsFromModelCapsAToolCenterPoint(t *testing.T) {
	m, err := referenceframe.UnmarshalModelJSON([]byte(orderSensitiveModel), "")
	test.That(t, err, test.ShouldBeNil)
	tcp := spatialmath.NewPose(r3.Vector{X: 10, Y: -20, Z: 150}, &spatialmath.EulerAngles{Roll: 0.3, Pitch: 0, Yaw: -0.5})
	chain := chainFromModel(t, m, tcp)

	rng := rand.New(rand.NewSource(2))
	for range 50 {
		q := randomPositions(m, rng)
		output, err := m.Transform(q)
		test.That(t, err, test.ShouldBeNil)
		endMatches(t, chain, q, spatialmath.Compose(output, tcp).Point())
	}
}

func TestJointsFromModelRefusesAMimicJointOnThePath(t *testing.T) {
	_, err := rdk.JointsFromModel(loadModel(t, "referenceframe/testfiles/test_mimic_serial.json", "test"), spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, `joint "joint3" of model "test" follows another joint`)
}

// In both gripper models the output frame hangs directly off the base while a
// finger elsewhere in the model still takes a position, so positions meant
// for the model would drive a chain that has no joint for them.
func TestJointsFromModelRefusesMovingJointsOffThePath(t *testing.T) {
	for _, file := range []string{
		"referenceframe/testfiles/test_gripper.json",
		"referenceframe/testfiles/test_mimic_gripper.json",
	} {
		t.Run(file, func(t *testing.T) {
			_, err := rdk.JointsFromModel(loadModel(t, file, "test"), spatialmath.NewZeroPose())
			test.That(t, err, test.ShouldNotBeNil)
			test.That(t, err.Error(), test.ShouldContainSubstring, "would drive the wrong joints")
		})
	}
}

// A model assembled from frames in code has no file behind it, and the chain
// comes from its frames all the same. Every frame here both rotates and
// translates and no joint axis is aligned with an axis of the frame, so a
// wrong composition or an unnormalized axis moves the end.
func TestJointsFromModelBuiltFromFramesMatchesRDK(t *testing.T) {
	static := func(name string, pose spatialmath.Pose) referenceframe.Frame {
		f, err := referenceframe.NewStaticFrame(name, pose)
		test.That(t, err, test.ShouldBeNil)
		return f
	}
	turn, err := referenceframe.NewRotationalFrame("turn", spatialmath.R4AA{RX: 0.2, RZ: 1}, referenceframe.Limit{Min: -math.Pi, Max: math.Pi})
	test.That(t, err, test.ShouldBeNil)
	slide, err := referenceframe.NewTranslationalFrame("slide", r3.Vector{X: 1, Y: 0.5}, referenceframe.Limit{Min: -50, Max: 50})
	test.That(t, err, test.ShouldBeNil)
	m, err := referenceframe.NewSerialModel("code", []referenceframe.Frame{
		static("base", spatialmath.NewPose(r3.Vector{X: 10, Y: -20, Z: 30}, &spatialmath.EulerAngles{Roll: 0.3, Pitch: -0.2, Yaw: 1.1})),
		turn,
		static("upper", spatialmath.NewPose(r3.Vector{X: 100, Z: 40}, &spatialmath.EulerAngles{Pitch: 0.5})),
		slide,
		static("tool", spatialmath.NewPose(r3.Vector{Y: 25, Z: 60}, &spatialmath.EulerAngles{Roll: -0.4})),
	})
	test.That(t, err, test.ShouldBeNil)
	assertChainMatchesModel(t, m, 6)

	_, err = rdk.JointsFromModel(nil, spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldNotBeNil)
	_, err = rdk.JointsFromModel(m, nil)
	test.That(t, err, test.ShouldNotBeNil)
}

// A frame that moves in more than one way at once, like RDK's pose frame with
// its seven inputs, has no trajex joint to become.
func TestJointsFromModelRefusesAFrameTrajexCannotExpress(t *testing.T) {
	free, err := referenceframe.NewPoseFrame("free", nil)
	test.That(t, err, test.ShouldBeNil)
	tip, err := referenceframe.NewStaticFrame("tip", spatialmath.NewPoseFromPoint(r3.Vector{Z: 10}))
	test.That(t, err, test.ShouldBeNil)
	m, err := referenceframe.NewSerialModel("free", []referenceframe.Frame{free, tip})
	test.That(t, err, test.ShouldBeNil)
	_, err = rdk.JointsFromModel(m, spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, `frame "free" of model "free" takes 7 positions`)
}

// partsToProtobuf serializes parts the way a machine serves them, with the
// typed model beside the frame.
func partsToProtobuf(t *testing.T, parts []*referenceframe.FrameSystemPart) []*robotpb.FrameSystemConfig {
	t.Helper()
	var out []*robotpb.FrameSystemConfig
	for _, part := range parts {
		pb, err := part.ToProtobuf()
		test.That(t, err, test.ShouldBeNil)
		out = append(out, pb)
	}
	return out
}

// A server older than FrameSystemConfig.model sends the deprecated kinematics
// field only. RDK still fills that field beside the model, so clearing the
// model from its reply reproduces what such a server sends, and the part has
// to build the same chain either way.
func TestJointsFromFrameSystemConfigReadsTheDeprecatedKinematicsField(t *testing.T) {
	arm := loadModel(t, "components/arm/kinematics/ur5e.json", "arm")
	part := &referenceframe.FrameSystemPart{
		FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewZeroPose(), "arm", nil),
		ModelFrame:  arm,
	}
	pb, err := part.ToProtobuf()
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(pb.GetKinematics().GetFields()), test.ShouldBeGreaterThan, 0)
	pb.Model = nil

	joints, err := rdk.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{pb}, "arm")
	test.That(t, err, test.ShouldBeNil)
	chain := newChain(t, joints)
	rng := rand.New(rand.NewSource(8))
	for range 50 {
		q := randomPositions(arm, rng)
		want, err := arm.Transform(q)
		test.That(t, err, test.ShouldBeNil)
		endMatches(t, chain, q, want.Point())
	}
}

// RDK lets a part hang from a frame inside another part's model, named
// part:frame once the model is flattened into the frame system. The probe
// here hangs from the tool link past the arm's output frame, so the chain has
// to follow the arm's model out to that link rather than to its output.
func TestJointsFromFrameSystemConfigHangsAPartInsideAModel(t *testing.T) {
	arm, err := referenceframe.UnmarshalModelJSON([]byte(orderSensitiveModel), "arm")
	test.That(t, err, test.ShouldBeNil)
	probePose := spatialmath.NewPose(r3.Vector{X: 5, Y: -8, Z: 30}, &spatialmath.EulerAngles{Roll: 0.2, Yaw: 1.3})
	parts := []*referenceframe.FrameSystemPart{
		{FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewZeroPose(), "arm", nil), ModelFrame: arm},
		{FrameConfig: referenceframe.NewLinkInFrame("arm:tool", probePose, "probe", nil)},
	}
	fs, err := referenceframe.NewFrameSystem("machine", parts, nil)
	test.That(t, err, test.ShouldBeNil)

	joints, err := rdk.JointsFromFrameSystemConfig(partsToProtobuf(t, parts), "probe")
	test.That(t, err, test.ShouldBeNil)
	chain := newChain(t, joints)

	rng := rand.New(rand.NewSource(7))
	for range 50 {
		q := randomPositions(arm, rng)
		inputs := referenceframe.NewLinearInputs()
		inputs.Put("arm", q)
		want, err := fs.Transform(inputs, referenceframe.NewPoseInFrame("probe", spatialmath.NewZeroPose()), referenceframe.World)
		test.That(t, err, test.ShouldBeNil)
		endMatches(t, chain, q, want.(*referenceframe.PoseInFrame).Pose().Point())
	}
}

// The frame system is a gantry on the floor, a mount rotated and tilted on
// its carriage, and an arm on the mount. The chain built from the parts'
// messages has to put the arm's end where RDK's frame system puts it for the
// same positions.
func TestJointsFromFrameSystemConfigMatchesRDK(t *testing.T) {
	gantry := loadModel(t, "referenceframe/testfiles/example_gantry.json", "gantry")
	arm := loadModel(t, "components/arm/kinematics/ur5e.json", "arm")
	mountPose := spatialmath.NewPose(r3.Vector{X: 120, Y: -40, Z: 25}, &spatialmath.EulerAngles{Roll: 0.4, Pitch: -0.2, Yaw: 2.1})
	armPose := spatialmath.NewPose(r3.Vector{X: 0, Y: 15, Z: 60}, &spatialmath.OrientationVectorDegrees{OX: 0.3, OY: -0.5, OZ: 0.8, Theta: 35})
	parts := []*referenceframe.FrameSystemPart{
		{FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewZeroPose(), "gantry", nil), ModelFrame: gantry},
		{FrameConfig: referenceframe.NewLinkInFrame("gantry", mountPose, "mount", nil)},
		{FrameConfig: referenceframe.NewLinkInFrame("mount", armPose, "arm", nil), ModelFrame: arm},
	}
	fs, err := referenceframe.NewFrameSystem("machine", parts, nil)
	test.That(t, err, test.ShouldBeNil)
	pbParts := partsToProtobuf(t, parts)

	for _, tc := range []struct {
		tip    string
		models []referenceframe.Model
	}{
		{"gantry", []referenceframe.Model{gantry}},
		{"mount", []referenceframe.Model{gantry}},
		{"arm", []referenceframe.Model{gantry, arm}},
	} {
		t.Run(tc.tip, func(t *testing.T) {
			joints, err := rdk.JointsFromFrameSystemConfig(pbParts, tc.tip)
			test.That(t, err, test.ShouldBeNil)
			chain := newChain(t, joints)

			rng := rand.New(rand.NewSource(5))
			for range 50 {
				inputs := referenceframe.NewLinearInputs()
				var q []float64
				for _, m := range tc.models {
					positions := randomPositions(m, rng)
					inputs.Put(m.Name(), positions)
					q = append(q, positions...)
				}
				want, err := fs.Transform(inputs, referenceframe.NewPoseInFrame(tc.tip, spatialmath.NewZeroPose()), referenceframe.World)
				test.That(t, err, test.ShouldBeNil)
				endMatches(t, chain, q, want.(*referenceframe.PoseInFrame).Pose().Point())
			}
		})
	}
}
