//go:build !windows && cgo

package rdk_test

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/golang/geo/r3"
	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/rdk/utils"
	"go.viam.com/test"

	"github.com/viam-modules/trajex/go/kinematics"
	"github.com/viam-modules/trajex/go/rdk"
)

// RDK's own forward kinematics is the reference throughout. Comparing the end
// position of the chain we build against the position RDK reports, at many
// joint positions, checks the order in which poses are composed, the
// quaternion convention, the sense of each axis, and the units all at once.

func loadModel(t *testing.T, file, name string) referenceframe.Model {
	t.Helper()
	m, err := referenceframe.KinematicModelFromFile(utils.ResolveFile(file), name)
	test.That(t, err, test.ShouldBeNil)
	return m
}

func chainFromModel(t *testing.T, m referenceframe.Model, outputFrame string) *kinematics.Chain {
	t.Helper()
	joints, err := rdk.JointsFromModel(m, outputFrame)
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

// randomPositions draws joint positions within each joint's limits. Continuous
// joints have infinite limits, so for those we draw from two full turns
// either side of zero, which is enough to show that the angle wraps the same
// way in both.
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
	endWithin(t, chain, q, want, 1e-6)
}

func endWithin(t *testing.T, chain *kinematics.Chain, q []float64, want r3.Vector, tolerance float64) {
	t.Helper()
	got, err := chain.EndPosition(q)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, got[0], test.ShouldAlmostEqual, want.X, tolerance)
	test.That(t, got[1], test.ShouldAlmostEqual, want.Y, tolerance)
	test.That(t, got[2], test.ShouldAlmostEqual, want.Z, tolerance)
}

// Every arm model RDK ships, plus a URDF, a DH model and a gantry from its
// test files, so that each file format and joint type is covered.
func TestJointsFromModelMatchesRDKForwardKinematics(t *testing.T) {
	arms, err := filepath.Glob(utils.ResolveFile("components/arm/kinematics/*.json"))
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(arms), test.ShouldBeGreaterThan, 5)
	files := append(arms,
		utils.ResolveFile("referenceframe/testfiles/ur5e.urdf"),
		utils.ResolveFile("referenceframe/testfiles/ur5eDH.json"),
		utils.ResolveFile("referenceframe/testfiles/example_gantry.json"),
	)
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			m, err := referenceframe.KinematicModelFromFile(file, "test")
			test.That(t, err, test.ShouldBeNil)
			chain := chainFromModel(t, m, "")

			rng := rand.New(rand.NewSource(1))
			for i := 0; i < 200; i++ {
				q := randomPositions(m, rng)
				want, err := m.Transform(q)
				test.That(t, err, test.ShouldBeNil)
				endMatches(t, chain, q, want.Point())
			}
		})
	}
}

// None of RDK's own models puts a rotated link before a translated one in the
// same run of links, so with them alone the order in which we compose a run
// could be reversed without any test noticing. In this model every run of
// links, before the first joint, between the joints, and after the last one,
// turns and then moves, so composing any of them backwards moves the end. The
// tool link past the output frame is for the test that caps a named frame.
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
	chain := chainFromModel(t, m, "")

	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 50; i++ {
		q := randomPositions(m, rng)
		want, err := m.Transform(q)
		test.That(t, err, test.ShouldBeNil)
		endMatches(t, chain, q, want.Point())
	}
}

// Naming a link past the model's output frame caps the point that link stands
// for, which is how a tool on a flange gets its own cap. The tool link sits at
// a turned offset from the output frame, so composing it on the wrong side
// would move the point.
func TestJointsFromModelCapsANamedFrame(t *testing.T) {
	m, err := referenceframe.UnmarshalModelJSON([]byte(orderSensitiveModel), "")
	test.That(t, err, test.ShouldBeNil)
	chain := chainFromModel(t, m, "tool")
	tool := spatialmath.NewPose(r3.Vector{X: 10, Y: -20, Z: 150}, &spatialmath.EulerAngles{Roll: 0.3, Pitch: 0, Yaw: -0.5})

	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 50; i++ {
		q := randomPositions(m, rng)
		output, err := m.Transform(q)
		test.That(t, err, test.ShouldBeNil)
		endMatches(t, chain, q, spatialmath.Compose(output, tool).Point())
	}
}

func TestJointsFromModelRefusesAMimicJointOnThePath(t *testing.T) {
	_, err := rdk.JointsFromModel(loadModel(t, "referenceframe/testfiles/test_mimic_serial.json", "test"), "")
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, "mimics")
}

// In both gripper models the output frame hangs directly off the base, while
// a finger elsewhere in the model still takes an input. Positions meant for
// the model would then drive a chain that has no joint for them.
func TestJointsFromModelRefusesMovingJointsOffThePath(t *testing.T) {
	for _, file := range []string{
		"referenceframe/testfiles/test_gripper.json",
		"referenceframe/testfiles/test_mimic_gripper.json",
	} {
		t.Run(file, func(t *testing.T) {
			_, err := rdk.JointsFromModel(loadModel(t, file, "test"), "")
			test.That(t, err, test.ShouldNotBeNil)
			test.That(t, err.Error(), test.ShouldContainSubstring, "takes inputs for joints")
		})
	}
}

// A model assembled from frames in code has no configuration, and RDK
// describes it from its frames instead. Every frame here turns and moves at
// once and the joint axes are off the grid, so a wrong composition or an
// unnormalized axis shows as a moved end.
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
	chain := chainFromModel(t, m, "")

	rng := rand.New(rand.NewSource(6))
	for i := 0; i < 50; i++ {
		q := randomPositions(m, rng)
		want, err := m.Transform(q)
		test.That(t, err, test.ShouldBeNil)
		endMatches(t, chain, q, want.Point())
	}

	_, err = rdk.JointsFromModel(nil, "")
	test.That(t, err, test.ShouldNotBeNil)
}

// A machine's whole chain as one model, a gantry carrying a mount, an arm on
// the mount and a tool on the arm, built with NewModel over the frame system.
// RDK flattens each nested model into namespaced frames but keeps the model
// itself as a frame under the part's name, and its frame walker refuses that
// frame, so ModelToProto cannot write the composite out yet. The chain for
// that machine shape comes from JointsFromFrameSystemConfig instead, and this
// pins the refusal until RDK's walker learns to skip a flattened model.
func TestJointsFromModelRefusesACompositeOfNestedModels(t *testing.T) {
	gantry := loadModel(t, "referenceframe/testfiles/example_gantry.json", "gantry")
	arm := loadModel(t, "components/arm/kinematics/ur5e.json", "arm")
	parts := []*referenceframe.FrameSystemPart{
		{FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewZeroPose(), "gantry", nil), ModelFrame: gantry},
		{FrameConfig: referenceframe.NewLinkInFrame("gantry", spatialmath.NewPose(r3.Vector{X: 120, Y: -40, Z: 25}, &spatialmath.EulerAngles{Roll: 0.4}), "mount", nil)},
		{FrameConfig: referenceframe.NewLinkInFrame("mount", spatialmath.NewZeroPose(), "arm", nil), ModelFrame: arm},
		{FrameConfig: referenceframe.NewLinkInFrame("arm", spatialmath.NewPoseFromPoint(r3.Vector{Z: 90}), "tool", nil)},
	}
	fs, err := referenceframe.NewFrameSystem("machine", parts, nil)
	test.That(t, err, test.ShouldBeNil)
	m, err := referenceframe.NewModel("composite", fs, "tool")
	test.That(t, err, test.ShouldBeNil)

	_, err = rdk.JointsFromModel(m, "")
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, "has no kinematics file form")
}

// The message allows four spellings of a link's orientation, and RDK reads
// each of them for us. What is left to check is that composing the poses it
// hands back puts the end where RDK's own forward kinematics puts it, for
// every form and on both sides of the orientation vector's pole.
func TestOrientationFormsReadAsRDKReadsThem(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	angle := func() float64 { return rng.Float64()*720 - 360 }
	unit := func() float64 { return rng.Float64()*2 - 1 }
	forms := []struct {
		name string
		form func() any
	}{
		{"quaternion", func() any {
			return &commonpb.LinkPose_Quaternion{Quaternion: &commonpb.Quaternion{W: 3 * unit(), X: 3 * unit(), Y: 3 * unit(), Z: 3 * unit()}}
		}},
		{"orientation vector", func() any {
			return &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{OX: unit(), OY: unit(), OZ: unit(), Theta: angle()}}
		}},
		{"orientation vector near a pole", func() any {
			return &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{OX: 1e-6 * unit(), OY: 1e-6 * unit(), OZ: math.Copysign(1, unit()), Theta: angle()}}
		}},
		{"euler angles", func() any {
			return &commonpb.LinkPose_EulerAngles{EulerAngles: &commonpb.EulerAngles{Roll: angle(), Pitch: angle(), Yaw: angle()}}
		}},
		{"axis angles", func() any {
			return &commonpb.LinkPose_AxisAngles{AxisAngles: &commonpb.AxisAngles{Rx: 2 * unit(), Ry: 2 * unit(), Rz: 2 * unit(), Theta: angle()}}
		}},
	}
	for _, tc := range forms {
		t.Run(tc.name, func(t *testing.T) {
			for i := 0; i < 25; i++ {
				turn := &commonpb.LinkPose{Translation: &commonpb.Vector3{X: 100 * unit(), Y: 100 * unit(), Z: 100 * unit()}}
				switch o := tc.form().(type) {
				case *commonpb.LinkPose_Quaternion:
					turn.Orientation = o
				case *commonpb.LinkPose_OrientationVector:
					turn.Orientation = o
				case *commonpb.LinkPose_EulerAngles:
					turn.Orientation = o
				case *commonpb.LinkPose_AxisAngles:
					turn.Orientation = o
				}
				pb := &commonpb.KinematicModel{Links: []*commonpb.KinematicLink{
					{Id: "turn", Parent: "world", Pose: turn},
					{Id: "after", Parent: "turn", Pose: &commonpb.LinkPose{Translation: &commonpb.Vector3{X: 100 * unit(), Y: 100 * unit(), Z: 100 * unit()}}},
				}}
				m, err := referenceframe.ModelFromProto(pb, "orientation")
				test.That(t, err, test.ShouldBeNil)
				want, err := m.Transform(nil)
				test.That(t, err, test.ShouldBeNil)

				joints, err := rdk.JointsFromKinematicModel(pb, "")
				test.That(t, err, test.ShouldBeNil)
				endMatches(t, newChain(t, joints), nil, want.Point())
			}
		})
	}
}

// A server older than FrameSystemConfig.model serves a part's model in the
// deprecated kinematics field only. RDK still fills that field beside the
// model, so dropping the model from its own message gives that server's
// reply. The field holds RDK's file format, which this reader does not read,
// so the part is refused by name rather than read as a fixed transform with
// no joints.
func TestJointsFromFrameSystemConfigRefusesTheDeprecatedKinematicsField(t *testing.T) {
	part := &referenceframe.FrameSystemPart{
		FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewZeroPose(), "arm", nil),
		ModelFrame:  loadModel(t, "components/arm/kinematics/ur5e.json", "arm"),
	}
	pb, err := part.ToProtobuf()
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(pb.GetKinematics().GetFields()), test.ShouldBeGreaterThan, 0)
	pb.Model = nil

	_, err = rdk.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{pb}, "arm")
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, `part "arm" carries its model in the deprecated kinematics field only`)
}

// partsToProtobuf writes parts the way a machine serves them, with the typed
// model beside the frame.
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

// A gantry on the floor, a mount turned and tilted on its carriage, and an
// arm on the mount. RDK's frame system places each part on its parent, and
// the chain built from the parts' messages has to put the arm's end in the
// same place for the same positions.
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
			for i := 0; i < 50; i++ {
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
