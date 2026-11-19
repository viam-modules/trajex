//go:build !windows && cgo

package rdk_test

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/golang/geo/r3"
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
	test.That(t, err.Error(), test.ShouldContainSubstring, `model "test": joint "joint3" follows another joint`)
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
	_, _, err = rdk.JointsFromChain(nil, nil)
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
	test.That(t, err.Error(), test.ShouldContainSubstring, `model "free": frame "free" takes 7 positions`)
}

// A caller holding a machine's frame system builds a model over it with RDK and
// hands that in. The frame system is a gantry on the floor, a mount rotated and
// tilted on its carriage, and an arm on the mount, and the chain has to put the
// arm's end where RDK's frame system puts it for the same positions, which run
// from the part nearest world outward as the model's inputs do.
func TestJointsFromModelOverFrameSystemPartsMatchesRDK(t *testing.T) {
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
	machine, err := referenceframe.NewModel("machine", fs, "arm")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(machine.DoF()), test.ShouldEqual, len(gantry.DoF())+len(arm.DoF()))
	chain := chainFromModel(t, machine, spatialmath.NewZeroPose())

	rng := rand.New(rand.NewSource(5))
	for range 50 {
		q := randomPositions(machine, rng)
		inputs := referenceframe.NewLinearInputs()
		inputs.Put("gantry", q[:len(gantry.DoF())])
		inputs.Put("arm", q[len(gantry.DoF()):])
		want, err := fs.Transform(inputs, referenceframe.NewPoseInFrame("arm", spatialmath.NewZeroPose()), referenceframe.World)
		test.That(t, err, test.ShouldBeNil)
		endMatches(t, chain, q, want.(*referenceframe.PoseInFrame).Pose().Point())
	}
}

// RDK lets a part hang from a frame inside another part's model, named
// part:frame once the model is flattened into the frame system. The probe
// here hangs from the tool link past the arm's output frame, and a model
// whose output is the probe follows the arm's model out to that link.
func TestJointsFromModelFollowsAPartHungInsideAModel(t *testing.T) {
	arm, err := referenceframe.UnmarshalModelJSON([]byte(orderSensitiveModel), "arm")
	test.That(t, err, test.ShouldBeNil)
	probePose := spatialmath.NewPose(r3.Vector{X: 5, Y: -8, Z: 30}, &spatialmath.EulerAngles{Roll: 0.2, Yaw: 1.3})
	parts := []*referenceframe.FrameSystemPart{
		{FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewZeroPose(), "arm", nil), ModelFrame: arm},
		{FrameConfig: referenceframe.NewLinkInFrame("arm:tool", probePose, "probe", nil)},
	}
	fs, err := referenceframe.NewFrameSystem("machine", parts, nil)
	test.That(t, err, test.ShouldBeNil)
	machine, err := referenceframe.NewModel("machine", fs, "probe")
	test.That(t, err, test.ShouldBeNil)
	chain := chainFromModel(t, machine, spatialmath.NewZeroPose())

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

// A part off the path to the output frame still takes positions in a model
// built over the whole frame system, so such a model is refused like a
// gripper with a finger. A caller builds the frame system from the parts on
// the path instead.
func TestJointsFromModelRefusesAPartOffThePath(t *testing.T) {
	arm := loadModel(t, "components/arm/kinematics/ur5e.json", "arm")
	other := loadModel(t, "components/arm/kinematics/xarm6.json", "other")
	parts := []*referenceframe.FrameSystemPart{
		{FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewZeroPose(), "arm", nil), ModelFrame: arm},
		{FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewPoseFromPoint(r3.Vector{X: 1000}), "other", nil), ModelFrame: other},
	}
	fs, err := referenceframe.NewFrameSystem("machine", parts, nil)
	test.That(t, err, test.ShouldBeNil)
	machine, err := referenceframe.NewModel("machine", fs, "arm")
	test.That(t, err, test.ShouldBeNil)
	_, err = rdk.JointsFromModel(machine, spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, `model "machine" takes 12 positions`)
}

// linearInputsOf keys a machine model's flat positions by the parts of the
// frame system it was built over, in the model's own order, which is what the
// frame system takes.
func linearInputsOf(t *testing.T, fs *referenceframe.FrameSystem, machine referenceframe.Model, q []float64) *referenceframe.LinearInputs {
	t.Helper()
	li := referenceframe.NewLinearInputs()
	next := 0
	for _, name := range machine.(*referenceframe.SimpleModel).MoveableFrameNames() {
		n := len(fs.Frame(name).DoF())
		li.Put(name, q[next:next+n])
		next += n
	}
	test.That(t, next, test.ShouldEqual, len(q))
	return li
}

// A chain cut out of the machine between two frames of the caller's choosing,
// here the arm seen from its mount, takes only the arm's positions, and the
// inputs JointsFromChain reports say where to gather them from. The composed
// end has to land where RDK's frame system puts the arm relative to the mount,
// whatever the gantry and the other arm are doing.
func TestJointsFromChainBetweenTwoFrames(t *testing.T) {
	gantry := loadModel(t, "referenceframe/testfiles/example_gantry.json", "gantry")
	arm := loadModel(t, "components/arm/kinematics/ur5e.json", "arm")
	mountPose := spatialmath.NewPose(r3.Vector{X: 120, Y: -40, Z: 25}, &spatialmath.EulerAngles{Roll: 0.4, Pitch: -0.2, Yaw: 2.1})
	armPose := spatialmath.NewPose(r3.Vector{X: 0, Y: 15, Z: 60}, &spatialmath.OrientationVectorDegrees{OX: 0.3, OY: -0.5, OZ: 0.8, Theta: 35})
	parts := []*referenceframe.FrameSystemPart{
		{FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewZeroPose(), "gantry", nil), ModelFrame: gantry},
		{FrameConfig: referenceframe.NewLinkInFrame("gantry", mountPose, "mount", nil)},
		{FrameConfig: referenceframe.NewLinkInFrame("mount", armPose, "arm", nil), ModelFrame: arm},
		{FrameConfig: referenceframe.NewLinkInFrame(referenceframe.World, spatialmath.NewPoseFromPoint(r3.Vector{X: 2000}), "other", nil), ModelFrame: arm},
	}
	fs, err := referenceframe.NewFrameSystem("machine", parts, nil)
	test.That(t, err, test.ShouldBeNil)
	machine, err := referenceframe.NewModel("machine", fs, "arm")
	test.That(t, err, test.ShouldBeNil)

	// the whole machine cannot be a chain, the other arm takes positions too
	_, err = rdk.JointsFromModel(machine, spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, "would drive the wrong joints")

	for _, base := range []string{"mount", "gantry", referenceframe.World} {
		t.Run(base, func(t *testing.T) {
			chain, err := machine.ChainFrom(base, "")
			test.That(t, err, test.ShouldBeNil)
			joints, inputs, err := rdk.JointsFromChain(chain, spatialmath.NewZeroPose())
			test.That(t, err, test.ShouldBeNil)
			c := newChain(t, joints)
			// a base at the gantry means the gantry's output frame, past its joint
			if base == referenceframe.World {
				test.That(t, len(inputs), test.ShouldEqual, len(gantry.DoF())+len(arm.DoF()))
			} else {
				test.That(t, len(inputs), test.ShouldEqual, len(arm.DoF()))
			}

			rng := rand.New(rand.NewSource(9))
			for range 50 {
				q := randomPositions(machine, rng)
				positions := make([]float64, len(inputs))
				for i, src := range inputs {
					positions[i] = q[src]
				}
				want, err := fs.Transform(linearInputsOf(t, fs, machine, q), referenceframe.NewPoseInFrame("arm", spatialmath.NewZeroPose()), base)
				test.That(t, err, test.ShouldBeNil)
				endMatches(t, c, positions, want.(*referenceframe.PoseInFrame).Pose().Point())
			}
		})
	}
}
