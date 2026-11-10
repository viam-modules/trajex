//go:build !windows && cgo

package rdk_test

import (
	"math"
	"testing"

	"github.com/golang/geo/r3"
	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/test"

	"github.com/viam-modules/trajex/go/kinematics"
	"github.com/viam-modules/trajex/go/rdk"
)

// The messages here are small enough to work out by hand, which pins down
// which way links compose, where a joint's origin sits, and how each
// orientation form turns into a rotation. joints_test.go checks the same
// things against RDK's forward kinematics.

func link(id, parent string, x, y, z float64, orientation any) *commonpb.KinematicLink {
	pose := &commonpb.LinkPose{Translation: &commonpb.Vector3{X: x, Y: y, Z: z}}
	switch o := orientation.(type) {
	case nil:
	case *commonpb.LinkPose_Quaternion:
		pose.Orientation = o
	case *commonpb.LinkPose_OrientationVector:
		pose.Orientation = o
	case *commonpb.LinkPose_EulerAngles:
		pose.Orientation = o
	case *commonpb.LinkPose_AxisAngles:
		pose.Orientation = o
	default:
		panic("link: orientation is not one of the LinkPose forms")
	}
	return &commonpb.KinematicLink{Id: id, Parent: parent, Pose: pose}
}

func yaw(degrees float64) *commonpb.LinkPose_EulerAngles {
	return &commonpb.LinkPose_EulerAngles{EulerAngles: &commonpb.EulerAngles{Yaw: degrees}}
}

func joint(id, parent string, jointType commonpb.JointType, x, y, z float64) *commonpb.KinematicJoint {
	return &commonpb.KinematicJoint{Id: id, Parent: parent, Type: jointType, Axis: &commonpb.Vector3{X: x, Y: y, Z: z}}
}

const (
	revolute  = commonpb.JointType_JOINT_TYPE_REVOLUTE
	prismatic = commonpb.JointType_JOINT_TYPE_PRISMATIC
)

// planarArm is two revolute joints about z with 100 mm and 80 mm links along
// x, so its end sits at (100 cos q1 + 80 cos(q1+q2), 100 sin q1 + 80 sin(q1+q2), 0).
func planarArm() *commonpb.KinematicModel {
	return &commonpb.KinematicModel{
		Name: "planar",
		Links: []*commonpb.KinematicLink{
			link("base", "world", 0, 0, 0, nil),
			link("upper", "shoulder", 100, 0, 0, nil),
			link("fore", "elbow", 80, 0, 0, nil),
		},
		Joints: []*commonpb.KinematicJoint{
			joint("shoulder", "base", revolute, 0, 0, 1),
			joint("elbow", "upper", revolute, 0, 0, 1),
		},
	}
}

func planarEnd(q1, q2 float64) r3.Vector {
	return r3.Vector{X: 100*math.Cos(q1) + 80*math.Cos(q1+q2), Y: 100*math.Sin(q1) + 80*math.Sin(q1+q2)}
}

func modelChain(t *testing.T, m *commonpb.KinematicModel, outputFrame string) *kinematics.Chain {
	t.Helper()
	joints, err := rdk.JointsFromKinematicModel(m, outputFrame)
	test.That(t, err, test.ShouldBeNil)
	return newChain(t, joints)
}

func partsChain(t *testing.T, parts []*robotpb.FrameSystemConfig, tip string) *kinematics.Chain {
	t.Helper()
	joints, err := rdk.JointsFromFrameSystemConfig(parts, tip)
	test.That(t, err, test.ShouldBeNil)
	return newChain(t, joints)
}

func TestPlanarArmMatchesTrigonometry(t *testing.T) {
	joints, err := rdk.JointsFromKinematicModel(planarArm(), "")
	test.That(t, err, test.ShouldBeNil)
	// two moving joints and the fixed joint every chain ends in
	test.That(t, len(joints), test.ShouldEqual, 3)
	c := newChain(t, joints)
	for _, q := range [][2]float64{{0, 0}, {0.3, -1.1}, {2.5, 0.4}, {-3, 3}} {
		endMatches(t, c, q[:], planarEnd(q[0], q[1]))
	}
}

// A rotated link before a prismatic joint turns the joint's axis with it, and
// a link after the joint rides along with the slide.
func TestRotatedLinkTurnsAPrismaticJoint(t *testing.T) {
	m := &commonpb.KinematicModel{
		Links: []*commonpb.KinematicLink{
			link("base", "world", 0, 0, 0, yaw(90)),
			link("tip", "slide", 10, 0, 0, nil),
		},
		Joints: []*commonpb.KinematicJoint{joint("slide", "base", prismatic, 1, 0, 0)},
	}
	c := modelChain(t, m, "")
	endMatches(t, c, []float64{0}, r3.Vector{Y: 10})
	endMatches(t, c, []float64{25}, r3.Vector{Y: 35})
}

// RDK normalizes a joint's axis when it builds the frame, and so does trajex,
// so an axis of any length gives the same chain.
func TestJointAxisLengthDoesNotMatter(t *testing.T) {
	m := planarArm()
	m.Joints[0].Axis.Z = 3
	m.Joints[1].Axis.Z = 0.5
	endMatches(t, modelChain(t, m, ""), []float64{0.3, -1.1}, planarEnd(0.3, -1.1))

	slide := &commonpb.KinematicModel{
		Links:  []*commonpb.KinematicLink{link("base", "world", 0, 0, 0, nil), link("tip", "slide", 10, 0, 0, nil)},
		Joints: []*commonpb.KinematicJoint{joint("slide", "base", prismatic, 2, 0, 0)},
	}
	endMatches(t, modelChain(t, slide, ""), []float64{25}, r3.Vector{X: 35})
}

// RDK hangs an element from world when its parent is not an element of the
// model, whatever the parent is called, and the walk to the root has to stop
// at the same place.
func TestRootIsWhereParentsStop(t *testing.T) {
	for _, tc := range []struct{ name, parent string }{
		{"no parent", ""},
		{"a parent that is not an element", "floor"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := planarArm()
			m.Links[0].Parent = tc.parent
			endMatches(t, modelChain(t, m, ""), []float64{0.3, -1.1}, planarEnd(0.3, -1.1))
		})
	}
}

// Each orientation form is checked by what it does to a link placed after it,
// compared with the rotation its values describe, so a change in how RDK reads
// a form shows up here as a moved end.
func TestOrientationForms(t *testing.T) {
	r := 1 / math.Sqrt2
	for _, tc := range []struct {
		name        string
		orientation any
		after       r3.Vector
		want        r3.Vector
	}{
		{"quaternion 90 about z", &commonpb.LinkPose_Quaternion{Quaternion: &commonpb.Quaternion{W: r, Z: r}}, r3.Vector{X: 10}, r3.Vector{Y: 10}},
		{"quaternion not of unit length", &commonpb.LinkPose_Quaternion{Quaternion: &commonpb.Quaternion{W: 2, Z: 2}}, r3.Vector{X: 10}, r3.Vector{Y: 10}},
		{"euler roll 90", &commonpb.LinkPose_EulerAngles{EulerAngles: &commonpb.EulerAngles{Roll: 90}}, r3.Vector{Y: 10}, r3.Vector{Z: 10}},
		{"euler yaw 90", yaw(90), r3.Vector{X: 10}, r3.Vector{Y: 10}},
		{"axis angles with a non-unit axis", &commonpb.LinkPose_AxisAngles{AxisAngles: &commonpb.AxisAngles{Rx: 3, Theta: 90}}, r3.Vector{Y: 10}, r3.Vector{Z: 10}},
		// an orientation vector along z with theta is a turn about z
		{"orientation vector about z", &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{OZ: 1, Theta: 90}}, r3.Vector{X: 10}, r3.Vector{Y: 10}},
		// RDK reads a zero quaternion as no rotation rather than refusing it
		{"zero quaternion", &commonpb.LinkPose_Quaternion{Quaternion: &commonpb.Quaternion{}}, r3.Vector{X: 10}, r3.Vector{X: 10}},
		// pointing z along x is a turn of 90 about y, which sends x to minus z
		{"orientation vector along x", &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{OX: 5}}, r3.Vector{X: 10}, r3.Vector{Z: -10}},
		// within the pole radius theta is measured without the vector's longitude, so a vector slightly off
		// z at 45 degrees of longitude with theta 30 is still a turn of 30 about z, not 75
		{"orientation vector near the pole", &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{OX: 1e-7, OY: 1e-7, OZ: 1, Theta: 30}}, r3.Vector{X: 10}, r3.Vector{X: 10 * math.Cos(math.Pi/6), Y: 10 * math.Sin(math.Pi/6)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &commonpb.KinematicModel{
				Links: []*commonpb.KinematicLink{
					link("turn", "world", 0, 0, 0, tc.orientation),
					link("after", "turn", tc.after.X, tc.after.Y, tc.after.Z, nil),
				},
			}
			// the near pole case leans the frame by its vector's small tilt, which moves the end by about a micrometre
			endWithin(t, modelChain(t, m, ""), nil, tc.want, 1e-5)
		})
	}
}

func TestOutputFrameChoice(t *testing.T) {
	// a fork: two leaves hang off the elbow, so the model has to say which is its output
	forked := func(outputFrames ...string) *commonpb.KinematicModel {
		m := planarArm()
		m.Links = append(m.Links, link("spur", "elbow", 0, 30, 0, nil))
		m.OutputFrames = outputFrames
		return m
	}

	t.Run("the model's output frame", func(t *testing.T) {
		c := modelChain(t, forked("fore"), "")
		endMatches(t, c, []float64{0, 0}, r3.Vector{X: 180})
	})
	t.Run("a frame the caller names", func(t *testing.T) {
		c := modelChain(t, forked("fore"), "spur")
		endMatches(t, c, []float64{0, 0}, r3.Vector{X: 100, Y: 30})
	})
	t.Run("a joint the caller names", func(t *testing.T) {
		// the chain ends at the elbow's own frame, which only the shoulder turns
		c := modelChain(t, planarArm(), "elbow")
		endMatches(t, c, []float64{0.3, -1.1}, r3.Vector{X: 100 * math.Cos(0.3), Y: 100 * math.Sin(0.3)})
	})
	t.Run("a frame short of the last moving joint", func(t *testing.T) {
		// the elbow still takes a position, which a chain ending at the upper link has no joint for
		_, err := rdk.JointsFromKinematicModel(planarArm(), "upper")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `moving joints [elbow] off the path to its output frame "upper"`)
	})
	t.Run("a frame past the model's output", func(t *testing.T) {
		m := planarArm()
		m.OutputFrames = []string{"fore"}
		m.Links = append(m.Links, link("tool", "fore", 0, 0, 25, nil))
		c := modelChain(t, m, "tool")
		endMatches(t, c, []float64{0, 0}, r3.Vector{X: 180, Z: 25})
	})
	t.Run("the single leaf when none is named", func(t *testing.T) {
		c := modelChain(t, planarArm(), "")
		endMatches(t, c, []float64{0, 0}, r3.Vector{X: 180})
	})
	t.Run("several output frames", func(t *testing.T) {
		_, err := rdk.JointsFromKinematicModel(forked("fore", "spur"), "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "declares 2 output frames [fore spur], and a chain ends at one")
	})
	t.Run("two leaves and none named", func(t *testing.T) {
		_, err := rdk.JointsFromKinematicModel(forked(), "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "has 2 leaves [fore spur] rather than one")
	})
	t.Run("a frame the model lacks", func(t *testing.T) {
		_, err := rdk.JointsFromKinematicModel(planarArm(), "nowhere")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `"nowhere" is not a link or joint`)
		m := planarArm()
		m.OutputFrames = []string{"nowhere"}
		_, err = rdk.JointsFromKinematicModel(m, "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `output frame "nowhere" is not a link or joint`)
	})
}

// RDK orders a model's positions breadth first from the root, not by the
// order joints appear in the message, so a message that lists them the other
// way round still gives a chain. Along one path that order is base to tip,
// and a moving joint off the path, which RDK would also take a position for,
// is refused.
func TestInputOrder(t *testing.T) {
	t.Run("message order does not matter", func(t *testing.T) {
		m := planarArm()
		m.Joints[0], m.Joints[1] = m.Joints[1], m.Joints[0]
		c := modelChain(t, m, "")
		endMatches(t, c, []float64{0.3, -1.1}, planarEnd(0.3, -1.1))
	})
	t.Run("a moving joint off the path", func(t *testing.T) {
		m := planarArm()
		m.Joints = append(m.Joints, joint("finger", "base", revolute, 0, 1, 0))
		m.Links = append(m.Links, link("fingertip", "finger", 0, 0, 5, nil))
		m.OutputFrames = []string{"fore"}
		_, err := rdk.JointsFromKinematicModel(m, "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `model "planar" has moving joints [finger] off the path to its output frame "fore"`)
	})
	t.Run("a mimic joint off the path takes no position", func(t *testing.T) {
		m := planarArm()
		m.Joints = append(m.Joints, &commonpb.KinematicJoint{
			Id: "follower", Parent: "base", Type: revolute, Axis: &commonpb.Vector3{Z: 1},
			Mimic: &commonpb.MimicJoint{Joint: "shoulder", Multiplier: -1},
		})
		m.Links = append(m.Links, link("follower_tip", "follower", 0, 0, 5, nil))
		m.OutputFrames = []string{"fore"}
		c := modelChain(t, m, "")
		endMatches(t, c, []float64{0.3, -1.1}, planarEnd(0.3, -1.1))
	})
}

// Each of these messages would give a wrong chain rather than no chain, so
// each has to be refused. RDK refuses to load every one of them, by an error
// or in two cases by a panic, and the reader has to refuse them on its own.
func TestRefusals(t *testing.T) {
	cyclic := planarArm()
	cyclic.Links[0].Parent = "fore"
	cyclic.OutputFrames = []string{"fore"}

	strayCycle := planarArm()
	strayCycle.Links = append(strayCycle.Links, link("loop_a", "loop_b", 0, 0, 0, nil), link("loop_b", "loop_a", 0, 0, 0, nil))
	strayCycle.OutputFrames = []string{"fore"}

	mimicOnPath := planarArm()
	mimicOnPath.Joints[1].Mimic = &commonpb.MimicJoint{Joint: "shoulder", Multiplier: 1}

	noAxis := planarArm()
	noAxis.Joints[1].Axis = nil

	unknownType := planarArm()
	unknownType.Joints[1].Type = commonpb.JointType_JOINT_TYPE_UNSPECIFIED

	zeroOrientationVector := planarArm()
	zeroOrientationVector.Links[0].Pose.Orientation = &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{Theta: 90}}

	zeroAxisAngles := planarArm()
	zeroAxisAngles.Links[0].Pose.Orientation = &commonpb.LinkPose_AxisAngles{AxisAngles: &commonpb.AxisAngles{Theta: 10}}

	// small enough that the sum of squares RDK normalizes by underflows to zero
	tinyAxis := planarArm()
	tinyAxis.Joints[1].Axis = &commonpb.Vector3{X: 1e-200}

	tinyAxisAngles := planarArm()
	tinyAxisAngles.Links[0].Pose.Orientation = &commonpb.LinkPose_AxisAngles{AxisAngles: &commonpb.AxisAngles{Rx: 1e-200, Theta: 10}}

	duplicate := planarArm()
	duplicate.Joints[1].Id = "upper"

	unnamed := planarArm()
	unnamed.Links[2].Id = ""

	reserved := planarArm()
	reserved.Links[0].Id = "world"
	reserved.Joints[0].Parent = "world"

	for _, tc := range []struct {
		name  string
		model *commonpb.KinematicModel
		want  string
	}{
		{"nil", nil, "kinematic model is nil"},
		{"empty", &commonpb.KinematicModel{Name: "hollow"}, `model "hollow" has no links or joints`},
		{"duplicate id", duplicate, `names "upper" more than once`},
		{"missing id", unnamed, "with no id"},
		{"world as an id", reserved, `named "world", which RDK keeps for the root`},
		{"cycle on the path", cyclic, "has a cycle of parents"},
		{"cycle off the path", strayCycle, "has a cycle of parents"},
		{"mimic on the path", mimicOnPath, `joint "elbow" of model "planar" mimics another joint`},
		{"joint without an axis", noAxis, `joint "elbow" of model "planar" has no axis`},
		{"joint axis too small to normalize", tinyAxis, `joint "elbow" of model "planar" has no axis`},
		{"joint of an unknown type", unknownType, `joint "elbow" of model "planar" has type "JOINT_TYPE_UNSPECIFIED", which trajex does not support`},
		{"zero orientation vector", zeroOrientationVector, "has a normal of 0"},
		{"zero axis in axis angles", zeroAxisAngles, `link "base" of model "planar" turns about a zero axis`},
		{"axis angles too small to normalize", tinyAxisAngles, `link "base" of model "planar" turns about a zero axis`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := rdk.JointsFromKinematicModel(tc.model, "")
			test.That(t, err, test.ShouldNotBeNil)
			test.That(t, err.Error(), test.ShouldContainSubstring, tc.want)
		})
	}
}

func part(name, parent string, x, y, z, thetaAboutZ float64, model *commonpb.KinematicModel) *robotpb.FrameSystemConfig {
	return &robotpb.FrameSystemConfig{
		Frame: &commonpb.Transform{
			ReferenceFrame: name,
			PoseInObserverFrame: &commonpb.PoseInFrame{
				ReferenceFrame: parent,
				Pose:           &commonpb.Pose{X: x, Y: y, Z: z, OZ: 1, Theta: thetaAboutZ},
			},
		},
		Model: model,
	}
}

func TestFrameSystemConfig(t *testing.T) {
	// a mount turned 90 degrees about z and 100 mm out along x, carrying the planar arm 50 mm up
	mount := part("mount", "world", 100, 0, 0, 90, nil)
	arm := part("arm", "mount", 0, 0, 50, 0, planarArm())
	// the arm's end seen from the mount is 50 mm up, and the mount turns x into y
	expect := func(q1, q2 float64) r3.Vector {
		p := planarEnd(q1, q2)
		return r3.Vector{X: 100 - p.Y, Y: p.X, Z: 50}
	}

	t.Run("a mount and an arm", func(t *testing.T) {
		joints, err := rdk.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{arm, mount}, "arm")
		test.That(t, err, test.ShouldBeNil)
		// the two fixed transforms fold into the first joint's origin
		test.That(t, len(joints), test.ShouldEqual, 3)
		c := newChain(t, joints)
		for _, q := range [][2]float64{{0, 0}, {0.3, -1.1}, {2.5, 0.4}} {
			endMatches(t, c, q[:], expect(q[0], q[1]))
		}
	})
	t.Run("a part off the path takes no positions", func(t *testing.T) {
		other := part("other", "world", -300, 0, 0, 0, planarArm())
		joints, err := rdk.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{other, arm, mount}, "arm")
		test.That(t, err, test.ShouldBeNil)
		test.That(t, len(joints), test.ShouldEqual, 3)
		endMatches(t, newChain(t, joints), []float64{0.3, -1.1}, expect(0.3, -1.1))
	})
	t.Run("a part pose with no orientation vector turns about z", func(t *testing.T) {
		flat := &robotpb.FrameSystemConfig{Frame: &commonpb.Transform{
			ReferenceFrame:      "flat",
			PoseInObserverFrame: &commonpb.PoseInFrame{ReferenceFrame: "world", Pose: &commonpb.Pose{X: 5, Theta: 90}},
		}}
		c := partsChain(t, []*robotpb.FrameSystemConfig{flat, part("on_flat", "flat", 10, 0, 0, 0, nil)}, "on_flat")
		endMatches(t, c, nil, r3.Vector{X: 5, Y: 10})
	})
	t.Run("an empty model is no model", func(t *testing.T) {
		c := partsChain(t, []*robotpb.FrameSystemConfig{part("bare", "world", 0, 30, 0, 0, &commonpb.KinematicModel{})}, "bare")
		endMatches(t, c, nil, r3.Vector{Y: 30})
	})
	t.Run("the mount alone is a fixed transform", func(t *testing.T) {
		c := partsChain(t, []*robotpb.FrameSystemConfig{arm, mount}, "mount")
		endMatches(t, c, nil, r3.Vector{X: 100})
	})
	t.Run("positions run from the part nearest world outward", func(t *testing.T) {
		gantry := part("gantry", "world", 0, 0, 0, 0, &commonpb.KinematicModel{
			Links: []*commonpb.KinematicLink{
				link("rail", "world", 0, 0, 0, nil),
				link("carriage", "slide", 0, 0, 0, nil),
			},
			Joints: []*commonpb.KinematicJoint{joint("slide", "rail", prismatic, 1, 0, 0)},
		})
		riding := part("arm", "gantry", 0, 0, 50, 0, planarArm())
		c := partsChain(t, []*robotpb.FrameSystemConfig{riding, gantry}, "arm")
		p := planarEnd(0.3, -1.1)
		endMatches(t, c, []float64{40, 0.3, -1.1}, r3.Vector{X: 40 + p.X, Y: p.Y, Z: 50})
	})
	t.Run("a model with no name is named after its part", func(t *testing.T) {
		m := planarArm()
		m.Name = ""
		m.Joints = append(m.Joints, joint("finger", "base", revolute, 0, 1, 0))
		m.OutputFrames = []string{"fore"}
		_, err := rdk.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{part("gripper", "world", 0, 0, 0, 0, m)}, "gripper")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `model "gripper" has moving joints [finger] off the path`)
	})

	for _, tc := range []struct {
		name  string
		parts []*robotpb.FrameSystemConfig
		tip   string
		want  string
	}{
		{"no tip", []*robotpb.FrameSystemConfig{mount}, "", "tip of the chain is not named"},
		{"unknown tip", []*robotpb.FrameSystemConfig{mount}, "arm", `has no part "arm"`},
		{"part without a frame", []*robotpb.FrameSystemConfig{{}}, "arm", "part 0 of the frame system"},
		{"part without a parent", []*robotpb.FrameSystemConfig{part("loose", "", 0, 0, 0, 0, nil)}, "loose", "part 0 of the frame system"},
		{"part without a pose", []*robotpb.FrameSystemConfig{{Frame: &commonpb.Transform{
			ReferenceFrame: "bare", PoseInObserverFrame: &commonpb.PoseInFrame{ReferenceFrame: "world"},
		}}}, "bare", "part 0 of the frame system has no pose"},
		{"duplicate part", []*robotpb.FrameSystemConfig{mount, mount}, "mount", `names part "mount" more than once`},
		{"parent that is not a part", []*robotpb.FrameSystemConfig{arm}, "arm", `part "arm" hangs from "mount", which is not a part`},
		{"cycle", []*robotpb.FrameSystemConfig{part("a", "b", 0, 0, 0, 0, nil), part("b", "a", 0, 0, 0, 0, nil)}, "a", "is a cycle"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := rdk.JointsFromFrameSystemConfig(tc.parts, tc.tip)
			test.That(t, err, test.ShouldNotBeNil)
			test.That(t, err.Error(), test.ShouldContainSubstring, tc.want)
		})
	}
}
