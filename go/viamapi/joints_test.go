//go:build !windows && cgo

package viamapi_test

import (
	"math"
	"testing"

	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/test"

	"github.com/viam-modules/trajex/go/kinematics"
	"github.com/viam-modules/trajex/go/viamapi"
)

// The chains here are small enough to work out by hand, which is what pins
// down the conventions: which way links compose, where a joint's origin sits,
// and how each orientation form on the wire turns into a rotation. Agreement
// with RDK's own reading of the same messages is checked in the rdk package,
// which may import RDK; this one must not.

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

func planarEnd(q1, q2 float64) [3]float64 {
	return [3]float64{100*math.Cos(q1) + 80*math.Cos(q1+q2), 100*math.Sin(q1) + 80*math.Sin(q1+q2), 0}
}

func chain(t *testing.T, joints []kinematics.Joint, err error) *kinematics.Chain {
	t.Helper()
	test.That(t, err, test.ShouldBeNil)
	c, err := kinematics.NewChain(joints)
	test.That(t, err, test.ShouldBeNil)
	t.Cleanup(c.Close)
	return c
}

func modelChain(t *testing.T, m *commonpb.KinematicModel, outputFrame string) *kinematics.Chain {
	t.Helper()
	joints, err := viamapi.JointsFromKinematicModel(m, outputFrame)
	return chain(t, joints, err)
}

func partsChain(t *testing.T, parts []*robotpb.FrameSystemConfig, tip string) *kinematics.Chain {
	t.Helper()
	joints, err := viamapi.JointsFromFrameSystemConfig(parts, tip)
	return chain(t, joints, err)
}

func endIs(t *testing.T, c *kinematics.Chain, q []float64, want [3]float64) {
	t.Helper()
	got, err := c.EndPosition(q)
	test.That(t, err, test.ShouldBeNil)
	for i := range want {
		test.That(t, got[i], test.ShouldAlmostEqual, want[i], 1e-9)
	}
}

func TestPlanarArmMatchesTrigonometry(t *testing.T) {
	joints, err := viamapi.JointsFromKinematicModel(planarArm(), "")
	c := chain(t, joints, err)
	// two moving joints and the fixed joint every chain ends in
	test.That(t, len(joints), test.ShouldEqual, 3)
	for _, q := range [][2]float64{{0, 0}, {0.3, -1.1}, {2.5, 0.4}, {-3, 3}} {
		endIs(t, c, q[:], planarEnd(q[0], q[1]))
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
	endIs(t, c, []float64{0}, [3]float64{0, 10, 0})
	endIs(t, c, []float64{25}, [3]float64{0, 35, 0})
}

// Every orientation form is checked by what it does to a link placed after
// it, against the rotation the values spell out.
func TestOrientationForms(t *testing.T) {
	r := 1 / math.Sqrt2
	for _, tc := range []struct {
		name        string
		orientation any
		after       [3]float64
		want        [3]float64
	}{
		{"quaternion 90 about z", &commonpb.LinkPose_Quaternion{Quaternion: &commonpb.Quaternion{W: r, Z: r}}, [3]float64{10, 0, 0}, [3]float64{0, 10, 0}},
		{"quaternion not of unit length", &commonpb.LinkPose_Quaternion{Quaternion: &commonpb.Quaternion{W: 2, Z: 2}}, [3]float64{10, 0, 0}, [3]float64{0, 10, 0}},
		{"euler roll 90", &commonpb.LinkPose_EulerAngles{EulerAngles: &commonpb.EulerAngles{Roll: 90}}, [3]float64{0, 10, 0}, [3]float64{0, 0, 10}},
		{"euler yaw 90", yaw(90), [3]float64{10, 0, 0}, [3]float64{0, 10, 0}},
		{"axis angles with a non-unit axis", &commonpb.LinkPose_AxisAngles{AxisAngles: &commonpb.AxisAngles{Rx: 3, Theta: 90}}, [3]float64{0, 10, 0}, [3]float64{0, 0, 10}},
		// an orientation vector along z with theta is a turn about z
		{"orientation vector about z", &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{OZ: 1, Theta: 90}}, [3]float64{10, 0, 0}, [3]float64{0, 10, 0}},
		// RDK reads a zero quaternion as no rotation rather than refusing it
		{"zero quaternion", &commonpb.LinkPose_Quaternion{Quaternion: &commonpb.Quaternion{}}, [3]float64{10, 0, 0}, [3]float64{10, 0, 0}},
		// pointing z along x is a turn of 90 about y, which sends x to minus z
		{"orientation vector along x", &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{OX: 5}}, [3]float64{10, 0, 0}, [3]float64{0, 0, -10}},
		// within the pole radius theta is measured without the vector's longitude, so a vector a hair off
		// z at 45 degrees of longitude with theta 30 is still a turn of 30 about z, not 75
		{"orientation vector near the pole", &commonpb.LinkPose_OrientationVector{OrientationVector: &commonpb.Orientation{OX: 1e-7, OY: 1e-7, OZ: 1, Theta: 30}}, [3]float64{10, 0, 0}, [3]float64{10 * math.Cos(math.Pi/6), 10 * math.Sin(math.Pi/6), 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &commonpb.KinematicModel{
				Links: []*commonpb.KinematicLink{
					link("turn", "world", 0, 0, 0, tc.orientation),
					link("after", "turn", tc.after[0], tc.after[1], tc.after[2], nil),
				},
			}
			c := modelChain(t, m, "")
			got, err := c.EndPosition(nil)
			test.That(t, err, test.ShouldBeNil)
			// the near pole case leans the frame by its vector's small tilt, which moves the end by about a micrometre
			for i := range tc.want {
				test.That(t, got[i], test.ShouldAlmostEqual, tc.want[i], 1e-5)
			}
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
		endIs(t, c, []float64{0, 0}, [3]float64{180, 0, 0})
	})
	t.Run("a frame the caller names", func(t *testing.T) {
		c := modelChain(t, forked("fore"), "spur")
		endIs(t, c, []float64{0, 0}, [3]float64{100, 30, 0})
	})
	t.Run("a frame past the model's output", func(t *testing.T) {
		m := planarArm()
		m.OutputFrames = []string{"fore"}
		m.Links = append(m.Links, link("tool", "fore", 0, 0, 25, nil))
		c := modelChain(t, m, "tool")
		endIs(t, c, []float64{0, 0}, [3]float64{180, 0, 25})
	})
	t.Run("the single leaf when none is named", func(t *testing.T) {
		c := modelChain(t, planarArm(), "")
		endIs(t, c, []float64{0, 0}, [3]float64{180, 0, 0})
	})
	t.Run("several output frames", func(t *testing.T) {
		_, err := viamapi.JointsFromKinematicModel(forked("fore", "spur"), "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "2 output frames [fore spur]")
	})
	t.Run("two leaves and none named", func(t *testing.T) {
		_, err := viamapi.JointsFromKinematicModel(forked(), "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "2 leaves [fore spur]")
	})
	t.Run("a frame the model lacks", func(t *testing.T) {
		_, err := viamapi.JointsFromKinematicModel(planarArm(), "nowhere")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `"nowhere" is not a link or joint`)
		m := planarArm()
		m.OutputFrames = []string{"nowhere"}
		_, err = viamapi.JointsFromKinematicModel(m, "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `"nowhere" is not a link or joint`)
	})
}

// RDK orders a model's positions breadth first with siblings sorted by id, not
// by the order joints appear in the message, so a message that lists them the
// other way round still gives a chain, and a moving joint off the path to the
// output frame, which RDK would also take a position for, is refused.
func TestInputOrder(t *testing.T) {
	t.Run("message order does not matter", func(t *testing.T) {
		m := planarArm()
		m.Joints[0], m.Joints[1] = m.Joints[1], m.Joints[0]
		c := modelChain(t, m, "")
		endIs(t, c, []float64{0.3, -1.1}, planarEnd(0.3, -1.1))
	})
	t.Run("siblings are ordered by id, not by the message", func(t *testing.T) {
		// aardvark is listed after elbow but sorts before it, so RDK takes its position first
		m := planarArm()
		m.Joints = append(m.Joints, joint("aardvark", "upper", revolute, 0, 1, 0))
		m.Links = append(m.Links, link("aardvark_tip", "aardvark", 0, 0, 5, nil))
		m.OutputFrames = []string{"fore"}
		_, err := viamapi.JointsFromKinematicModel(m, "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "takes inputs for joints [shoulder aardvark elbow]")
	})
	t.Run("a moving joint off the path", func(t *testing.T) {
		m := planarArm()
		m.Joints = append(m.Joints, joint("finger", "base", revolute, 0, 1, 0))
		m.Links = append(m.Links, link("fingertip", "finger", 0, 0, 5, nil))
		m.OutputFrames = []string{"fore"}
		_, err := viamapi.JointsFromKinematicModel(m, "")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "takes inputs for joints [finger shoulder elbow]")
		test.That(t, err.Error(), test.ShouldContainSubstring, "moves joints [shoulder elbow]")
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
		endIs(t, c, []float64{0.3, -1.1}, planarEnd(0.3, -1.1))
	})
}

// Each refusal stands in for a chain that would be wrong rather than missing:
// a mimic joint has no position of its own, a cycle never reaches a root, a
// joint without an axis would only be refused by index once the chain is
// built, and a bad orientation would quietly become the identity.
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

	duplicate := planarArm()
	duplicate.Joints[1].Id = "upper"

	unnamed := planarArm()
	unnamed.Links[2].Id = ""

	for _, tc := range []struct {
		name  string
		model *commonpb.KinematicModel
		want  string
	}{
		{"nil", nil, "kinematic model is nil"},
		{"empty", &commonpb.KinematicModel{Name: "hollow"}, `model "hollow" has no links or joints`},
		{"duplicate id", duplicate, `names "upper" more than once`},
		{"missing id", unnamed, "with no id"},
		{"cycle on the path", cyclic, "is a cycle"},
		{"cycle off the path", strayCycle, "do not descend from its root"},
		{"mimic on the path", mimicOnPath, `joint "elbow" mimics another joint`},
		{"joint without an axis", noAxis, `joint "elbow" has no axis`},
		{"joint of an unknown type", unknownType, `joint "elbow" has type JOINT_TYPE_UNSPECIFIED`},
		{"zero orientation vector", zeroOrientationVector, `link "base" of model "planar": orientation vector is zero`},
		{"zero axis in axis angles", zeroAxisAngles, `link "base" of model "planar": axis angles have a zero axis`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := viamapi.JointsFromKinematicModel(tc.model, "")
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
	expect := func(q1, q2 float64) [3]float64 {
		p := planarEnd(q1, q2)
		return [3]float64{100 - p[1], p[0], 50}
	}

	t.Run("a mount and an arm", func(t *testing.T) {
		joints, err := viamapi.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{arm, mount}, "arm")
		c := chain(t, joints, err)
		// the two fixed transforms fold into the first joint's origin
		test.That(t, len(joints), test.ShouldEqual, 3)
		for _, q := range [][2]float64{{0, 0}, {0.3, -1.1}, {2.5, 0.4}} {
			endIs(t, c, q[:], expect(q[0], q[1]))
		}
	})
	t.Run("a part off the path takes no positions", func(t *testing.T) {
		other := part("other", "world", -300, 0, 0, 0, planarArm())
		joints, err := viamapi.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{other, arm, mount}, "arm")
		c := chain(t, joints, err)
		test.That(t, len(joints), test.ShouldEqual, 3)
		endIs(t, c, []float64{0.3, -1.1}, expect(0.3, -1.1))
	})
	t.Run("a part pose with no orientation vector turns about z", func(t *testing.T) {
		flat := &robotpb.FrameSystemConfig{Frame: &commonpb.Transform{
			ReferenceFrame:      "flat",
			PoseInObserverFrame: &commonpb.PoseInFrame{ReferenceFrame: "world", Pose: &commonpb.Pose{X: 5, Theta: 90}},
		}}
		c := partsChain(t, []*robotpb.FrameSystemConfig{flat, part("on_flat", "flat", 10, 0, 0, 0, nil)}, "on_flat")
		endIs(t, c, nil, [3]float64{5, 10, 0})
	})
	t.Run("the mount alone is a fixed transform", func(t *testing.T) {
		c := partsChain(t, []*robotpb.FrameSystemConfig{arm, mount}, "mount")
		endIs(t, c, nil, [3]float64{100, 0, 0})
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
		endIs(t, c, []float64{40, 0.3, -1.1}, [3]float64{40 + p[0], p[1], 50})
	})
	t.Run("a model with no name is named after its part", func(t *testing.T) {
		m := planarArm()
		m.Name = ""
		m.Joints = append(m.Joints, joint("finger", "base", revolute, 0, 1, 0))
		m.OutputFrames = []string{"fore"}
		_, err := viamapi.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{part("gripper", "world", 0, 0, 0, 0, m)}, "gripper")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `model "gripper" takes inputs`)
	})

	for _, tc := range []struct {
		name  string
		parts []*robotpb.FrameSystemConfig
		tip   string
		want  string
	}{
		{"no tip", []*robotpb.FrameSystemConfig{mount}, "", "tip of the chain is not named"},
		{"unknown tip", []*robotpb.FrameSystemConfig{mount}, "arm", `has no part "arm"`},
		{"part without a frame", []*robotpb.FrameSystemConfig{{}}, "arm", "part 0 of the frame system has no frame name"},
		{"duplicate part", []*robotpb.FrameSystemConfig{mount, mount}, "mount", `names part "mount" more than once`},
		{"parent that is not a part", []*robotpb.FrameSystemConfig{arm}, "arm", `part "arm" hangs from "mount", which is not a part`},
		{"cycle", []*robotpb.FrameSystemConfig{part("a", "b", 0, 0, 0, 0, nil), part("b", "a", 0, 0, 0, 0, nil)}, "a", "is a cycle"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := viamapi.JointsFromFrameSystemConfig(tc.parts, tc.tip)
			test.That(t, err, test.ShouldNotBeNil)
			test.That(t, err.Error(), test.ShouldContainSubstring, tc.want)
		})
	}
}
