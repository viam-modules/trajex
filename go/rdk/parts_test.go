//go:build !windows && cgo

package rdk_test

import (
	"math"
	"testing"

	"github.com/golang/geo/r3"
	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/test"
	"google.golang.org/protobuf/proto"

	"github.com/viam-modules/trajex/go/kinematics"
	"github.com/viam-modules/trajex/go/rdk"
)

// These tests build FrameSystemConfig messages by hand, with a planar arm
// whose end position is a line of trigonometry, so the part handling is
// checked against numbers rather than against RDK.

func link(id, parent string, x, y, z float64) *commonpb.KinematicLink {
	return &commonpb.KinematicLink{Id: id, Parent: parent, Pose: &commonpb.LinkPose{Translation: &commonpb.Vector3{X: x, Y: y, Z: z}}}
}

func joint(id, parent string, jointType commonpb.JointType, x, y, z float64) *commonpb.KinematicJoint {
	return &commonpb.KinematicJoint{
		Id: id, Parent: parent, Type: jointType, Axis: &commonpb.Vector3{X: x, Y: y, Z: z},
		Limits: &commonpb.JointLimits{Min: proto.Float64(-1000), Max: proto.Float64(1000)},
	}
}

const (
	revolute  = commonpb.JointType_JOINT_TYPE_REVOLUTE
	prismatic = commonpb.JointType_JOINT_TYPE_PRISMATIC
)

// planarArm is two revolute joints about z with 100 mm and 80 mm links, so
// its end is a line of trigonometry.
func planarArm() *commonpb.KinematicModel {
	return &commonpb.KinematicModel{
		Name: "planar",
		Links: []*commonpb.KinematicLink{
			link("base", "world", 0, 0, 0),
			link("upper", "shoulder", 100, 0, 0),
			link("fore", "elbow", 80, 0, 0),
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

func partsChain(t *testing.T, parts []*robotpb.FrameSystemConfig, tip string) *kinematics.Chain {
	t.Helper()
	joints, err := rdk.JointsFromFrameSystemConfig(parts, tip)
	test.That(t, err, test.ShouldBeNil)
	return newChain(t, joints)
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
			Name: "gantry",
			Links: []*commonpb.KinematicLink{
				link("rail", "world", 0, 0, 0),
				link("carriage", "slide", 0, 0, 0),
			},
			Joints: []*commonpb.KinematicJoint{joint("slide", "rail", prismatic, 1, 0, 0)},
		})
		riding := part("arm", "gantry", 0, 0, 50, 0, planarArm())
		c := partsChain(t, []*robotpb.FrameSystemConfig{riding, gantry}, "arm")
		p := planarEnd(0.3, -1.1)
		endMatches(t, c, []float64{40, 0.3, -1.1}, r3.Vector{X: 40 + p.X, Y: p.Y, Z: 50})
	})
	// RDK flattens a part's model into the frame system as part:frame, and a part may hang
	// from one of those frames, here a tool link past the arm's output frame
	armWithTool := planarArm()
	armWithTool.OutputFrames = []string{"fore"}
	armWithTool.Links = append(armWithTool.Links, link("tool", "fore", 0, 0, 25))
	armOnFloor := part("arm", "world", 0, 0, 0, 0, armWithTool)
	t.Run("a part hung from a frame inside another part's model", func(t *testing.T) {
		probe := part("probe", "arm:tool", 0, 0, 10, 0, nil)
		joints, err := rdk.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{probe, armOnFloor}, "probe")
		test.That(t, err, test.ShouldBeNil)
		test.That(t, len(joints), test.ShouldEqual, 3)
		p := planarEnd(0.3, -1.1)
		endMatches(t, newChain(t, joints), []float64{0.3, -1.1}, r3.Vector{X: p.X, Y: p.Y, Z: 35})
	})
	t.Run("a model is named after its part", func(t *testing.T) {
		m := planarArm()
		m.Joints = append(m.Joints, joint("finger", "base", revolute, 0, 1, 0))
		m.OutputFrames = []string{"fore"}
		_, err := rdk.JointsFromFrameSystemConfig([]*robotpb.FrameSystemConfig{part("gripper", "world", 0, 0, 0, 0, m)}, "gripper")
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, `model "gripper" takes 3 positions, but the path to "fore" moves joints [shoulder elbow]`)
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
		{"a model RDK refuses", []*robotpb.FrameSystemConfig{part("bad", "world", 0, 0, 0, 0, &commonpb.KinematicModel{
			Links:  []*commonpb.KinematicLink{link("a", "world", 0, 0, 0)},
			Joints: []*commonpb.KinematicJoint{joint("j", "a", commonpb.JointType_JOINT_TYPE_UNSPECIFIED, 0, 0, 1)},
		})}, "bad", `part "bad"`},
		{"parent that is not a part", []*robotpb.FrameSystemConfig{arm}, "arm", `part "arm" hangs from "mount", which is not a part`},
		{"cycle", []*robotpb.FrameSystemConfig{part("a", "b", 0, 0, 0, 0, nil), part("b", "a", 0, 0, 0, 0, nil)}, "a", "is a cycle"},
		{"hung from a frame short of the model's last joint", []*robotpb.FrameSystemConfig{part("probe", "arm:upper", 0, 0, 0, 0, nil), armOnFloor}, "probe",
			`the path to "upper" moves joints [shoulder] at positions [0]`},
		{"hung from a frame the model lacks", []*robotpb.FrameSystemConfig{part("probe", "arm:nowhere", 0, 0, 0, 0, nil), armOnFloor}, "probe",
			`part "arm": frame with name "nowhere"`},
		{"hung from a frame of a part with no model", []*robotpb.FrameSystemConfig{part("probe", "mount:top", 0, 0, 0, 0, nil), mount}, "probe",
			`part "probe" hangs from frame "top" of part "mount", which has no model`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := rdk.JointsFromFrameSystemConfig(tc.parts, tc.tip)
			test.That(t, err, test.ShouldNotBeNil)
			test.That(t, err.Error(), test.ShouldContainSubstring, tc.want)
		})
	}
}
