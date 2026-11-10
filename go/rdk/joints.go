//go:build !windows && cgo

// Package rdk builds trajex kinematic chains from the KinematicModel and
// FrameSystemConfig messages of the Viam API. It reads a message by the
// rules RDK's referenceframe package applies when it loads one, so a chain
// built here agrees with the model RDK itself drives, and the tests check
// that against RDK's forward kinematics. Units follow RDK: millimetres and
// radians.
package rdk

import (
	"math"
	"slices"
	"sort"

	"github.com/golang/geo/r3"
	"github.com/pkg/errors"
	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/rdk/spatialmath"

	"github.com/viam-modules/trajex/go/kinematics"
)

// world is the name RDK gives the root of every frame tree.
const world = "world"

// JointsFromKinematicModel describes the serial chain from the base of model
// out to outputFrame, for kinematics.NewChain. The chain always ends in a
// fixed joint.
//
// outputFrame names a link or joint of the model, or is empty for the model's
// own output frame, which is the one it declares or else its single leaf.
// RDK takes positions breadth first from the root, which along one path is
// base to tip, so every moving joint of the model has to lie on the path to
// outputFrame. A mimic joint on the path, a moving joint off it such as a
// gripper finger, and a message RDK would refuse to load are errors.
//
// Joint limits are not part of the chain. A caller takes them from the
// model, the DoF of referenceframe.ModelFromProto on the same message, whose
// order is the chain's since both run from the base to the tip.
func JointsFromKinematicModel(model *commonpb.KinematicModel, outputFrame string) ([]kinematics.Joint, error) {
	if model == nil {
		return nil, errors.New("trajex/rdk: kinematic model is nil")
	}
	b := &chainBuilder{origin: spatialmath.NewZeroPose()}
	if err := b.appendModel(model, model.GetName(), outputFrame); err != nil {
		return nil, err
	}
	return b.finish(), nil
}

// JointsFromFrameSystemConfig describes the serial chain from the world frame
// out to the output frame of the part named tip, for a caller holding a
// robot's FrameSystemConfig reply. Each part on the path contributes the fixed
// transform that places it on its parent, then the joints of its model as
// JointsFromKinematicModel builds them, so positions run from the part
// nearest world out to tip. Parts off the path contribute nothing.
//
// A parent has to be world or another part of the reply. A part whose model
// arrives only in the deprecated kinematics field, which a server older than
// FrameSystemConfig.model sends, is refused.
func JointsFromFrameSystemConfig(parts []*robotpb.FrameSystemConfig, tip string) ([]kinematics.Joint, error) {
	if tip == "" {
		return nil, errors.New("trajex/rdk: the tip of the chain is not named")
	}
	type placed struct {
		name, parent string
		pose         spatialmath.Pose
		model        *commonpb.KinematicModel
		// legacy is set when the model came in the deprecated kinematics field,
		// which holds RDK's file format rather than a KinematicModel
		legacy bool
	}
	byName := map[string]placed{}
	for i, part := range parts {
		frame := part.GetFrame()
		name := frame.GetReferenceFrame()
		if name == "" {
			return nil, errors.Errorf("trajex/rdk: part %d of the frame system has no name", i)
		}
		parent := frame.GetPoseInObserverFrame().GetReferenceFrame()
		if parent == "" {
			return nil, errors.Errorf("trajex/rdk: part %d of the frame system, %q, has no parent", i, name)
		}
		// spatialmath reads the pose without a nil check, so refuse a missing one rather than let it panic
		pose := frame.GetPoseInObserverFrame().GetPose()
		if pose == nil {
			return nil, errors.Errorf("trajex/rdk: part %d of the frame system has no pose", i)
		}
		if _, dup := byName[name]; dup {
			return nil, errors.Errorf("trajex/rdk: the frame system names part %q more than once", name)
		}
		// an empty model is no model, which is how RDK reads it too
		model := part.GetModel()
		if len(model.GetLinks())+len(model.GetJoints()) == 0 {
			model = nil
		}
		byName[name] = placed{
			name:   name,
			parent: parent,
			pose:   spatialmath.NewPoseFromProtobuf(pose),
			model:  model,
			legacy: model == nil && len(part.GetKinematics().GetFields()) > 0,
		}
	}
	if _, ok := byName[tip]; !ok {
		return nil, errors.Errorf("trajex/rdk: the frame system has no part %q", tip)
	}

	var path []placed
	seen := map[string]bool{}
	for name, child := tip, ""; name != world; {
		part, ok := byName[name]
		if !ok {
			return nil, errors.Errorf("trajex/rdk: part %q hangs from %q, which is not a part of the frame system", child, name)
		}
		if seen[name] {
			return nil, errors.Errorf("trajex/rdk: the path from %q to the world frame is a cycle", tip)
		}
		seen[name] = true
		path = append(path, part)
		name, child = part.parent, name
	}
	slices.Reverse(path)

	b := &chainBuilder{origin: spatialmath.NewZeroPose()}
	for _, part := range path {
		b.fixed(part.pose)
		if part.legacy {
			return nil, errors.Errorf(
				"trajex/rdk: part %q carries its model in the deprecated kinematics field only, which a server older than FrameSystemConfig.model sends and this reader does not read",
				part.name)
		}
		if part.model == nil {
			continue
		}
		label := part.model.GetName()
		if label == "" {
			label = part.name
		}
		if err := b.appendModel(part.model, label, ""); err != nil {
			return nil, err
		}
	}
	return b.finish(), nil
}

// chainBuilder accumulates joints from the base outward. trajex attaches the
// fixed transform ahead of a joint to that joint as its origin, whereas RDK
// models fixed transforms as links of their own, so the links between two
// joints are composed into origin and attached to the next joint.
type chainBuilder struct {
	joints []kinematics.Joint
	origin spatialmath.Pose
}

func (b *chainBuilder) fixed(p spatialmath.Pose) {
	b.origin = spatialmath.Compose(b.origin, p)
}

func (b *chainBuilder) rotational(axis [3]float64) {
	xyz, wxyz := components(b.origin)
	b.joints = append(b.joints, kinematics.RotationalJoint(xyz, wxyz, axis))
	b.origin = spatialmath.NewZeroPose()
}

func (b *chainBuilder) translational(axis [3]float64) {
	xyz, wxyz := components(b.origin)
	b.joints = append(b.joints, kinematics.TranslationalJoint(xyz, wxyz, axis))
	b.origin = spatialmath.NewZeroPose()
}

func (b *chainBuilder) finish() []kinematics.Joint {
	xyz, wxyz := components(b.origin)
	return append(b.joints, kinematics.FixedJoint(xyz, wxyz))
}

// appendModel adds the model's path to outputFrame to the chain. label names
// the model in errors, since a model inside a frame system part may have no
// name of its own.
func (b *chainBuilder) appendModel(model *commonpb.KinematicModel, label, outputFrame string) error {
	elements, err := index(model, label)
	if err != nil {
		return err
	}
	output, err := outputFrameOf(model, elements, label, outputFrame)
	if err != nil {
		return err
	}

	onPath := map[string]bool{}
	for _, id := range pathToRoot(elements, output) {
		e := elements[id]
		if e.link != nil {
			pose, err := linkPose(e.link, label)
			if err != nil {
				return err
			}
			b.fixed(pose)
			continue
		}
		if e.joint.GetMimic() != nil {
			return errors.Errorf("trajex/rdk: joint %q of model %q mimics another joint, which trajex cannot express", id, label)
		}
		axis := e.joint.GetAxis()
		switch e.joint.GetType() {
		case commonpb.JointType_JOINT_TYPE_REVOLUTE, commonpb.JointType_JOINT_TYPE_CONTINUOUS:
			b.rotational([3]float64{axis.GetX(), axis.GetY(), axis.GetZ()})
		case commonpb.JointType_JOINT_TYPE_PRISMATIC:
			b.translational([3]float64{axis.GetX(), axis.GetY(), axis.GetZ()})
		default:
			return errors.Errorf("trajex/rdk: joint %q of model %q has type %q, which trajex does not support", id, label, e.joint.GetType())
		}
		onPath[id] = true
	}

	// RDK numbers its positions breadth first from the root, which along one path is base
	// to tip, the order the chain was just built in. A moving joint anywhere else would
	// take a position the chain has no joint for, so the chain would read the rest wrong.
	var off []string
	for _, joint := range model.GetJoints() {
		if joint.GetMimic() == nil && !onPath[joint.GetId()] {
			off = append(off, joint.GetId())
		}
	}
	if len(off) > 0 {
		return errors.Errorf(
			"trajex/rdk: model %q has moving joints %v off the path to its output frame %q, so a chain for that path cannot take the model's positions",
			label, off, output)
	}
	return nil
}

// element is one link or joint of the message. Exactly one of link and joint
// is set.
type element struct {
	parent string
	link   *commonpb.KinematicLink
	joint  *commonpb.KinematicJoint
}

// index maps the model's links and joints by id and refuses what RDK refuses
// to load, plus what it does not refuse cleanly: an element with no id, a
// duplicate id, which RDK silently merges, and a zero joint axis, on which RDK
// panics. A joint of a type trajex cannot drive is refused here too, wherever
// it sits in the model, since RDK would not load the model at all.
func index(model *commonpb.KinematicModel, label string) (map[string]*element, error) {
	if len(model.GetLinks()) == 0 && len(model.GetJoints()) == 0 {
		return nil, errors.Errorf("trajex/rdk: model %q has no links or joints", label)
	}
	elements := map[string]*element{}
	add := func(id string, e *element) error {
		if id == "" {
			return errors.Errorf("trajex/rdk: model %q has a link or joint with no id", label)
		}
		if id == world {
			return errors.Errorf("trajex/rdk: model %q has a link or joint named %q, which RDK keeps for the root", label, world)
		}
		if _, dup := elements[id]; dup {
			return errors.Errorf("trajex/rdk: model %q names %q more than once", label, id)
		}
		elements[id] = e
		return nil
	}
	for _, link := range model.GetLinks() {
		if err := add(link.GetId(), &element{parent: link.GetParent(), link: link}); err != nil {
			return nil, err
		}
	}
	for _, joint := range model.GetJoints() {
		switch joint.GetType() {
		case commonpb.JointType_JOINT_TYPE_REVOLUTE, commonpb.JointType_JOINT_TYPE_CONTINUOUS, commonpb.JointType_JOINT_TYPE_PRISMATIC:
		default:
			return nil, errors.Errorf("trajex/rdk: joint %q of model %q has type %q, which trajex does not support", joint.GetId(), label, joint.GetType())
		}
		if axis := joint.GetAxis(); zeroLength(axis.GetX(), axis.GetY(), axis.GetZ()) {
			return nil, errors.Errorf("trajex/rdk: joint %q of model %q has no axis", joint.GetId(), label)
		}
		if err := add(joint.GetId(), &element{parent: joint.GetParent(), joint: joint}); err != nil {
			return nil, err
		}
	}
	if err := checkTree(elements, label); err != nil {
		return nil, err
	}
	return elements, nil
}

// checkTree refuses a cycle of parents anywhere in the model, on or off the
// path, since RDK refuses to load the model.
func checkTree(elements map[string]*element, label string) error {
	rooted := map[string]bool{}
	for start := range elements {
		var trail []string
		id := start
		for elements[id] != nil && !rooted[id] {
			if slices.Contains(trail, id) {
				return errors.Errorf("trajex/rdk: model %q has a cycle of parents through %q", label, id)
			}
			trail = append(trail, id)
			id = elements[id].parent
		}
		for _, id := range trail {
			rooted[id] = true
		}
	}
	return nil
}

// zeroLength matches the test RDK's normalization applies, so a vector whose
// squared length underflows to zero is zero here too.
func zeroLength(x, y, z float64) bool {
	return x*x+y*y+z*z == 0
}

// outputFrameOf picks the frame the chain ends at the way RDK picks a model's
// output frame: the caller's if named, else the one the model declares, else
// its single leaf. More than one declared output frame, or no declared frame
// and more than one leaf, is refused as RDK refuses it.
func outputFrameOf(model *commonpb.KinematicModel, elements map[string]*element, label, requested string) (string, error) {
	if requested != "" {
		if _, ok := elements[requested]; !ok {
			return "", errors.Errorf("trajex/rdk: output frame %q is not a link or joint of model %q", requested, label)
		}
		return requested, nil
	}
	switch declared := model.GetOutputFrames(); len(declared) {
	case 0:
	case 1:
		if _, ok := elements[declared[0]]; !ok {
			return "", errors.Errorf("trajex/rdk: output frame %q is not a link or joint of model %q", declared[0], label)
		}
		return declared[0], nil
	default:
		return "", errors.Errorf("trajex/rdk: model %q declares %d output frames %v, and a chain ends at one", label, len(declared), declared)
	}
	hasChild := map[string]bool{}
	for _, e := range elements {
		hasChild[e.parent] = true
	}
	var leaves []string
	for id := range elements {
		if !hasChild[id] {
			leaves = append(leaves, id)
		}
	}
	if len(leaves) != 1 {
		sort.Strings(leaves)
		return "", errors.Errorf("trajex/rdk: model %q declares no output frame and has %d leaves %v rather than one", label, len(leaves), leaves)
	}
	return leaves[0], nil
}

// pathToRoot lists the elements from the model's root, the first whose parent
// is not an element, down to output. RDK hangs such an element from world
// whatever its parent is called, so the walk stops there too.
func pathToRoot(elements map[string]*element, output string) []string {
	var path []string
	for id := output; elements[id] != nil; id = elements[id].parent {
		path = append(path, id)
	}
	slices.Reverse(path)
	return path
}

// linkPose reads a link's pose the way RDK does. Each orientation form maps
// onto the spatialmath type it names, degrees become radians where the
// message carries degrees, a quaternion is normalized and a zero one is no
// rotation. An orientation vector of zero length is refused as RDK refuses
// it, and axis angles about a zero axis are refused because RDK panics on
// them.
func linkPose(link *commonpb.KinematicLink, label string) (spatialmath.Pose, error) {
	t := link.GetPose().GetTranslation()
	point := r3.Vector{X: t.GetX(), Y: t.GetY(), Z: t.GetZ()}
	var o spatialmath.Orientation
	switch v := link.GetPose().GetOrientation().(type) {
	case nil:
		return spatialmath.NewPoseFromPoint(point), nil
	case *commonpb.LinkPose_Quaternion:
		q := v.Quaternion
		raw := spatialmath.Quaternion{Real: q.GetW(), Imag: q.GetX(), Jmag: q.GetY(), Kmag: q.GetZ()}
		unit := spatialmath.Quaternion(spatialmath.Normalize(raw.Quaternion()))
		o = &unit
	case *commonpb.LinkPose_OrientationVector:
		ov := v.OrientationVector
		ovd := &spatialmath.OrientationVectorDegrees{Theta: ov.GetTheta(), OX: ov.GetOX(), OY: ov.GetOY(), OZ: ov.GetOZ()}
		if err := ovd.IsValid(); err != nil {
			return nil, errors.Wrapf(err, "trajex/rdk: link %q of model %q", link.GetId(), label)
		}
		o = ovd
	case *commonpb.LinkPose_EulerAngles:
		e := v.EulerAngles
		o = &spatialmath.EulerAngles{Roll: radians(e.GetRoll()), Pitch: radians(e.GetPitch()), Yaw: radians(e.GetYaw())}
	case *commonpb.LinkPose_AxisAngles:
		aa := v.AxisAngles
		if zeroLength(aa.GetRx(), aa.GetRy(), aa.GetRz()) {
			return nil, errors.Errorf("trajex/rdk: link %q of model %q turns about a zero axis", link.GetId(), label)
		}
		o = &spatialmath.R4AA{Theta: radians(aa.GetTheta()), RX: aa.GetRx(), RY: aa.GetRy(), RZ: aa.GetRz()}
	default:
		return nil, errors.Errorf("trajex/rdk: link %q of model %q has an orientation of type %T, which this reader does not know", link.GetId(), label, v)
	}
	return spatialmath.NewPose(point, o), nil
}

func radians(degrees float64) float64 {
	return degrees * math.Pi / 180
}

// components splits a pose into the translation and the w, x, y, z quaternion
// that trajex takes as a joint's origin.
func components(p spatialmath.Pose) ([3]float64, [4]float64) {
	t := p.Point()
	q := p.Orientation().Quaternion()
	return [3]float64{t.X, t.Y, t.Z}, [4]float64{q.Real, q.Imag, q.Jmag, q.Kmag}
}
