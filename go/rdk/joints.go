//go:build !windows && cgo

// Package rdk builds trajex kinematic chains from the KinematicModel and
// FrameSystemConfig messages of the Viam API. The messages are parsed by
// RDK's referenceframe package, so a chain built here agrees with the model
// RDK itself drives. Units follow RDK: millimetres and radians.
package rdk

import (
	"slices"

	"github.com/pkg/errors"
	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"

	"github.com/viam-modules/trajex/go/kinematics"
)

// JointsFromKinematicModel describes the serial chain from the base of model
// out to outputFrame, for kinematics.NewChain. The chain always ends in a
// fixed joint.
//
// outputFrame names a link or joint of the model, or is empty for the model's
// own output frame. The chain takes positions in the order RDK's model does,
// so the moving joints on the path to outputFrame must be exactly the model's
// moving joints. A mimic joint on the path, a moving joint off it such as a
// gripper finger, and anything RDK refuses to load are errors.
//
// Joint limits are not part of the chain. They are the DoF of the model that
// referenceframe.ModelFromProto builds from the same message, in the same
// order.
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
		link  *referenceframe.LinkInFrame
		model *commonpb.KinematicModel
		// legacy is set when the model came in the deprecated kinematics field,
		// which holds RDK's file format rather than a KinematicModel
		legacy bool
	}
	byName := map[string]placed{}
	for i, part := range parts {
		// rdk reads the pose without a nil check, so refuse a missing one rather than let it panic
		if part.GetFrame().GetPoseInObserverFrame().GetPose() == nil {
			return nil, errors.Errorf("trajex/rdk: part %d of the frame system has no pose", i)
		}
		link, err := referenceframe.LinkInFrameFromTransformProtobuf(part.GetFrame())
		if err != nil {
			return nil, errors.Wrapf(err, "trajex/rdk: part %d of the frame system", i)
		}
		if _, dup := byName[link.Name()]; dup {
			return nil, errors.Errorf("trajex/rdk: the frame system names part %q more than once", link.Name())
		}
		// an empty model is no model, which is how RDK reads it too
		model := part.GetModel()
		if len(model.GetLinks())+len(model.GetJoints()) == 0 {
			model = nil
		}
		byName[link.Name()] = placed{
			link:   link,
			model:  model,
			legacy: model == nil && len(part.GetKinematics().GetFields()) > 0,
		}
	}
	if _, ok := byName[tip]; !ok {
		return nil, errors.Errorf("trajex/rdk: the frame system has no part %q", tip)
	}

	var path []placed
	seen := map[string]bool{}
	for name, child := tip, ""; name != referenceframe.World; {
		part, ok := byName[name]
		if !ok {
			return nil, errors.Errorf("trajex/rdk: part %q hangs from %q, which is not a part of the frame system", child, name)
		}
		if seen[name] {
			return nil, errors.Errorf("trajex/rdk: the path from %q to the world frame is a cycle", tip)
		}
		seen[name] = true
		path = append(path, part)
		name, child = part.link.Parent(), name
	}
	slices.Reverse(path)

	b := &chainBuilder{origin: spatialmath.NewZeroPose()}
	for _, part := range path {
		b.fixed(part.link.Pose())
		if part.legacy {
			return nil, errors.Errorf(
				"trajex/rdk: part %q carries its model in the deprecated kinematics field only, which a server older than FrameSystemConfig.model sends and this reader does not read",
				part.link.Name())
		}
		if part.model == nil {
			continue
		}
		label := part.model.GetName()
		if label == "" {
			label = part.link.Name()
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
	if err := checkMessage(model, label); err != nil {
		return err
	}
	loaded, err := referenceframe.ModelFromProto(model, label)
	if err != nil {
		return errors.Wrapf(err, "trajex/rdk: model %q", label)
	}
	// only a SimpleModel exposes the parsed config and the order it takes inputs in
	sm, ok := loaded.(*referenceframe.SimpleModel)
	if !ok {
		return errors.Errorf("trajex/rdk: model %q loaded as a %T, which does not report its moving joints", label, loaded)
	}
	cfg := sm.ModelConfig()

	elements := index(cfg)
	output, err := outputFrameOf(cfg, elements, label, outputFrame)
	if err != nil {
		return err
	}
	path, err := pathToRoot(elements, output)
	if err != nil {
		return err
	}

	var moving []string
	for _, id := range path {
		e := elements[id]
		if e.link != nil {
			pose, err := e.link.Pose()
			if err != nil {
				return errors.Wrapf(err, "trajex/rdk: link %q of model %q", id, label)
			}
			b.fixed(pose)
			continue
		}
		if e.joint.Mimic != nil {
			return errors.Errorf("trajex/rdk: joint %q of model %q mimics another joint, which trajex cannot express", id, label)
		}
		axis := [3]float64{e.joint.Axis.X, e.joint.Axis.Y, e.joint.Axis.Z}
		switch e.joint.Type {
		case referenceframe.RevoluteJoint:
			b.rotational(axis)
		case referenceframe.PrismaticJoint:
			b.translational(axis)
		default:
			return errors.Errorf("trajex/rdk: joint %q of model %q has type %q, which trajex does not support", id, label, e.joint.Type)
		}
		moving = append(moving, id)
	}

	// the chain is indexed like the model's inputs, so the path must move exactly those joints in that order
	if want := sm.MoveableFrameNames(); !slices.Equal(moving, want) {
		return errors.Errorf(
			"trajex/rdk: model %q takes inputs for joints %v, but the path to its output frame %q moves joints %v",
			label, want, output, moving)
	}
	return nil
}

// element is one link or joint of the parsed model. Exactly one of link and
// joint is set.
type element struct {
	parent string
	link   *referenceframe.LinkConfig
	joint  *referenceframe.JointConfig
}

// checkMessage catches what RDK does not report well: an empty model, an
// element with no id, a duplicate id, which RDK silently merges, and a zero
// axis on a joint or in an axis angle orientation, which makes RDK panic.
func checkMessage(model *commonpb.KinematicModel, label string) error {
	if len(model.GetLinks()) == 0 && len(model.GetJoints()) == 0 {
		return errors.Errorf("trajex/rdk: model %q has no links or joints", label)
	}
	ids := map[string]bool{}
	add := func(id string) error {
		if id == "" {
			return errors.Errorf("trajex/rdk: model %q has a link or joint with no id", label)
		}
		if ids[id] {
			return errors.Errorf("trajex/rdk: model %q names %q more than once", label, id)
		}
		ids[id] = true
		return nil
	}
	for _, link := range model.GetLinks() {
		if err := add(link.GetId()); err != nil {
			return err
		}
		if aa, ok := link.GetPose().GetOrientation().(*commonpb.LinkPose_AxisAngles); ok {
			if a := aa.AxisAngles; zeroLength(a.GetRx(), a.GetRy(), a.GetRz()) {
				return errors.Errorf("trajex/rdk: link %q of model %q turns about a zero axis", link.GetId(), label)
			}
		}
	}
	for _, joint := range model.GetJoints() {
		if err := add(joint.GetId()); err != nil {
			return err
		}
		if axis := joint.GetAxis(); zeroLength(axis.GetX(), axis.GetY(), axis.GetZ()) {
			return errors.Errorf("trajex/rdk: joint %q of model %q has no axis", joint.GetId(), label)
		}
	}
	return nil
}

// zeroLength matches the test RDK's normalization applies, so a vector whose
// squared length underflows to zero is zero here too.
func zeroLength(x, y, z float64) bool {
	return x*x+y*y+z*z == 0
}

func index(cfg *referenceframe.ModelConfigJSON) map[string]*element {
	elements := map[string]*element{}
	for i := range cfg.Links {
		link := &cfg.Links[i]
		elements[link.ID] = &element{parent: link.Parent, link: link}
	}
	for i := range cfg.Joints {
		joint := &cfg.Joints[i]
		elements[joint.ID] = &element{parent: joint.Parent, joint: joint}
	}
	return elements
}

// outputFrameOf picks the frame the chain ends at: the caller's if named,
// else the model's declared output frame, else its single leaf. RDK has
// already refused models with more than one leaf, so only the caller's name
// can fail here.
func outputFrameOf(cfg *referenceframe.ModelConfigJSON, elements map[string]*element, label, requested string) (string, error) {
	if requested != "" {
		if _, ok := elements[requested]; !ok {
			return "", errors.Errorf("trajex/rdk: output frame %q is not a link or joint of model %q", requested, label)
		}
		return requested, nil
	}
	if len(cfg.OutputFrames) > 0 {
		return cfg.OutputFrames[0], nil
	}
	hasChild := map[string]bool{}
	for _, e := range elements {
		hasChild[e.parent] = true
	}
	for id := range elements {
		if !hasChild[id] {
			return id, nil
		}
	}
	return "", errors.Errorf("trajex/rdk: model %q has no leaf", label)
}

// pathToRoot lists the elements from the model's root, the first whose parent
// is not an element, down to output. RDK has already refused cycles, so the
// bound only guards against a bug.
func pathToRoot(elements map[string]*element, output string) ([]string, error) {
	var path []string
	id := output
	for range len(elements) + 1 {
		e, ok := elements[id]
		if !ok {
			slices.Reverse(path)
			return path, nil
		}
		path = append(path, id)
		id = e.parent
	}
	return nil, errors.Errorf("trajex/rdk: the path from %q to the root of the model is a cycle", output)
}

// components splits a pose into the translation and the w, x, y, z quaternion
// that trajex takes as a joint's origin.
func components(p spatialmath.Pose) ([3]float64, [4]float64) {
	t := p.Point()
	q := p.Orientation().Quaternion()
	return [3]float64{t.X, t.Y, t.Z}, [4]float64{q.Real, q.Imag, q.Jmag, q.Kmag}
}
