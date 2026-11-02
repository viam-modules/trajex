//go:build !windows && cgo

// Package rdk turns the kinematic messages of the Viam API, KinematicModel
// and FrameSystemConfig, and RDK's own models into trajex joints. It is the
// one place in the chain building of this module that knows what an RDK
// model is. RDK loads each message here the way it loads one it receives
// itself, so a chain built from a message never reads the message
// differently from the model RDK drives.
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

// JointsFromModel describes the serial chain from the base of m out to
// outputFrame, which is where a TCP speed cap applies. It gives the chain
// JointsFromKinematicModel gives for the message RDK serves for m, and that
// function says what outputFrame means, the units, and which models are
// refused.
//
// An empty outputFrame caps the model's own output frame. A caller whose
// model has no link for the point it wants capped, such as a tool on the
// flange of an arm that came from the robot's config, appends a
// kinematics.FixedJoint carrying the tool's offset to the result.
//
// It returns an error for a model the pinned RDK release cannot write out as
// a message, which is one built in code rather than loaded from a file.
func JointsFromModel(m referenceframe.Model, outputFrame string) ([]kinematics.Joint, error) {
	if m == nil {
		return nil, errors.New("trajex/rdk: model is nil")
	}
	pb, err := referenceframe.ModelToProto(m)
	if err != nil {
		return nil, errors.Wrap(err, "trajex/rdk")
	}
	return JointsFromKinematicModel(pb, outputFrame)
}

// JointsFromKinematicModel describes the serial chain from the base of model
// out to outputFrame, which is where a TCP speed cap applies. The result is
// meant for kinematics.NewChain and always ends in a fixed joint, so a chain
// has the same shape whether or not links follow the last moving joint.
//
// RDK loads the message first, so a model RDK refuses, one with a cycle,
// with several output frames, with no output frame and other than one leaf,
// or with a mimic joint that carries limits of its own, is refused here with
// RDK's reason. outputFrame names a link or joint of the model. When it is
// empty the model's own output frame is used, which is the first of its
// output_frames or else its single leaf. A caller capping a point past that
// frame, such as the tip of a tool on a flange, names the link that stands
// for the point, or appends a kinematics.FixedJoint to the result. It
// returns an error if outputFrame names no link or joint of model.
//
// Lengths are in millimetres, as they are in the message, so prismatic joint
// positions and a TCP speed cap for the chain are in millimetres as well.
// Only geometry is mapped. The limits and properties the message carries are
// left to the caller's policy.
//
// RDK drives a model with one position for every moving joint in the whole
// model, in its own order, and a caller hands trajex those same positions. So
// the moving joints on the path to outputFrame must be exactly the model's
// moving joints in that order, and a model that fails that is refused rather
// than given a chain that would read the positions wrongly. That covers a
// mimic joint on the path, which has no position of its own, and a moving
// joint off the path, such as a gripper's finger.
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
// of a machine out to the output frame of the part named tip, through every
// part between. It is for a caller holding a machine's FrameSystemConfig
// reply rather than one component's model, such as one capping the speed of a
// gripper carried by an arm that itself rides a gantry.
//
// Each part contributes the fixed transform that places its frame on its
// parent, then the joints of its model, built as JointsFromKinematicModel
// builds them with the model's own output frame. A part with no model is a
// fixed transform alone. Positions for the chain are the positions of each
// part's model in turn, from the part nearest world out to tip, an order of
// this function's own, since a frame system keys positions by part rather
// than listing them. Parts off that path contribute nothing and take no
// positions.
//
// A part whose parent is world hangs from the root. Any other parent has to
// be a part of the same reply. RDK also lets a part hang from a frame inside
// another part's model, or from a part's origin frame, but we walk parts
// alone, so such a parent is refused. A part whose model arrives in the
// deprecated kinematics field only, which is what a server older than
// FrameSystemConfig.model sends, is refused rather than read as a fixed
// transform with no joints.
func JointsFromFrameSystemConfig(parts []*robotpb.FrameSystemConfig, tip string) ([]kinematics.Joint, error) {
	if tip == "" {
		return nil, errors.New("trajex/rdk: the tip of the chain is not named")
	}
	type placed struct {
		link  *referenceframe.LinkInFrame
		model *commonpb.KinematicModel
		// legacy is set when the model came in the deprecated kinematics field,
		// which holds RDK's file format rather than the message
		legacy bool
	}
	byName := map[string]placed{}
	for i, part := range parts {
		// rdk reads the pose without a nil check, so we refuse a missing one rather than let it panic
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
		byName[link.Name()] = placed{
			link:   link,
			model:  part.GetModel(),
			legacy: part.GetModel() == nil && len(part.GetKinematics().GetFields()) > 0,
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

// chainBuilder collects joints base first. A trajex joint carries the static
// transform that precedes its motion as its origin, while RDK gives links as
// static poses of their own, so we compose the links between one joint and
// the next into origin and hand it to the next joint.
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

// appendModel walks model from its root to outputFrame, adding what it finds
// to the chain. label names the model in errors, since a model served inside
// a frame system part may carry no name of its own.
//
// RDK loads the message before we walk it. That is what refuses the shapes
// RDK cannot drive and reads each orientation form as RDK reads it, and it
// leaves us the configuration RDK built from the message, with every link's
// pose parsed and the order RDK takes positions in.
func (b *chainBuilder) appendModel(model *commonpb.KinematicModel, label, outputFrame string) error {
	if err := checkMessage(model, label); err != nil {
		return err
	}
	loaded, err := referenceframe.ModelFromProto(model, label)
	if err != nil {
		return errors.Wrapf(err, "trajex/rdk: model %q", label)
	}
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

	if want := sm.MoveableFrameNames(); !slices.Equal(moving, want) {
		return errors.Errorf(
			"trajex/rdk: model %q takes inputs for joints %v, but the path to its output frame %q moves joints %v",
			label, want, output, moving)
	}
	return nil
}

// element is one link or joint of the configuration RDK built from the
// message. Exactly one of link and joint is set.
type element struct {
	parent string
	link   *referenceframe.LinkConfig
	joint  *referenceframe.JointConfig
}

// checkMessage refuses what RDK would not refuse clearly on its own. RDK
// keeps the last of two elements with the same id, so a chain from such a
// model would be a guess at which was meant. A zero axis on a revolute
// joint, or in a link's axis angle orientation, makes RDK panic while it
// loads the model rather than return an error, and on a prismatic joint it
// gets an error that does not name the joint.
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

// zeroLength is the test RDK's normalization applies, a sum of squares that
// is zero, so components small enough to underflow it count as zero here too.
func zeroLength(x, y, z float64) bool {
	return x*x+y*y+z*z == 0
}

// index maps the links and joints of the configuration RDK built by id.
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

// outputFrameOf picks the frame the chain ends at: the one the caller named,
// else the one the model names, else its single leaf. RDK has already refused
// a model that names several output frames, names one it lacks, or names none
// and has other than one leaf, so only the caller's name can fail here.
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

// pathToRoot lists the elements from the model's root down to output. The
// root is the first element whose parent is not an element, which RDK hangs
// from world. RDK has refused any cycle, so the walk is bounded only to keep
// it finite should that ever change.
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

// components splits a pose into the translation and w, x, y, z quaternion
// that trajex takes as a joint's origin.
func components(p spatialmath.Pose) ([3]float64, [4]float64) {
	t := p.Point()
	q := p.Orientation().Quaternion()
	return [3]float64{t.X, t.Y, t.Z}, [4]float64{q.Real, q.Imag, q.Jmag, q.Kmag}
}
