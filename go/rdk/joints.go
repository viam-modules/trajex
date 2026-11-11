//go:build !windows && cgo

// Package rdk turns RDK's kinematic models into trajex joints. It is the one
// place in this module that knows what an RDK model is; everything else
// describes kinematic chains. Lengths are in millimetres and angles in
// radians, as in RDK.
package rdk

import (
	"slices"
	"strings"

	"github.com/golang/geo/r3"
	"github.com/pkg/errors"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"

	"github.com/viam-modules/trajex/go/kinematics"
)

// JointsFromModel describes the serial chain from the base of m out to a tool
// center point, which is where a TCP speed cap applies. The result is meant
// to be passed to kinematics.NewChain, and always ends in a fixed joint.
//
// tcp is the pose of the tool center point relative to the model's output
// frame, the frame whose pose m.Transform reports. Where that point lies
// depends on what is mounted on the arm, which only the caller knows, so it
// must always be given. A caller capping the end of the arm itself passes
// spatialmath.NewZeroPose().
//
// The translation in tcp is in millimetres, like the model, and a TCP speed
// cap for the resulting chain is in millimetres per second, so a cap held in
// metres per second, such as MaxTCPSpeedMPerSec, must be multiplied by 1000.
//
// JointsFromModel only maps geometry. The model's joint limits and TCP cap
// are not carried over, since how they combine with what a particular call
// asks for is the caller's policy. The limits are m.DoF(), in the order the
// chain's moving joints take their positions.
//
// It returns an error for a model trajex cannot drive with the positions RDK
// drives it with: one with a joint on the path that follows another joint,
// one that takes a position for a joint off the path, such as a gripper's
// finger, and one with a frame that moves but is neither a revolute nor a
// prismatic joint.
func JointsFromModel(m referenceframe.Model, tcp spatialmath.Pose) ([]kinematics.Joint, error) {
	if m == nil {
		return nil, errors.New("trajex/rdk: model is nil")
	}
	if tcp == nil {
		return nil, errors.New("trajex/rdk: tcp pose is nil; pass spatialmath.NewZeroPose() to cap the end of the arm itself")
	}
	chain, err := m.Chain("")
	if err != nil {
		return nil, errors.Wrapf(err, "trajex/rdk: model %q", m.Name())
	}
	b := &chainBuilder{origin: spatialmath.NewZeroPose()}
	if err := b.appendChain(m, chain); err != nil {
		return nil, err
	}
	b.fixed(tcp)
	return b.finish(), nil
}

// JointsFromFrameSystemConfig describes the serial chain from the world frame
// out to the part named tip, for a caller holding a robot's FrameSystemConfig
// reply. Each part on the path contributes the fixed transform that places it
// on its parent, then the joints of its model as JointsFromModel builds them,
// so positions run from the part nearest world out to tip. Parts off the path
// contribute nothing and take no positions.
//
// A parent is world, another part of the reply, or a frame inside another
// part's model written the way RDK names it, part:frame, in which case that
// part's model is followed to the frame rather than to its output frame. The
// parts are read as RDK reads them, so a server older than
// FrameSystemConfig.model, which sends a model in the deprecated kinematics
// field only, is read as well.
func JointsFromFrameSystemConfig(parts []*robotpb.FrameSystemConfig, tip string) ([]kinematics.Joint, error) {
	if tip == "" {
		return nil, errors.New("trajex/rdk: the tip of the chain is not named")
	}
	byName := map[string]*referenceframe.FrameSystemPart{}
	for i, pb := range parts {
		frame := pb.GetFrame()
		name := frame.GetReferenceFrame()
		if name == "" {
			return nil, errors.Errorf("trajex/rdk: part %d of the frame system has no name", i)
		}
		if frame.GetPoseInObserverFrame().GetReferenceFrame() == "" {
			return nil, errors.Errorf("trajex/rdk: part %d of the frame system, %q, has no parent", i, name)
		}
		// RDK reads the pose without a nil check, so refuse a missing one rather than let it panic
		if frame.GetPoseInObserverFrame().GetPose() == nil {
			return nil, errors.Errorf("trajex/rdk: part %d of the frame system has no pose", i)
		}
		if _, dup := byName[name]; dup {
			return nil, errors.Errorf("trajex/rdk: the frame system names part %q more than once", name)
		}
		part, err := referenceframe.ProtobufToFrameSystemPart(pb)
		if err != nil {
			return nil, errors.Wrapf(err, "trajex/rdk: part %q", name)
		}
		byName[name] = part
	}
	if _, ok := byName[tip]; !ok {
		return nil, errors.Errorf("trajex/rdk: the frame system has no part %q", tip)
	}

	// a step is a part on the path and the frame inside its model that the next part hangs
	// from, empty when the next part hangs from the part itself
	type step struct {
		part  *referenceframe.FrameSystemPart
		frame string
	}
	var path []step
	seen := map[string]bool{}
	for name, child := tip, ""; name != referenceframe.World; {
		owner, frame, _ := strings.Cut(name, ":")
		part, ok := byName[owner]
		if !ok {
			return nil, errors.Errorf("trajex/rdk: part %q hangs from %q, which is not a part of the frame system", child, name)
		}
		if frame != "" && part.ModelFrame == nil {
			return nil, errors.Errorf("trajex/rdk: part %q hangs from frame %q of part %q, which has no model", child, frame, owner)
		}
		if seen[owner] {
			return nil, errors.Errorf("trajex/rdk: the path from %q to the world frame is a cycle", tip)
		}
		seen[owner] = true
		path = append(path, step{part, frame})
		name, child = part.FrameConfig.Parent(), owner
	}
	slices.Reverse(path)

	b := &chainBuilder{origin: spatialmath.NewZeroPose()}
	for _, s := range path {
		b.fixed(s.part.FrameConfig.Pose())
		model := s.part.ModelFrame
		if model == nil {
			continue
		}
		chain, err := model.Chain(s.frame)
		if err != nil {
			return nil, errors.Wrapf(err, "trajex/rdk: part %q", s.part.FrameConfig.Name())
		}
		if err := b.appendChain(model, chain); err != nil {
			return nil, err
		}
	}
	return b.finish(), nil
}

// chainBuilder accumulates joints from the base outward. trajex attaches the
// fixed transform ahead of a joint to that joint as its origin, whereas RDK
// models fixed transforms as frames of their own, so the static frames
// between two joints are composed and handed to the second as its origin.
type chainBuilder struct {
	joints []kinematics.Joint
	origin spatialmath.Pose
}

func (b *chainBuilder) fixed(p spatialmath.Pose) {
	b.origin = spatialmath.Compose(b.origin, p)
}

func (b *chainBuilder) moving(make func(xyz [3]float64, wxyz [4]float64, axis [3]float64) kinematics.Joint, axis r3.Vector) {
	xyz, wxyz := components(b.origin)
	b.joints = append(b.joints, make(xyz, wxyz, [3]float64{axis.X, axis.Y, axis.Z}))
	b.origin = spatialmath.NewZeroPose()
}

// finish ends the chain with whatever fixed transform follows the last joint.
// The joint is added even when that transform is the identity, since it costs
// nothing and gives every chain the same shape.
func (b *chainBuilder) finish() []kinematics.Joint {
	xyz, wxyz := components(b.origin)
	return append(b.joints, kinematics.FixedJoint(xyz, wxyz))
}

// appendChain adds the frames of a chain the model returned. RDK drives the
// model with one position per input, in the model's order, and a caller
// hands trajex those same positions, so the moving joints on the chain have
// to be exactly the model's inputs in order or the positions would drive the
// wrong joints. A mimic joint has no position of its own and trajex cannot
// derive one, so it is refused too.
func (b *chainBuilder) appendChain(model referenceframe.Model, chain []referenceframe.ChainFrame) error {
	var names []string
	var offsets []int
	for _, cf := range chain {
		name := cf.Frame.Name()
		dof := len(cf.Frame.DoF())
		if dof == 0 {
			pose, err := cf.Frame.Transform([]referenceframe.Input{})
			if err != nil {
				return errors.Wrapf(err, "trajex/rdk: frame %q of model %q", name, model.Name())
			}
			b.fixed(pose)
			continue
		}
		if cf.Mimic != nil {
			return errors.Errorf("trajex/rdk: joint %q of model %q follows another joint, which trajex cannot express", name, model.Name())
		}
		switch f := cf.Frame.(type) {
		case referenceframe.RevoluteFrame:
			b.moving(kinematics.RotationalJoint, f.RotationAxis())
		case referenceframe.PrismaticFrame:
			b.moving(kinematics.TranslationalJoint, f.TranslationAxis())
		default:
			return errors.Errorf(
				"trajex/rdk: frame %q of model %q takes %d positions and is neither a revolute nor a prismatic joint, which trajex cannot express",
				name, model.Name(), dof)
		}
		names = append(names, name)
		offsets = append(offsets, cf.Offset)
	}
	inOrder := len(offsets) == len(model.DoF())
	for i, offset := range offsets {
		inOrder = inOrder && offset == i
	}
	if !inOrder {
		return errors.Errorf(
			"trajex/rdk: model %q takes %d positions, but the path to %q moves joints %v at positions %v, so positions meant for the model would drive the wrong joints",
			model.Name(), len(model.DoF()), chain[len(chain)-1].Frame.Name(), names, offsets)
	}
	return nil
}

// components splits a pose into the translation and w, x, y, z quaternion
// that trajex takes as a joint's origin.
func components(p spatialmath.Pose) ([3]float64, [4]float64) {
	t := p.Point()
	q := p.Orientation().Quaternion()
	return [3]float64{t.X, t.Y, t.Z}, [4]float64{q.Real, q.Imag, q.Jmag, q.Kmag}
}
