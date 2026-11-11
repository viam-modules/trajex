//go:build !windows && cgo

// Package rdk turns RDK's kinematic models into trajex joints. It is the one
// place in this module that knows what an RDK model is; everything else
// describes kinematic chains. Lengths are in millimetres and angles in
// radians, as in RDK.
package rdk

import (
	"github.com/golang/geo/r3"
	"github.com/pkg/errors"
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
// A model built over a frame system of parts, which referenceframe.NewModel
// gives, works the same way, since its chain expands the parts. Its inputs
// are every part's, so a part off the path to the output frame makes the
// model one trajex cannot drive, as a gripper's finger does.
//
// It returns an error for a model trajex cannot drive with the positions RDK
// drives it with: one with a joint on the path that follows another joint,
// one that takes a position for a joint off the path, and one with a frame
// that moves but is neither a revolute nor a prismatic joint.
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

	// trajex attaches the fixed transform ahead of a joint to that joint as its
	// origin, whereas RDK models fixed transforms as frames of their own, so the
	// static frames between two joints are composed and handed to the second
	var joints []kinematics.Joint
	var names []string
	var offsets []int
	origin := spatialmath.NewZeroPose()
	moving := func(make func(xyz [3]float64, wxyz [4]float64, axis [3]float64) kinematics.Joint, axis r3.Vector) {
		xyz, wxyz := components(origin)
		joints = append(joints, make(xyz, wxyz, [3]float64{axis.X, axis.Y, axis.Z}))
		origin = spatialmath.NewZeroPose()
	}
	for _, cf := range chain {
		name := cf.Frame.Name()
		dof := len(cf.Frame.DoF())
		if dof == 0 {
			pose, err := cf.Frame.Transform([]referenceframe.Input{})
			if err != nil {
				return nil, errors.Wrapf(err, "trajex/rdk: frame %q of model %q", name, m.Name())
			}
			origin = spatialmath.Compose(origin, pose)
			continue
		}
		if cf.Mimic != nil {
			return nil, errors.Errorf("trajex/rdk: joint %q of model %q follows another joint, which trajex cannot express", name, m.Name())
		}
		switch f := cf.Frame.(type) {
		case referenceframe.RevoluteFrame:
			moving(kinematics.RotationalJoint, f.RotationAxis())
		case referenceframe.PrismaticFrame:
			moving(kinematics.TranslationalJoint, f.TranslationAxis())
		default:
			return nil, errors.Errorf(
				"trajex/rdk: frame %q of model %q takes %d positions and is neither a revolute nor a prismatic joint, which trajex cannot express",
				name, m.Name(), dof)
		}
		names = append(names, name)
		offsets = append(offsets, cf.Offset)
	}

	// RDK drives the model with one position per input, in the model's order,
	// and a caller hands trajex those same positions, so the moving joints on
	// the chain have to be exactly the model's inputs in order or the positions
	// would drive the wrong joints
	inOrder := len(offsets) == len(m.DoF())
	for i, offset := range offsets {
		inOrder = inOrder && offset == i
	}
	if !inOrder {
		return nil, errors.Errorf(
			"trajex/rdk: model %q takes %d positions, but the path to its output frame moves joints %v at positions %v, so positions meant for the model would drive the wrong joints",
			m.Name(), len(m.DoF()), names, offsets)
	}

	// the tool center point extends whatever follows the last joint, and the two
	// together become the chain's final fixed joint. It is added even when both
	// are the identity, since it costs nothing and gives every chain the same shape
	xyz, wxyz := components(spatialmath.Compose(origin, tcp))
	return append(joints, kinematics.FixedJoint(xyz, wxyz)), nil
}

// components splits a pose into the translation and w, x, y, z quaternion
// that trajex takes as a joint's origin.
func components(p spatialmath.Pose) ([3]float64, [4]float64) {
	t := p.Point()
	q := p.Orientation().Quaternion()
	return [3]float64{t.X, t.Y, t.Z}, [4]float64{q.Real, q.Imag, q.Jmag, q.Kmag}
}
