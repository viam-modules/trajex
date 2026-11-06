//go:build !windows && cgo

package rdk

import (
	"slices"

	"github.com/pkg/errors"
	commonpb "go.viam.com/api/common/v1"
	robotpb "go.viam.com/api/robot/v1"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/kinematics"
	"github.com/viam-modules/trajex/go/totg"
)

// Arm is what a KinematicModel gives trajex: the chain out to the capped
// frame, and the limits and properties the message carries.
type Arm struct {
	// Joints is the chain JointsFromKinematicModel gives.
	Joints []kinematics.Joint
	// Limits has one entry per position the chain takes, in that order, as
	// RDK holds them: radians or millimetres, an absent position bound
	// infinite, an absent or infinite velocity or acceleration bound nil.
	Limits []referenceframe.Limit
	// OperatingFreqHz is the rate the arm expects to be driven at, and
	// MaxTCPSpeedMMPerSec the speed its tool center point may not exceed.
	// Either is zero when the message carries none.
	OperatingFreqHz     float64
	MaxTCPSpeedMMPerSec float64
}

// ArmFromKinematicModel builds the chain JointsFromKinematicModel builds and
// keeps the limits and properties beside it, for a caller feeding TOTG from
// the message alone. A negative property is refused.
func ArmFromKinematicModel(model *commonpb.KinematicModel, outputFrame string) (*Arm, error) {
	if model == nil {
		return nil, errors.New("trajex/rdk: kinematic model is nil")
	}
	sm, err := load(model, model.GetName())
	if err != nil {
		return nil, err
	}
	b := &chainBuilder{origin: spatialmath.NewZeroPose()}
	if err := b.appendLoaded(sm, model.GetName(), outputFrame); err != nil {
		return nil, err
	}
	arm := &Arm{Joints: b.finish(), Limits: slices.Clone(sm.DoF())}
	if err := arm.setProperties(model, model.GetName()); err != nil {
		return nil, err
	}
	return arm, nil
}

// ArmFromModel is ArmFromKinematicModel for the message RDK serves for m,
// with Limits taken from m itself, since the message carries degrees and
// gives back radians that differ in the last place. It returns an error for
// a model RDK cannot write out as a message.
func ArmFromModel(m referenceframe.Model, outputFrame string) (*Arm, error) {
	if m == nil {
		return nil, errors.New("trajex/rdk: model is nil")
	}
	pb, err := referenceframe.ModelToProto(m)
	if err != nil {
		return nil, errors.Wrap(err, "trajex/rdk")
	}
	arm, err := ArmFromKinematicModel(pb, outputFrame)
	if err != nil {
		return nil, err
	}
	arm.Limits = slices.Clone(m.DoF())
	return arm, nil
}

// ArmFromFrameSystemConfig is JointsFromFrameSystemConfig with the limits of
// every part on the path, in the order the chain takes positions, and the
// properties of the part named tip. A part with no model contributes no
// limits, and a tip with no model leaves the properties zero.
func ArmFromFrameSystemConfig(parts []*robotpb.FrameSystemConfig, tip string) (*Arm, error) {
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

	arm := &Arm{}
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
		sm, err := load(part.model, label)
		if err != nil {
			return nil, err
		}
		if err := b.appendLoaded(sm, label, ""); err != nil {
			return nil, err
		}
		arm.Limits = append(arm.Limits, sm.DoF()...)
		if part.link.Name() == tip {
			if err := arm.setProperties(part.model, label); err != nil {
				return nil, err
			}
		}
	}
	arm.Joints = b.finish()
	return arm, nil
}

// setProperties reads the model's properties, refusing a negative one, which
// Inputs would otherwise drop without a word.
func (a *Arm) setProperties(model *commonpb.KinematicModel, label string) error {
	p := model.GetProperties()
	a.OperatingFreqHz, a.MaxTCPSpeedMMPerSec = p.GetOperatingFreqHz(), p.GetMaxLinearTcpSpeedMmPerSec()
	if a.OperatingFreqHz < 0 {
		return errors.Errorf("trajex/rdk: model %q has a negative operating frequency %v", label, a.OperatingFreqHz)
	}
	if a.MaxTCPSpeedMMPerSec < 0 {
		return errors.Errorf("trajex/rdk: model %q has a negative TCP speed cap %v", label, a.MaxTCPSpeedMMPerSec)
	}
	return nil
}

// Inputs writes the TOTG inputs the model carries into in: the velocity and
// acceleration limits when every joint has one, and the sampling frequency
// and TCP speed cap when the properties name them. The caller sets whatever
// the model lacks, and the waypoints are always the caller's. The cap takes
// effect through totg.GenerateWithKinematics with the chain. A bound of zero
// is refused, since TOTG fails on it without naming the joint.
func (a *Arm) Inputs(in *trajex.TensorMap) error {
	dof := []uint64{uint64(len(a.Limits))}
	for _, key := range []struct {
		name  string
		key   string
		bound func(referenceframe.Limit) *float64
	}{
		{"velocity", totg.KeyVelocityLimitsRadsPerSec, func(l referenceframe.Limit) *float64 { return l.MaxVelocity }},
		{"acceleration", totg.KeyAccelerationLimitsRadsPerSec2, func(l referenceframe.Limit) *float64 { return l.MaxAcceleration }},
	} {
		values, ok, err := bounds(a.Limits, key.name, key.bound)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		if err := in.InsertFloat64s(key.key, dof, values); err != nil {
			return err
		}
	}
	if a.OperatingFreqHz > 0 {
		if err := in.InsertScalarFloat64(totg.KeyTrajectorySamplingFreqHz, a.OperatingFreqHz); err != nil {
			return err
		}
	}
	if a.MaxTCPSpeedMMPerSec > 0 {
		if err := in.InsertScalarFloat64(totg.KeyTCPMaxLinearSpeed, a.MaxTCPSpeedMMPerSec); err != nil {
			return err
		}
	}
	return nil
}

// bounds collects one bound per joint, or reports that some joint has none,
// since TOTG takes a limit for every joint or for none.
func bounds(limits []referenceframe.Limit, name string, bound func(referenceframe.Limit) *float64) ([]float64, bool, error) {
	if len(limits) == 0 {
		return nil, false, nil
	}
	out := make([]float64, 0, len(limits))
	for i, l := range limits {
		b := bound(l)
		if b == nil {
			return nil, false, nil
		}
		if *b <= 0 {
			return nil, false, errors.Errorf("trajex/rdk: joint %d has a %s bound of %v, which TOTG cannot plan with", i, name, *b)
		}
		out = append(out, *b)
	}
	return out, true, nil
}
