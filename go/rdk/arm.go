//go:build !windows && cgo

package rdk

import (
	"slices"

	"github.com/pkg/errors"
	commonpb "go.viam.com/api/common/v1"
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
	// infinite, an absent velocity or acceleration bound nil.
	Limits []referenceframe.Limit
	// OperatingHz is the rate the arm expects to be driven at, and
	// MaxTCPSpeedMMPerSec the speed its tool center point may not exceed.
	// Either is zero when the message carries none.
	OperatingHz         float64
	MaxTCPSpeedMMPerSec float64
}

// ArmFromKinematicModel builds the chain JointsFromKinematicModel builds and
// keeps the limits and properties beside it, for a caller feeding TOTG from
// the message alone.
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
	p := model.GetProperties()
	return &Arm{
		Joints:              b.finish(),
		Limits:              slices.Clone(sm.DoF()),
		OperatingHz:         p.GetOperatingFreqHz(),
		MaxTCPSpeedMMPerSec: p.GetMaxLinearTcpSpeedMmPerSec(),
	}, nil
}

// ArmFromModel is ArmFromKinematicModel for the message RDK serves for m. It
// returns an error for a model RDK cannot write out as a message.
func ArmFromModel(m referenceframe.Model, outputFrame string) (*Arm, error) {
	if m == nil {
		return nil, errors.New("trajex/rdk: model is nil")
	}
	pb, err := referenceframe.ModelToProto(m)
	if err != nil {
		return nil, errors.Wrap(err, "trajex/rdk")
	}
	return ArmFromKinematicModel(pb, outputFrame)
}

// Inputs writes the TOTG inputs the model carries into in: the velocity and
// acceleration limits when every joint has one, and the sampling frequency
// and TCP speed cap when the properties name them. The caller sets whatever
// the model lacks, and the waypoints are always the caller's. The cap takes
// effect through totg.GenerateWithKinematics with the chain.
func (a *Arm) Inputs(in *trajex.TensorMap) error {
	dof := []uint64{uint64(len(a.Limits))}
	if v, ok := bounds(a.Limits, func(l referenceframe.Limit) *float64 { return l.MaxVelocity }); ok {
		if err := in.InsertFloat64s(totg.KeyVelocityLimitsRadsPerSec, dof, v); err != nil {
			return err
		}
	}
	if v, ok := bounds(a.Limits, func(l referenceframe.Limit) *float64 { return l.MaxAcceleration }); ok {
		if err := in.InsertFloat64s(totg.KeyAccelerationLimitsRadsPerSec2, dof, v); err != nil {
			return err
		}
	}
	if a.OperatingHz > 0 {
		if err := in.InsertScalarFloat64(totg.KeyTrajectorySamplingFreqHz, a.OperatingHz); err != nil {
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
func bounds(limits []referenceframe.Limit, bound func(referenceframe.Limit) *float64) ([]float64, bool) {
	if len(limits) == 0 {
		return nil, false
	}
	out := make([]float64, 0, len(limits))
	for _, l := range limits {
		b := bound(l)
		if b == nil {
			return nil, false
		}
		out = append(out, *b)
	}
	return out, true
}
