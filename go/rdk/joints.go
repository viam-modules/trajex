//go:build !windows && cgo

// Package rdk turns RDK's kinematic models into trajex joints. It is the one
// place in this module that knows what an RDK model is; everything else
// describes kinematic chains or reads the API's messages.
package rdk

import (
	"github.com/pkg/errors"
	"go.viam.com/rdk/referenceframe"

	"github.com/viam-modules/trajex/go/kinematics"
	"github.com/viam-modules/trajex/go/viamapi"
)

// JointsFromModel describes the serial chain from the base of m out to
// outputFrame, which is where a TCP speed cap applies. It gives the chain
// viamapi.JointsFromKinematicModel gives for the message RDK serves for m,
// and that function says what outputFrame means, the units, and which models
// are refused.
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
	return viamapi.JointsFromKinematicModel(pb, outputFrame)
}
