//go:build !windows && cgo

// Package rdk adapts trajex's TOTG surface to RDK's MLModel service
// interface, so RDK consumers can use trajex as a drop-in MLModel resource
// interchangeable with the existing trajex sidecar service.
package rdk

import (
	"context"

	"github.com/pkg/errors"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
	"gorgonia.org/tensor"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/totg"
)

// Compile-time interface check.
var _ mlmodel.Service = (*Service)(nil)

// Service is an in-process MLModel service backed by trajex TOTG. It
// duck-types into mlmodel.Service without going through RDK's resource
// registry; consumers construct it directly with NewService and pass it to
// whatever motion code expects an mlmodel.Service.
type Service struct {
	resource.Named
	resource.TriviallyReconfigurable
	resource.TriviallyCloseable
}

// NewService returns a Service named per the supplied resource.Name.
func NewService(name resource.Name) *Service {
	return &Service{Named: name.AsNamed()}
}

// Infer adapts an ml.Tensors input map to the trajex CAPI, runs TOTG, and
// returns the trajectory as an ml.Tensors output map. Closes the
// underlying CAPI tensor maps before returning.
func (s *Service) Infer(ctx context.Context, inputs ml.Tensors) (ml.Tensors, error) {
	inMap, err := trajex.NewTensorMap()
	if err != nil {
		return nil, errors.Wrap(err, "trajex/totg/rdk: allocating input map")
	}
	defer inMap.Close()

	outMap, err := trajex.NewTensorMap()
	if err != nil {
		return nil, errors.Wrap(err, "trajex/totg/rdk: allocating output map")
	}
	defer outMap.Close()

	for key, t := range inputs {
		if err := insertTensor(inMap, key, t); err != nil {
			return nil, errors.Wrapf(err, "trajex/totg/rdk: marshaling input %q", key)
		}
	}

	if err := totg.Generate(ctx, inMap, outMap); err != nil {
		return nil, err
	}

	return readOutputs(outMap)
}

// Metadata returns the static TOTG input/output schema. Derived from the
// existing trajex C++ MLModelService implementation, sans the legacy
// generator-sequence and segment-for-trajex inputs (per the umbrella plan;
// trajex handles segmentation internally now).
func (s *Service) Metadata(_ context.Context) (mlmodel.MLMetadata, error) {
	return mlmodel.MLMetadata{
		ModelName:        "trajex-totg",
		ModelType:        "trajectory_generator",
		ModelDescription: "Time-optimal trajectory generation via the Kunz & Stilman TOTG algorithm.",
		Inputs: []mlmodel.TensorInfo{
			{
				Name:        totg.KeyWaypointsRads,
				Description: "Joint configurations to follow, in radians [n_waypoints, n_dof]",
				DataType:    "float64",
				Shape:       []int{-1, -1},
			},
			{
				Name:        totg.KeyVelocityLimitsRadsPerSec,
				Description: "Per-joint maximum velocity, in rad/s [n_dof]",
				DataType:    "float64",
				Shape:       []int{-1},
			},
			{
				Name:        totg.KeyAccelerationLimitsRadsPerSec2,
				Description: "Per-joint maximum acceleration, in rad/s^2 [n_dof]",
				DataType:    "float64",
				Shape:       []int{-1},
			},
			{
				Name:        totg.KeyPathToleranceDeltaRads,
				Description: "Path-blending tolerance, in radians [scalar]",
				DataType:    "float64",
				Shape:       []int{1},
			},
			{
				Name:        totg.KeyPathColinearizationRatio,
				Description: "Path colinearization aggressiveness [scalar]",
				DataType:    "float64",
				Shape:       []int{1},
			},
			{
				Name:        totg.KeyWaypointDeduplicationToleranceRads,
				Description: "Distance below which adjacent waypoints are merged, in radians [scalar]",
				DataType:    "float64",
				Shape:       []int{1},
			},
			{
				Name:        totg.KeyTrajectorySamplingFreqHz,
				Description: "Output sample rate, in Hz [scalar]",
				DataType:    "int64",
				Shape:       []int{1},
			},
		},
		Outputs: []mlmodel.TensorInfo{
			{
				Name:        totg.KeySampleTimesSec,
				Description: "Sample timestamps from t=0, in seconds [n_samples]",
				DataType:    "float64",
				Shape:       []int{-1},
			},
			{
				Name:        totg.KeyConfigurationsRads,
				Description: "Joint configurations at each sample, in radians [n_samples, n_dof]",
				DataType:    "float64",
				Shape:       []int{-1, -1},
			},
			{
				Name:        totg.KeyVelocitiesRadsPerSec,
				Description: "Joint velocities at each sample, in rad/s [n_samples, n_dof]",
				DataType:    "float64",
				Shape:       []int{-1, -1},
			},
			{
				Name:        totg.KeyAccelerationsRadsPerSec2,
				Description: "Joint accelerations at each sample, in rad/s^2 [n_samples, n_dof]",
				DataType:    "float64",
				Shape:       []int{-1, -1},
			},
		},
	}, nil
}

// insertTensor dispatches on the gorgonia tensor's dtype and inserts it
// into m under key. Returns an error for unsupported dtypes.
func insertTensor(m *trajex.TensorMap, key string, t *tensor.Dense) error {
	if t == nil {
		return errors.New("nil tensor")
	}
	shape := uint64Shape(t.Shape())
	switch t.Dtype() {
	case tensor.Float64:
		data, ok := t.Data().([]float64)
		if !ok {
			return errors.New("Float64 tensor backing is not []float64")
		}
		return m.InsertFloat64s(key, shape, data)
	case tensor.Int64:
		data, ok := t.Data().([]int64)
		if !ok {
			return errors.New("Int64 tensor backing is not []int64")
		}
		return m.InsertInt64s(key, shape, data)
	default:
		return errors.Errorf("unsupported tensor dtype %v", t.Dtype())
	}
}

// readOutputs copies the trajex output map's four float64 tensors out of
// C-owned storage into Go-owned gorgonia tensors.
func readOutputs(m *trajex.TensorMap) (ml.Tensors, error) {
	outputs := ml.Tensors{}
	for _, key := range []string{
		totg.KeySampleTimesSec,
		totg.KeyConfigurationsRads,
		totg.KeyVelocitiesRadsPerSec,
		totg.KeyAccelerationsRadsPerSec2,
	} {
		shape, data, ok, err := m.ViewFloat64s(key)
		if err != nil {
			return nil, errors.Wrapf(err, "trajex/totg/rdk: reading output %q", key)
		}
		if !ok {
			return nil, errors.Errorf("trajex/totg/rdk: output %q missing", key)
		}
		copyData := append([]float64(nil), data...)
		outputs[key] = tensor.New(
			tensor.WithShape(intShape(shape)...),
			tensor.WithBacking(copyData),
		)
	}
	return outputs, nil
}

func uint64Shape(s tensor.Shape) []uint64 {
	out := make([]uint64, len(s))
	for i, v := range s {
		out[i] = uint64(v)
	}
	return out
}

func intShape(s []uint64) []int {
	out := make([]int, len(s))
	for i, v := range s {
		out[i] = int(v)
	}
	return out
}
