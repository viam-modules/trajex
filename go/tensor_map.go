//go:build !windows && !no_cgo

// Package trajex provides Go bindings for the trajex C ABI defined in
// src/viam/trajex/capi/capi.h. The public surface is algorithm-agnostic: a
// TensorMap handle plus typed insert and view operations. Algorithm-specific
// entry points (e.g. TOTG) live in subpackages such as totg.
package trajex

/*
#cgo CFLAGS: -I${SRCDIR}/../src/viam/trajex/capi

#include <stdlib.h>
#include "capi.h"
*/
import "C"

import (
	"unsafe"

	"github.com/pkg/errors"

	"github.com/viam-modules/trajex/go/internal/capi"
)

// TensorMap is a Go-owned handle to a CAPI tensor map. The underlying C
// resource is released by Close, which must be called (typically via defer)
// to avoid leaks. TensorMap is not safe for concurrent use; concurrent
// operations on the same handle are the caller's responsibility, matching
// the model of the underlying C ABI.
type TensorMap struct {
	handle *C.viam_trajex_tensor_map_t
}

// NewTensorMap constructs an empty TensorMap. The returned handle must be
// released by calling Close, typically via defer.
func NewTensorMap() (*TensorMap, error) {
	h := C.viam_trajex_tensor_map_create()
	if h == nil {
		return nil, errors.New("trajex: tensor map allocation failed")
	}
	return &TensorMap{handle: h}, nil
}

// Close releases the underlying C tensor map. Safe to call on a nil
// receiver and idempotent: subsequent calls are no-ops.
func (m *TensorMap) Close() {
	if m == nil || m.handle == nil {
		return
	}
	C.viam_trajex_tensor_map_destroy(m.handle)
	m.handle = nil
}

// UnsafeHandle returns the underlying C tensor map handle as an
// unsafe.Pointer. Intended for sibling Go packages implementing other CAPI
// entry points (e.g. totg.Generate); not part of the supported consumer
// surface. Returns nil if the TensorMap has been closed.
func (m *TensorMap) UnsafeHandle() unsafe.Pointer {
	if m == nil {
		return nil
	}
	return unsafe.Pointer(m.handle)
}

// InsertFloat64s copies a float64 tensor into the map under key. shape is a
// row-major dims vector; len(data) must equal the product of shape entries.
// Replaces any existing entry at key transactionally (mid-call failure
// leaves the prior entry intact).
func (m *TensorMap) InsertFloat64s(key string, shape []uint64, data []float64) error {
	if err := validateShape(shape, len(data)); err != nil {
		return err
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cDims := toCDims(shape)
	rc := C.viam_trajex_tensor_map_insert_f64(
		m.handle,
		cKey,
		C.size_t(len(shape)),
		&cDims[0],
		(*C.double)(unsafe.Pointer(&data[0])),
	)
	if rc != 0 {
		return errors.Errorf("trajex: insert %q (f64) failed", key)
	}
	return nil
}

// InsertInt64s copies an int64 tensor into the map under key. shape is a
// row-major dims vector; len(data) must equal the product of shape entries.
// Replaces any existing entry at key transactionally (mid-call failure
// leaves the prior entry intact).
func (m *TensorMap) InsertInt64s(key string, shape []uint64, data []int64) error {
	if err := validateShape(shape, len(data)); err != nil {
		return err
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cDims := toCDims(shape)
	rc := C.viam_trajex_tensor_map_insert_i64(
		m.handle,
		cKey,
		C.size_t(len(shape)),
		&cDims[0],
		(*C.int64_t)(unsafe.Pointer(&data[0])),
	)
	if rc != 0 {
		return errors.Errorf("trajex: insert %q (i64) failed", key)
	}
	return nil
}

// InsertScalarFloat64 stores value as a rank-1 shape-[1] float64 tensor
// under key. Convenience for the schema convention used for scalar inputs.
func (m *TensorMap) InsertScalarFloat64(key string, value float64) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	rc := C.viam_trajex_tensor_map_insert_scalar_f64(m.handle, cKey, C.double(value))
	if rc != 0 {
		return errors.Errorf("trajex: insert scalar %q (f64) failed", key)
	}
	return nil
}

// InsertScalarInt64 stores value as a rank-1 shape-[1] int64 tensor under
// key. Convenience for the schema convention used for scalar inputs.
func (m *TensorMap) InsertScalarInt64(key string, value int64) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	rc := C.viam_trajex_tensor_map_insert_scalar_i64(m.handle, cKey, C.int64_t(value))
	if rc != 0 {
		return errors.Errorf("trajex: insert scalar %q (i64) failed", key)
	}
	return nil
}

// ViewFloat64s returns the shape and data of the float64 tensor under key.
// The returned data slice aliases C-owned storage and remains valid until
// the map is destroyed (via Close) or the key is replaced (via another
// insert). Callers that need persistence beyond that lifetime must copy
// the slice.
//
// ok is false (with nil err) if the key is absent. If the key is present
// but holds a tensor of a different dtype, err is non-nil.
func (m *TensorMap) ViewFloat64s(key string) (shape []uint64, data []float64, ok bool, err error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	var (
		dtype C.viam_trajex_dtype_t
		rank  C.size_t
		dims  *C.size_t
		dPtr  unsafe.Pointer
	)
	rc := C.viam_trajex_tensor_map_view(
		m.handle, cKey, &dtype, &rank, &dims,
		(*unsafe.Pointer)(unsafe.Pointer(&dPtr)),
	)
	switch rc {
	case 0:
	case 1:
		return nil, nil, false, nil
	default:
		return nil, nil, false, errors.Errorf("trajex: view %q failed", key)
	}
	if capi.Dtype(dtype) != capi.DtypeF64 {
		return nil, nil, false, errors.Errorf(
			"trajex: view %q: expected dtype f64, got dtype %d", key, int(dtype))
	}
	shape, total := copyDimsAndTotal(dims, rank)
	return shape, unsafe.Slice((*float64)(dPtr), int(total)), true, nil
}

// ViewInt64s returns the shape and data of the int64 tensor under key. The
// returned data slice aliases C-owned storage and remains valid until the
// map is destroyed (via Close) or the key is replaced (via another insert).
// Callers that need persistence beyond that lifetime must copy the slice.
//
// ok is false (with nil err) if the key is absent. If the key is present
// but holds a tensor of a different dtype, err is non-nil.
func (m *TensorMap) ViewInt64s(key string) (shape []uint64, data []int64, ok bool, err error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	var (
		dtype C.viam_trajex_dtype_t
		rank  C.size_t
		dims  *C.size_t
		dPtr  unsafe.Pointer
	)
	rc := C.viam_trajex_tensor_map_view(
		m.handle, cKey, &dtype, &rank, &dims,
		(*unsafe.Pointer)(unsafe.Pointer(&dPtr)),
	)
	switch rc {
	case 0:
	case 1:
		return nil, nil, false, nil
	default:
		return nil, nil, false, errors.Errorf("trajex: view %q failed", key)
	}
	if capi.Dtype(dtype) != capi.DtypeI64 {
		return nil, nil, false, errors.Errorf(
			"trajex: view %q: expected dtype i64, got dtype %d", key, int(dtype))
	}
	shape, total := copyDimsAndTotal(dims, rank)
	return shape, unsafe.Slice((*int64)(dPtr), int(total)), true, nil
}

// validateShape checks that shape is non-empty, every dim is at least 1,
// and the product of dims equals dataLen.
func validateShape(shape []uint64, dataLen int) error {
	if len(shape) == 0 {
		return errors.New("trajex: shape must have at least one dimension")
	}
	var total uint64 = 1
	for i, d := range shape {
		if d == 0 {
			return errors.Errorf("trajex: shape[%d] is zero", i)
		}
		total *= d
	}
	if uint64(dataLen) != total {
		return errors.Errorf(
			"trajex: data length %d does not match shape product %d", dataLen, total)
	}
	return nil
}

// toCDims allocates a []C.size_t backing array sized to shape and copies
// the values.
func toCDims(shape []uint64) []C.size_t {
	out := make([]C.size_t, len(shape))
	for i, d := range shape {
		out[i] = C.size_t(d)
	}
	return out
}

// copyDimsAndTotal copies a C dims pointer of length rank into a fresh
// []uint64 and returns it alongside the product of dimensions.
func copyDimsAndTotal(dims *C.size_t, rank C.size_t) ([]uint64, uint64) {
	n := int(rank)
	cDims := unsafe.Slice(dims, n)
	out := make([]uint64, n)
	var total uint64 = 1
	for i, d := range cDims {
		out[i] = uint64(d)
		total *= uint64(d)
	}
	return out, total
}
