//go:build !windows && !no_cgo

// Package capi is the single owner of the link directives for the trajex C
// ABI shared library. Any Go package in this module that calls into the C
// ABI (directly via `import "C"` and `C.viam_*` symbols) must transitively
// depend on this package; otherwise the resulting binary will not link
// against the trajex CAPI library.
//
// Beyond carrying linkage, this package owns the small set of C-ABI types
// and helpers shared by every cgo wrapper in the module: Dtype and CStr.
// Higher-level concepts (TensorMap, TOTG entry points) live in their own
// sibling packages.
package capi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../src/viam/trajex/capi
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../build -lviam-trajex-capi -Wl,-rpath,${SRCDIR}/../../../build
#cgo linux  LDFLAGS: -L${SRCDIR}/../../../build -lviam-trajex-capi -Wl,-rpath,${SRCDIR}/../../../build

#include "capi.h"
*/
import "C"

import "unsafe"

// Dtype is the element type carried in a CAPI tensor map value. Integer
// values match the VIAM_TRAJEX_DTYPE_* constants in the C ABI and are
// stable across releases.
type Dtype int

const (
	// DtypeF64 represents IEEE 754 double-precision floating point.
	DtypeF64 Dtype = 1
	// DtypeI64 represents signed 64-bit two's-complement integer.
	DtypeI64 Dtype = 2
)

// CStr converts the address of a cgo-exposed `extern const char foo[]`
// symbol into a Go string. cgo declares such externs as zero-length arrays
// (`[0]C.char`), so direct indexing fails at compile time; reading them as
// C strings requires reinterpreting their address as `*C.char`.
//
// Callers pass &C.viam_<symbol> wrapped in unsafe.Pointer; the conversion
// to *C.char is finalized here in capi's own cgo context.
func CStr(p unsafe.Pointer) string {
	return C.GoString((*C.char)(p))
}
