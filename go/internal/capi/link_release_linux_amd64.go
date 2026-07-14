//go:build linux && amd64

package capi

// Linkage for the prebuilt trajex archives on linux/amd64. See the darwin/arm64
// variant for the rationale behind naming both archives.

/*
#cgo LDFLAGS: -L${SRCDIR}/../../artifacts/lib/linux_amd64 -lviam-trajex-capi -lviam-trajex-totg
*/
import "C"
