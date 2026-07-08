//go:build linux && arm64 && !no_cgo

package capi

// Linkage for the prebuilt trajex archives on linux/arm64. See the darwin/arm64
// variant for the rationale behind naming both archives.

/*
#cgo LDFLAGS: -L${SRCDIR}/../../artifacts/lib/linux_arm64 -lviam-trajex-capi -lviam-trajex-totg
*/
import "C"
