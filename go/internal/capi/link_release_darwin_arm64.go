//go:build darwin && arm64

package capi

// Linkage for the prebuilt trajex archives on darwin/arm64. The archives are
// not self-contained: capi references totg, so both are named here, capi
// first. Paths resolve into go/artifacts, populated by `cmake --install` (dev)
// or the release pipeline (checked into the go/vX.Y.Z tag commit).

/*
#cgo LDFLAGS: -L${SRCDIR}/../../artifacts/lib/darwin_arm64 -lviam-trajex-capi -lviam-trajex-totg
*/
import "C"
