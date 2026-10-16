#!/usr/bin/env bash
# Builds the trajex archives and C API header that the Go module links, and stages them into
# go/artifacts as the Go release workflow does. The artifact is defined here and nowhere else.
# The toolchain, and the platform floor a release pins, belong to the caller.
#
# Usage: build-go-artifacts.sh [platform] [-- cmake-configure-arguments...]
#
#   platform  darwin_arm64, linux_amd64 or linux_arm64, defaulting to the host's. Naming another
#             platform is only meaningful with cross-compiling arguments after `--`.
#   --        Everything after it is passed to the cmake configure. The release passes these:
#               build-go-artifacts.sh linux_amd64 -- -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10
#               build-go-artifacts.sh darwin_arm64 -- -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
#
# Builds in build/go and stages into go/artifacts, both under the repository root and ignored by
# git. Nothing already in go/artifacts is removed: release commits keep files there.

set -euxo pipefail

cd "$(dirname "$0")/../.."

host_platform() {
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) echo darwin_arm64 ;;
        Linux-x86_64) echo linux_amd64 ;;
        Linux-aarch64 | Linux-arm64) echo linux_arm64 ;;
        *)
            echo "build-go-artifacts: no Go platform for host $(uname -s) $(uname -m)" >&2
            return 1
            ;;
    esac
}

platform=""
if [ $# -gt 0 ] && [ "$1" != "--" ]; then
    platform=$1
    shift
fi
if [ $# -gt 0 ]; then
    if [ "$1" != "--" ]; then
        echo "usage: $0 [platform] [-- cmake-configure-arguments...]" >&2
        exit 2
    fi
    shift
fi
if [ -z "$platform" ]; then
    platform=$(host_platform)
fi
case "$platform" in
    darwin_arm64 | linux_amd64 | linux_arm64) ;;
    *)
        echo "build-go-artifacts: unknown platform '$platform'" >&2
        exit 2
        ;;
esac

# The distribution profile, all on the command line, so CMakeLists.txt stays unaware of it.
# Minimal dependencies (legacy, compat, tools, service off) leave just capi and totg. Tests off, so
# a fresh container needs no Boost; xtl and xtensor come via FetchContent. Static and PIC, so the
# archives link into Go's PIE. -Werror off: whatever toolchain builds this may warn where the
# pinned development compiler doesn't, and the artifact must not fail on that. Warnings are gated
# by main.yml's build-test-lint job, on the pinned toolchain.
configure_arguments=(
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
    -DVIAM_TRAJEX_ENABLE_LEGACY=OFF
    -DVIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT=OFF
    -DVIAM_TRAJEX_ENABLE_TOOLS=OFF
    -DVIAM_TRAJEX_ENABLE_SERVICE=OFF
    -DVIAM_TRAJEX_BUILD_TESTS=OFF
    -DVIAM_TRAJEX_BUILD_BENCHMARKS=OFF
    -DVIAM_TRAJEX_USE_WALL_WERROR=OFF
    -DVIAM_TRAJEX_INSTALL_ARCHIVES_AS_RUNTIME=ON
    -DCMAKE_INSTALL_LIBDIR="lib/$platform"
    -DCMAKE_INSTALL_INCLUDEDIR=include
)

cmake -S . -B build/go -G Ninja "${configure_arguments[@]}" "$@"

# `all`, not a hand-picked target list: the minimal configuration makes `all` just capi and totg
# anyway, and building everything keeps what is built from drifting away from what is staged.
cmake --build build/go --target all

# Stage, rather than install to a prefix. These are relocatable archives and a header packaged for
# shipping, whose real home is an unknowable module-cache path, so this is a DESTDIR install, not a
# CMAKE_INSTALL_PREFIX: a prefix would bake this machine's absolute path into any generated config,
# .pc or rpath (none today, but a landmine). `--prefix /` gives a neutral root, so the staged tree is
# flat: go/artifacts/{lib/<platform>,include}.
#
# Exactly what cgo links and includes, one component each, since a component install takes one
# component and CMake follows no dependencies between them: the capi and totg archives (runtime,
# because of VIAM_TRAJEX_INSTALL_ARCHIVES_AS_RUNTIME) and the capi header (capi dev, which with that
# option holds only the header). The C++ headers, and the FetchContent'd xtl and xtensor, are in
# other components and excluded, so they don't bloat the shipped module.
for component in trajex-capi-runtime trajex-totg-runtime trajex-capi-dev; do
    DESTDIR="$PWD/go/artifacts" cmake --install build/go --prefix / --component "$component"
done
