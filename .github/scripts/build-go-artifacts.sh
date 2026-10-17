#!/usr/bin/env bash
# Builds the trajex archives and C API headers that the Go module links, and stages them into
# go/artifacts as the Go release workflow does. Keeping this in a script rather than in the
# workflow means a developer can reproduce a release build locally. The caller chooses the
# toolchain and the oldest platform to support.
#
# Usage: build-go-artifacts.sh [platform] [-- cmake-configure-arguments...]
#
#   platform  darwin_arm64, linux_amd64 or linux_arm64, defaulting to the host's. Naming another
#             platform is only meaningful with cross-compiling arguments after `--`.
#   --        Everything after it is passed to the cmake configure, as the release workflow does:
#               build-go-artifacts.sh linux_amd64 -- -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10
#               build-go-artifacts.sh darwin_arm64 -- -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
#
# Builds in build/go and stages into go/artifacts, both under the repository root and ignored by
# git. Nothing already in go/artifacts is removed, since release commits keep files there.

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

# Everything that sets this build apart from a development build is here on the command line, so
# CMakeLists.txt doesn't need to know about it. Turning off legacy, compat, tools and the service
# leaves just capi and totg. With the tests off a fresh container doesn't need Boost, and xtl and
# xtensor come through FetchContent. The libraries are static and position independent, so that
# they can link into Go's position independent executables. -Werror is off, because the release
# toolchain may warn where ours doesn't, and we'd rather not fail a release over a warning. The
# build-test-lint job in CI enforces warnings with our toolchain. The CMake package isn't installed,
# because its exports would land in the capi dev component and refer to components that never get
# staged.
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
    -DVIAM_TRAJEX_INSTALL_CMAKE_PACKAGE=OFF
    -DCMAKE_INSTALL_LIBDIR="lib/$platform"
    -DCMAKE_INSTALL_INCLUDEDIR=include
)

cmake -S . -B build/go -G Ninja "${configure_arguments[@]}" "$@"

# Build `all` rather than a list of targets, so that what gets built can't fall out of step with
# what gets staged. With the configuration above, `all` is only capi and totg anyway.
cmake --build build/go --target all

# We stage with DESTDIR rather than installing to a real prefix. These files end up somewhere in
# Go's module cache that we can't know in advance, and an install prefix would write this machine's
# absolute path into any generated config, .pc file or rpath. Nothing generates one today, but
# something might. Using `--prefix /` also keeps the staged tree flat, as
# go/artifacts/{lib/<platform>,include}.
#
# We install just the three components cgo needs, one at a time, since CMake won't follow
# dependencies between components for us. VIAM_TRAJEX_INSTALL_ARCHIVES_AS_RUNTIME moved the capi and
# totg archives into the runtime components, which leaves only capi.h, and the export.h it
# includes, in the capi dev component. The C++ headers and any fetched xtl and xtensor stay out of
# the shipped module.
for component in trajex-capi-runtime trajex-totg-runtime trajex-capi-dev; do
    DESTDIR="$PWD/go/artifacts" cmake --install build/go --prefix / --component "$component"
done
