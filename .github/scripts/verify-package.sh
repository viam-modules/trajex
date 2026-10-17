#!/usr/bin/env bash
# Checks that an installed trajex can be found with find_package and used, whether it found xtl and xtensor on the
# system or fetched and installed its own. It installs the given build, then makes and installs a second one that
# fetches them, and builds and tests the consumer project in src/viam/trajex/test/cmake against both.
#
# Usage: verify-package.sh <build-dir> [component...]
#
#   build-dir  An existing standalone trajex build directory. It's installed as it is, with whatever xtl and xtensor
#              it found (in CI, the container's).
#   component  Components to test in addition to totg: any of compat, capi and moveit. Defaults to compat and capi.
#
# Everything goes under build/verify-package, which is removed at the start of each run.

set -euxo pipefail

cd "$(dirname "$0")/../.."

if [ $# -lt 1 ]; then
    echo "usage: $0 <build-dir> [component...]" >&2
    exit 2
fi
build=$1
shift
components=("$@")
if [ ${#components[@]} -eq 0 ]; then
    components=(compat capi)
fi
component_list=$(IFS=';' && echo "${components[*]}")

root=$PWD/build/verify-package
rm -rf "$root"

# Installs a build into its own prefix, then builds and tests the consumer against that prefix alone. Any further
# arguments go to the consumer's configure.
verify() {
    local name=$1 from=$2
    shift 2
    cmake --install "$from" --prefix "$root/$name/install"
    cmake -S src/viam/trajex/test/cmake -B "$root/$name/consumer" -G Ninja \
        -DCMAKE_PREFIX_PATH="$root/$name/install" -DTRAJEX_COMPONENTS="$component_list" "$@"
    cmake --build "$root/$name/consumer"
    ctest --test-dir "$root/$name/consumer" --output-on-failure --timeout 60
}

verify as-built "$build"

# Since this build fetches every dependency, we turn on only what the requested components need. Compat and capi are
# on by default, and MoveIt is added below if it was asked for. -Werror is off, because the main build already
# enforces warnings. This build is static, so that together with the shared main build, both kinds of library get
# tested.
vendored_arguments=(
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DBUILD_SHARED_LIBS=OFF
    -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
    -DVIAM_TRAJEX_ENABLE_LEGACY=OFF
    -DVIAM_TRAJEX_ENABLE_TOOLS=OFF
    -DVIAM_TRAJEX_ENABLE_SERVICE=OFF
    -DVIAM_TRAJEX_BUILD_TESTS=OFF
    -DVIAM_TRAJEX_BUILD_BENCHMARKS=OFF
    -DVIAM_TRAJEX_USE_WALL_WERROR=OFF
)
for component in "${components[@]}"; do
    if [ "$component" = moveit ]; then
        vendored_arguments+=(-DVIAM_TRAJEX_ENABLE_MOVEIT=ON)
    fi
done
cmake -S . -B "$root/vendored/build" -G Ninja "${vendored_arguments[@]}"
cmake --build "$root/vendored/build"
verify vendored "$root/vendored/build" -DTRAJEX_VENDORED_PREFIX="$root/vendored/install"
