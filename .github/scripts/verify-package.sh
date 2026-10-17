#!/usr/bin/env bash
# Shows that an installed trajex can be found with find_package and used, built both ways: with xtl and xtensor
# found on the system, and with them fetched and installed alongside trajex. Builds the consumer project in
# src/viam/trajex/test/cmake against a fresh install of each, and runs its tests.
#
# Usage: verify-package.sh <build-dir> [component...]
#
#   build-dir  A built standalone trajex build, installed as it stands, with whatever xtl and xtensor it found. In
#              CI, those are the container's.
#   component  Components to require and exercise besides totg: compat, capi or moveit. Defaults to compat and capi.
#
# The fetched case is a second build, configured here with only the requested components' dependencies. Everything
# goes under build/verify-package, which each run replaces.

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

# Fetching is forced for every dependency, so the second build enables only what the components need: compat and
# capi are on by default, MoveIt is not. -Werror off, as warnings are gated by the main build.
vendored_arguments=(
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
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
