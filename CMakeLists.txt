cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(trajex VERSION 0.101.0
  DESCRIPTION "Viam trajectory execution library"
  HOMEPAGE_URL https://github.com/viam-modules/trajex
  LANGUAGES C CXX
)

# Defines the CMAKE_INSTALL_* directories, which the RPATH setup and the targets' install
# include directories below read. It must come before any of them: GNUInstallDirs caches its
# variables, so a later include only takes effect from the second configure onward.
include(GNUInstallDirs)

# Detect if we're building trajex as a standalone module
# vs being included via FetchContent by universal-robots
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(VIAM_TRAJEX_STANDALONE_BUILD ON)
  message(STATUS "Trajex: Standalone build mode")
else()
  set(VIAM_TRAJEX_STANDALONE_BUILD OFF)
  message(STATUS "Trajex: FetchContent mode (included by parent project)")
endif()

# Build the trajex test programs (boost.test executables registered with ctest).
# Defaults ON for standalone builds, OFF for FetchContent consumers.
option(VIAM_TRAJEX_BUILD_TESTS "Build the Trajex test programs" ${VIAM_TRAJEX_STANDALONE_BUILD})

# Build the trajex benchmark programs (google-benchmark executables, deliberately NOT
# registered with ctest: benchmarks under ctest give noisy numbers and slow the test loop).
# Requires `VIAM_TRAJEX_ENABLE_TOOLS` for replay record parsing.
# Defaults ON for standalone builds, OFF for FetchContent consumers.
option(VIAM_TRAJEX_BUILD_BENCHMARKS "Build the Trajex benchmark programs" ${VIAM_TRAJEX_STANDALONE_BUILD})

# Enable the third-party reference implementation (`third_party/trajectories`) and the
# `viam::trajex::totg::legacy::` colinearization helpers used by the legacy generator
# path. Pulls in Eigen. Useful for cross-validating the new TOTG implementation.
option(VIAM_TRAJEX_ENABLE_LEGACY "Enable the reference implementation" ON)

# Enable the TOTG compatibility layer (`viam::trajex::totg::compat`): the reference
# implementation's `Path` and `Trajectory` interface backed by trajex, so code written against
# the original can try trajex without changes, plus a shim target that serves that interface
# under the original header names. Pulls in Eigen. Defaults ON for standalone builds, OFF for
# FetchContent consumers.
option(VIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT "Enable the TOTG reference implementation compatibility layer" ${VIAM_TRAJEX_STANDALONE_BUILD})

# Enable the MoveIt 2 integration (`viam::trajex::moveit::totg`): a MoveIt time parameterization backed by trajex,
# for calling directly from C++. Requires an installed MoveIt 2 found through the ROS environment, so source the ROS
# setup script before configuring. The MoveIt plugins are not built here; they are built by the ament package in
# `src/viam/trajex/moveit`, which enables this option. Defaults OFF.
option(VIAM_TRAJEX_ENABLE_MOVEIT "Enable the MoveIt 2 integration" OFF)

# Install static archives in each component's runtime role rather than its dev role. For consumers
# that take the archives as a payload rather than as something to develop against, such as the Go
# module, which stages the C API's archives and header. Not for find_package consumers: the package
# config is a dev artifact that refers to the archives. No effect on a shared build. Defaults
# OFF.
option(VIAM_TRAJEX_INSTALL_ARCHIVES_AS_RUNTIME "Install static archives in the runtime role" OFF)

# Install the CMake package, so that an installed trajex can be found with `find_package(trajex)`:
# the package config, and an export of each library that installs public headers. Turn it off for
# consumers that stage individual components as a payload, such as the Go module, where the export
# files would refer to components that were never staged. Defaults ON for standalone builds, OFF for
# FetchContent consumers, which can turn it on to make trajex findable in what they install.
option(VIAM_TRAJEX_INSTALL_CMAKE_PACKAGE "Install the CMake package config and target exports" ${VIAM_TRAJEX_STANDALONE_BUILD})

# Enable the planner, replay, and JSON serialization tooling layer
# (`viam::trajex::totg::tools`). Required by the service layer and by the regression test
# suite (the latter for opt-in trajectory dumping; see the TODO at the test target).
# Pulls in jsoncpp; pulls in Eigen when `VIAM_TRAJEX_ENABLE_LEGACY` is also ON.
# Defaults ON for standalone builds, OFF for FetchContent consumers.
option(VIAM_TRAJEX_ENABLE_TOOLS "Enable the planner, replay, and serialization tooling layer" ${VIAM_TRAJEX_STANDALONE_BUILD})

# Enable the Viam SDK MLModel service layer (`viam::trajex::service`) and the
# `viam-trajex-service` module executable. Requires `VIAM_TRAJEX_ENABLE_TOOLS`. Pulls in
# `viam-cpp-sdk` (the heaviest external dependency, transitively grpc/protobuf/abseil).
# Defaults ON for standalone builds, OFF for FetchContent consumers.
option(VIAM_TRAJEX_ENABLE_SERVICE "Enable the Viam MLModel service layer (requires VIAM_TRAJEX_ENABLE_TOOLS)" ${VIAM_TRAJEX_STANDALONE_BUILD})

# Enable the trajex C ABI shim layer (`viam::trajex::capi`): library `viam-trajex-capi`
# plus header `src/viam/trajex/capi/capi.h`. Provides stable, language-agnostic entry
# points to the TOTG generator for non-C++ callers (cgo, FFI, etc.). Links only against
# the core TOTG library; no additional dependencies beyond what trajex core already
# pulls in. Defaults ON in all configurations so CI exercises the surface continuously.
option(VIAM_TRAJEX_ENABLE_CAPI "Enable the C ABI shim" ON)

# Compile trajex targets with `-Wall -Werror` (and the rest of the project's strict
# warning set) on supported compilers. Defaults ON for standalone builds, OFF for
# FetchContent consumers so a parent project's looser warning posture isn't broken by
# upstream warnings.
option(VIAM_TRAJEX_USE_WALL_WERROR "Build with -Wall and -Werror flags" ${VIAM_TRAJEX_STANDALONE_BUILD})

# Set minimum macOS deployment target to 14.0 (Sonoma) for mature
# C++20 support in Xcode 15's libc++.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")
endif()

# Configure RPATH for relocatable installs (standalone builds only)
if(VIAM_TRAJEX_STANDALONE_BUILD)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH false)
  set(CMAKE_BUILD_RPATH_USE_ORIGIN true)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
  set(CMAKE_SKIP_BUILD_RPATH false)

  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
       "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
    file(RELATIVE_PATH BINTOPREFIXRELPATH
         ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
         ${CMAKE_INSTALL_PREFIX})
    file(RELATIVE_PATH PREFIXTOLIBRELPATH
         ${CMAKE_INSTALL_PREFIX}
         ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      list(PREPEND CMAKE_INSTALL_RPATH
           "@loader_path/${BINTOPREFIXRELPATH}/${PREFIXTOLIBRELPATH}")
    else()
      list(PREPEND CMAKE_INSTALL_RPATH
           "$ORIGIN/${BINTOPREFIXRELPATH}/${PREFIXTOLIBRELPATH}")
    endif()
  endif()
endif()

function(apply_wall_werror target_name)
  if (NOT VIAM_TRAJEX_USE_WALL_WERROR)
    return()
  endif()

  set(GNULIKE_COMPILERS "Clang" "AppleClang" "GNU")
  if (CMAKE_CXX_COMPILER_ID IN_LIST GNULIKE_COMPILERS)
    target_compile_options(${target_name} PRIVATE
      -Wall
      -Werror
      -Wextra
      -pedantic
      -Wconversion
      -Wdouble-promotion
      -Wenum-conversion
      -Wnull-dereference
      -Wunused
      # TODO(RSDK-11300): rm after upgrading boost https://github.com/boostorg/mpl/issues/69
      -Wno-enum-constexpr-conversion
      -Wno-unknown-warning-option
    )
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${target_name} PRIVATE /W4 /WX)
  else()
    message(FATAL_ERROR "VIAM_TRAJEX_USE_WALL_WERROR is set, but not known how to enable for compiler ID ${CMAKE_CXX_COMPILER_ID}")
  endif()
endfunction()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED True)
if (MSVC)
  string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check that every public header compiles on its own, so that none depends on what happens to
# be included before it. Standalone builds only, so that a parent project's build is not slowed
# by checks of headers it did not write.
if (VIAM_TRAJEX_STANDALONE_BUILD)
  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS ON)
endif()


#
# Dependencies
#

# Minimum required version for each external dependency. An empty value means
# we do not enforce a minimum; the package's `find_package` (or its
# `FIND_PACKAGE_ARGS`-mediated equivalent) accepts whatever is found.
# Below 1.9.5 `RegisterBenchmarkInternal` takes a raw pointer and clang-tidy reports every
# benchmark registration as a leak.
set(VIAM_TRAJEX_BENCHMARK_VERSION_MINIMUM 1.9.5)
set(VIAM_TRAJEX_BOOST_VERSION_MINIMUM "")    # TODO: Decide this
set(VIAM_TRAJEX_EIGEN3_VERSION_MINIMUM "")   # TODO: Decide this
set(VIAM_TRAJEX_JSONCPP_VERSION_MINIMUM "")  # TODO: Decide this
set(VIAM_TRAJEX_VIAMCPPSDK_VERSION_MINIMUM 0.44.0)
set(VIAM_TRAJEX_XTENSOR_VERSION_MINIMUM 0.24.3)
set(VIAM_TRAJEX_XTL_VERSION_MINIMUM 0.7.2)

# Each external dependency is acquired here, gated to the feature flag(s) whose
# enablement requires it. The gates below mirror but do not duplicate the
# target-construction gates further down: each is a single boolean check that
# documents which flag activates which dependency.
#
#   xtl, xtensor   core; FetchContent with system fallback via FIND_PACKAGE_ARGS.
#                  When ENABLE_SERVICE is on, viam-cpp-sdk is fetched first and its
#                  own xtl/xtensor declares win by first-wins; our declarations
#                  below become silent no-ops. Pins match the SDK's current pins so
#                  the no-SDK path tracks the same versions.
#   viam-cpp-sdk   ENABLE_SERVICE; FetchContent with find_package fallback.
#   Eigen3         ENABLE_LEGACY (reference impl + colinearization helpers), and
#                  TOTG_ENABLE_LEGACY_COMPAT (the compatibility layer's interface).
#   moveit_core    ENABLE_MOVEIT; found through the ROS environment, never fetched.
#   Boost          BUILD_TESTS (boost.test framework, consumed in header-only mode
#                  via `<boost/test/included/unit_test.hpp>`; we still declare the
#                  `unit_test_framework` component to validate it is installed.
#                  The service code path uses Boost headers too but receives them
#                  transitively through viam-cpp-sdk).
#   jsoncpp        ENABLE_TOOLS (planner / replay / serialization sources).
#   benchmark      BUILD_BENCHMARKS; FetchContent with system fallback via
#                  FIND_PACKAGE_ARGS, so a system install (e.g. homebrew) is preferred
#                  and the fetch is only a backstop.

include(FetchContent)

if (VIAM_TRAJEX_ENABLE_SERVICE)
  FetchContent_Declare(
    viam-cpp-sdk
    GIT_REPOSITORY https://github.com/viamrobotics/viam-cpp-sdk
    GIT_TAG        releases/v0.44.0
    GIT_SHALLOW    TRUE
    SYSTEM
    # Delete conflicting protobuf headers
    PATCH_COMMAND find ./src/viam/api -name "*.pb.h" -type f -exec rm {} +
    FIND_PACKAGE_ARGS ${VIAM_TRAJEX_VIAMCPPSDK_VERSION_MINIMUM} CONFIG COMPONENTS viamsdk
  )
  set(VIAMCPPSDK_USE_WALL_WERROR OFF)
  FetchContent_MakeAvailable(viam-cpp-sdk)
endif()

FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG        a7c1c5444dfc57f76620391af4c94785ff82c8d6 # 0.7.7
  FIND_PACKAGE_ARGS ${VIAM_TRAJEX_XTL_VERSION_MINIMUM} CONFIG
)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
  GIT_TAG        3634f2ded19e0cf38208c8b86cea9e1d7c8e397d # 0.25.0
  FIND_PACKAGE_ARGS ${VIAM_TRAJEX_XTENSOR_VERSION_MINIMUM} CONFIG
)

# When xtl and xtensor are fetched rather than found, they install themselves alongside trajex,
# because the installed trajex headers include theirs. Neither names a component for its install
# rules, so this default gives them one of ours, rather than CMake's "Unspecified".
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME trajex-vendored-dependencies-dev)
FetchContent_MakeAvailable(xtl xtensor)
unset(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME)

if (VIAM_TRAJEX_ENABLE_LEGACY)
  find_package(Eigen3 ${VIAM_TRAJEX_EIGEN3_VERSION_MINIMUM} CONFIG REQUIRED)
endif()

if (VIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT)
  find_package(Eigen3 ${VIAM_TRAJEX_EIGEN3_VERSION_MINIMUM} CONFIG REQUIRED)
endif()

if (VIAM_TRAJEX_ENABLE_MOVEIT)
  find_package(moveit_core CONFIG REQUIRED)
endif()

if (VIAM_TRAJEX_BUILD_TESTS)
  find_package(Boost ${VIAM_TRAJEX_BOOST_VERSION_MINIMUM} CONFIG REQUIRED COMPONENTS unit_test_framework)
endif()

if (VIAM_TRAJEX_ENABLE_TOOLS)
  # jsoncppConfig.cmake doesn't guard against re-import, so a second find_package
  # creates a duplicate imported target and fails. Guard for the FetchContent case
  # where a parent project already provided it.
  if (NOT TARGET jsoncpp_lib)
    find_package(jsoncpp ${VIAM_TRAJEX_JSONCPP_VERSION_MINIMUM} CONFIG REQUIRED)
  endif()
endif()

if (VIAM_TRAJEX_BUILD_BENCHMARKS)
  # google-benchmark's own test suite pulls googletest, which we have no use for.
  set(BENCHMARK_ENABLE_TESTING OFF)
  set(BENCHMARK_ENABLE_INSTALL OFF)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.5
    SYSTEM
    FIND_PACKAGE_ARGS ${VIAM_TRAJEX_BENCHMARK_VERSION_MINIMUM} CONFIG
  )
  FetchContent_MakeAvailable(benchmark)
endif()


#
# Library targets
#

# Legacy library: reference implementation from third_party/trajectories submodule. Only contains
# meaningful content if `VIAM_TRAJEX_ENABLE_LEGACY` is truthy.

if (VIAM_TRAJEX_ENABLE_LEGACY)
  add_library(viam-trajex-totg-legacy)

  set_target_properties(viam-trajex-totg-legacy PROPERTIES SYSTEM TRUE)

  target_sources(viam-trajex-totg-legacy
    PRIVATE
    src/third_party/trajectories/Trajectory.cpp
    src/third_party/trajectories/Path.cpp
  )

  target_compile_definitions(viam-trajex-totg-legacy
    INTERFACE
    VIAM_TRAJEX_LEGACY_ENABLED=1
  )

  target_include_directories(viam-trajex-totg-legacy
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  target_precompile_headers(viam-trajex-totg-legacy PRIVATE <assert.h>)

  target_link_libraries(viam-trajex-totg-legacy
    PUBLIC
    Eigen3::Eigen
  )
else()
  add_library(viam-trajex-totg-legacy INTERFACE)
endif()

add_library(viam::trajex::totg::legacy ALIAS viam-trajex-totg-legacy)


# New TOTG implementation
add_library(viam-trajex-totg)

target_sources(viam-trajex-totg
  PRIVATE
    src/viam/trajex/trajex.cpp
    src/viam/trajex/kinematics/chain.cpp
    src/viam/trajex/totg/observers.cpp
    src/viam/trajex/totg/path.cpp
    src/viam/trajex/totg/streaming/session.cpp
    src/viam/trajex/totg/trajectory.cpp
    src/viam/trajex/totg/uniform_sampler.cpp
    src/viam/trajex/totg/waypoint_accumulator.cpp
    src/viam/trajex/totg/waypoint_utils.cpp
)

# The public headers, installed with their paths under src preserved. Headers under a
# `private` directory are deliberately absent: they are not installed. The base directory is
# also the include directory, for this target and for anything that links it.
target_sources(viam-trajex-totg
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES
    src/viam/trajex/trajex.hpp
    src/viam/trajex/jacobian/jacobian.hpp
    src/viam/trajex/totg/observers.hpp
    src/viam/trajex/totg/path.hpp
    src/viam/trajex/totg/streaming/session.hpp
    src/viam/trajex/totg/totg.hpp
    src/viam/trajex/totg/trajectory.hpp
    src/viam/trajex/totg/uniform_sampler.hpp
    src/viam/trajex/totg/waypoint_accumulator.hpp
    src/viam/trajex/totg/waypoint_utils.hpp
    src/viam/trajex/types/angles.hpp
    src/viam/trajex/types/arc_acceleration.hpp
    src/viam/trajex/types/arc_length.hpp
    src/viam/trajex/types/arc_operations.hpp
    src/viam/trajex/types/arc_velocity.hpp
    src/viam/trajex/types/epsilon.hpp
    src/viam/trajex/types/hertz.hpp
    src/viam/trajex/types/xt.hpp
)

target_link_libraries(viam-trajex-totg
  PUBLIC
    xtensor
    xtl
)

apply_wall_werror(viam-trajex-totg)

add_library(viam::trajex::totg ALIAS viam-trajex-totg)


# Compatibility layer: the reference implementation's `Path` and `Trajectory` interface in the
# `viam::trajex::totg::compat` namespace, backed by trajex. Its headers need only Eigen, so
# trajex stays a private dependency. The shim target adds a directory of headers named after
# the originals, which bring the compat classes into the global namespace. Only contains
# meaningful content if `VIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT` is truthy.

if (VIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT)
  add_library(viam-trajex-totg-compat)

  target_sources(viam-trajex-totg-compat
    PRIVATE
    src/viam/trajex/totg/compat/path.cpp
    src/viam/trajex/totg/compat/trajectory.cpp
  )

  target_compile_definitions(viam-trajex-totg-compat
    INTERFACE
    VIAM_TRAJEX_TOTG_LEGACY_COMPAT_ENABLED=1
  )

  target_sources(viam-trajex-totg-compat
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES
    src/viam/trajex/totg/compat/path.hpp
    src/viam/trajex/totg/compat/trajectory.hpp
  )

  target_link_libraries(viam-trajex-totg-compat
    PUBLIC
    Eigen3::Eigen
    PRIVATE
    viam::trajex::totg
  )

  apply_wall_werror(viam-trajex-totg-compat)

  add_library(viam-trajex-totg-compat-shim INTERFACE)

  # Installed with their paths preserved like every other header, so that the shim's names
  # do not land at the top of the include tree. Callers include them by bare name, so the
  # shim directory itself is the include directory, in the build tree and once installed.
  target_sources(viam-trajex-totg-compat-shim
    INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES
    src/viam/trajex/totg/compat/shim/Path.h
    src/viam/trajex/totg/compat/shim/Trajectory.h
  )

  target_include_directories(viam-trajex-totg-compat-shim
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/viam/trajex/totg/compat/shim>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/viam/trajex/totg/compat/shim>
  )

  target_link_libraries(viam-trajex-totg-compat-shim
    INTERFACE
    viam::trajex::totg::compat
  )
else()
  add_library(viam-trajex-totg-compat INTERFACE)
  add_library(viam-trajex-totg-compat-shim INTERFACE)
endif()

add_library(viam::trajex::totg::compat ALIAS viam-trajex-totg-compat)
add_library(viam::trajex::totg::compat::shim ALIAS viam-trajex-totg-compat-shim)


# MoveIt integration: a MoveIt `TimeParameterization` backed by trajex TOTG. Its header includes both MoveIt's and
# trajex's, so both are public dependencies. Only contains meaningful content if `VIAM_TRAJEX_ENABLE_MOVEIT` is
# truthy.

if (VIAM_TRAJEX_ENABLE_MOVEIT)
  add_library(viam-trajex-moveit-totg)

  target_sources(viam-trajex-moveit-totg
    PRIVATE
    src/viam/trajex/moveit/totg/time_parameterization.cpp
  )

  target_sources(viam-trajex-moveit-totg
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES
    src/viam/trajex/moveit/totg/time_parameterization.hpp
  )

  target_link_libraries(viam-trajex-moveit-totg
    PUBLIC
    viam::trajex::totg
    moveit_core::moveit_robot_trajectory
    moveit_core::moveit_trajectory_processing
    PRIVATE
    moveit_core::moveit_robot_model
    moveit_core::moveit_robot_state
    moveit_core::moveit_utils
  )

  apply_wall_werror(viam-trajex-moveit-totg)
else()
  add_library(viam-trajex-moveit-totg INTERFACE)
endif()

add_library(viam::trajex::moveit::totg ALIAS viam-trajex-moveit-totg)


# Tools-legacy library: colinearization helpers in the `viam::trajex::totg::legacy` namespace,
# used only by the legacy generator code path. Only contains meaningful content if both
# `VIAM_TRAJEX_ENABLE_TOOLS` and `VIAM_TRAJEX_ENABLE_LEGACY` are truthy.

if (VIAM_TRAJEX_ENABLE_TOOLS AND VIAM_TRAJEX_ENABLE_LEGACY)
  add_library(viam-trajex-totg-tools-legacy)

  target_sources(viam-trajex-totg-tools-legacy
    PRIVATE
    src/viam/trajex/totg/tools/legacy.cpp
  )

  target_include_directories(viam-trajex-totg-tools-legacy
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  target_link_libraries(viam-trajex-totg-tools-legacy
    PUBLIC
    Eigen3::Eigen
  )

  apply_wall_werror(viam-trajex-totg-tools-legacy)
else()
  add_library(viam-trajex-totg-tools-legacy INTERFACE)
endif()

add_library(viam::trajex::totg::tools::legacy ALIAS viam-trajex-totg-tools-legacy)


# Support and tooling for new TOTG and legacy. Only contains meaningful content if
# `VIAM_TRAJEX_ENABLE_TOOLS` is truthy.

if (VIAM_TRAJEX_ENABLE_TOOLS)
  add_library(viam-trajex-totg-tools)

  target_sources(viam-trajex-totg-tools
    PRIVATE
      src/viam/trajex/totg/tools/json_serialization.cpp
      src/viam/trajex/totg/tools/planner.cpp
      src/viam/trajex/totg/tools/replay.cpp
  )

  target_link_libraries(viam-trajex-totg-tools
    PUBLIC
      viam::trajex::totg
      viam::trajex::totg::legacy
      viam::trajex::totg::tools::legacy
    PRIVATE
      jsoncpp_lib
  )

  apply_wall_werror(viam-trajex-totg-tools)
else()
  add_library(viam-trajex-totg-tools INTERFACE)
endif()

add_library(viam::trajex::totg::tools ALIAS viam-trajex-totg-tools)


# Service layer: SDK-facing MLModel service. Only layer that depends on viam-cpp-sdk. Only
# contains meaningful content if `VIAM_TRAJEX_ENABLE_SERVICE` is truthy.

if (VIAM_TRAJEX_ENABLE_SERVICE)
  if (NOT VIAM_TRAJEX_ENABLE_TOOLS)
    message(FATAL_ERROR "VIAM_TRAJEX_ENABLE_SERVICE=ON requires VIAM_TRAJEX_ENABLE_TOOLS=ON")
  endif()

  add_library(viam-trajex-service)

  target_sources(viam-trajex-service
    PRIVATE
      src/viam/trajex/service/mlmodel.cpp
  )

  target_include_directories(viam-trajex-service
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  # NOTE: viam::trajex::totg::tools is linked PUBLIC rather than PRIVATE so that the
  # `VIAM_TRAJEX_LEGACY_ENABLED` compile definition (set INTERFACE on the legacy library
  # and re-exported by tools) reaches viam-trajex-service-test. Without it, service's
  # public header `mlmodel.hpp` would compile with a different `config::generator_sequence`
  # default initializer in the test TU than in service's own TU -- a latent ODR mismatch
  # and a source of silently-skipped tests in the LEGACY-conditional branches. Strictly
  # speaking the service is not "really" a library for external consumers (it exists so
  # service-exe and service-test can both link it), so the PRIVATE/PUBLIC compile
  # firewall isn't load-bearing here. Once the legacy generator is removed entirely,
  # this can revert to PRIVATE.
  target_link_libraries(viam-trajex-service
    PUBLIC
      viam-cpp-sdk::viamsdk
      viam::trajex::totg::tools
  )

  apply_wall_werror(viam-trajex-service)
else()
  add_library(viam-trajex-service INTERFACE)
endif()

add_library(viam::trajex::service ALIAS viam-trajex-service)


# C ABI shim layer: language-agnostic boundary for non-C++ callers (cgo / FFI). Only
# contains meaningful content if `VIAM_TRAJEX_ENABLE_CAPI` is truthy. Links only the
# core TOTG library; no fallback/orchestration is needed (TOTG handles reversals
# natively), so the planner abstraction is sidestepped and the tools layer is not a
# dependency.
if (VIAM_TRAJEX_ENABLE_CAPI)
  add_library(viam-trajex-capi)
  target_sources(viam-trajex-capi
    PRIVATE
      src/viam/trajex/capi/capi.cpp
    PUBLIC
      FILE_SET HEADERS
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
      FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/viam/trajex/capi/capi.h
  )
  target_include_directories(viam-trajex-capi
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_link_libraries(viam-trajex-capi
    PRIVATE
      viam::trajex::totg
  )
  apply_wall_werror(viam-trajex-capi)
else()
  add_library(viam-trajex-capi INTERFACE)
endif()

add_library(viam::trajex::capi ALIAS viam-trajex-capi)


#
# Service executable target
#

if (VIAM_TRAJEX_ENABLE_SERVICE)
  add_executable(viam-trajex-service-exe)

  target_sources(viam-trajex-service-exe
    PRIVATE
      src/viam/trajex/service/main.cpp
  )

  set_target_properties(viam-trajex-service-exe PROPERTIES OUTPUT_NAME viam-trajex-service)

  target_link_libraries(viam-trajex-service-exe
    PRIVATE
      viam::trajex::service
  )

  apply_wall_werror(viam-trajex-service-exe)
endif()


#
# Tools
#

if (VIAM_TRAJEX_ENABLE_TOOLS)
  add_executable(viam-trajex-totg-replay-exe)

  target_sources(viam-trajex-totg-replay-exe
    PRIVATE
      src/viam/trajex/totg/tools/replay_main.cpp
  )

  set_target_properties(viam-trajex-totg-replay-exe PROPERTIES OUTPUT_NAME viam-trajex-totg-replay)

  target_link_libraries(viam-trajex-totg-replay-exe
    PRIVATE
      viam::trajex::totg::tools
  )

  apply_wall_werror(viam-trajex-totg-replay-exe)
endif()


#
# Benchmarks
#

# Deliberately not registered with ctest. Benchmarks are run by hand, compared with
# google-benchmark's `compare.py`, and would only make the test loop slower and noisier.
if (VIAM_TRAJEX_BUILD_BENCHMARKS)
  if (NOT VIAM_TRAJEX_ENABLE_TOOLS)
    message(FATAL_ERROR "VIAM_TRAJEX_BUILD_BENCHMARKS requires VIAM_TRAJEX_ENABLE_TOOLS (replay record parsing)")
  endif()

  add_executable(viam-trajex-totg-bench)

  target_sources(viam-trajex-totg-bench
    PRIVATE
      src/viam/trajex/totg/bench/main.cpp
      src/viam/trajex/totg/bench/pipeline.cpp
      src/viam/trajex/totg/bench/xtensor_idioms.cpp
  )

  target_link_libraries(viam-trajex-totg-bench
    PRIVATE
      viam::trajex::totg::tools
      benchmark::benchmark
  )

  target_compile_definitions(viam-trajex-totg-bench PRIVATE
    VIAM_TRAJEX_BENCH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/viam/trajex/totg/test/data"
  )

  apply_wall_werror(viam-trajex-totg-bench)

  # Instruments refuses to attach to a process whose binary does not carry
  # `get-task-allow`, so `xctrace record` launches the benchmark, fails with "Failed to
  # attach to target process", and still writes out a trace. The trace has no samples in it,
  # which is easy to mistake for a profiling result. Re-signing ad-hoc adds the entitlement
  # without weakening anything, since the linker already ad-hoc signs arm64 binaries, and
  # the benchmarks are neither installed nor shipped.
  if (APPLE)
    set(VIAM_TRAJEX_BENCH_ENTITLEMENTS
      "${CMAKE_CURRENT_BINARY_DIR}/viam-trajex-totg-bench.entitlements")

    file(WRITE "${VIAM_TRAJEX_BENCH_ENTITLEMENTS}"
[=[<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.get-task-allow</key>
    <true/>
</dict>
</plist>
]=])

    add_custom_command(TARGET viam-trajex-totg-bench POST_BUILD
      COMMAND codesign -s - -f
        --entitlements "${VIAM_TRAJEX_BENCH_ENTITLEMENTS}"
        $<TARGET_FILE:viam-trajex-totg-bench>
      COMMENT "Re-signing viam-trajex-totg-bench with get-task-allow for Instruments"
      VERBATIM
    )
  endif()
endif()


#
# Tests
#

if (VIAM_TRAJEX_BUILD_TESTS)

  enable_testing()

  # TODO: viam-trajex-totg-test currently depends on the tools library for opt-in JSON
  # trajectory dumping (json_serialization.hpp). Until that dependency is refactored to be
  # truly opt-in (e.g. separately compiled and conditionally linked), the test target is
  # only buildable when VIAM_TRAJEX_ENABLE_TOOLS is ON.
  if (VIAM_TRAJEX_ENABLE_TOOLS)
    add_executable(viam-trajex-totg-test)

    target_sources(viam-trajex-totg-test
      PRIVATE
      src/viam/trajex/totg/test/main.cpp
      src/viam/trajex/totg/test/test_utils.cpp
      src/viam/trajex/totg/test/waypoint_accumulator.cpp
      src/viam/trajex/totg/test/path_basic.cpp
      src/viam/trajex/totg/test/path_coalescing.cpp
      src/viam/trajex/totg/test/path_blends.cpp
      src/viam/trajex/totg/test/path_segments.cpp
      src/viam/trajex/totg/test/path_cursor.cpp
      src/viam/trajex/totg/test/trajectory.cpp
      src/viam/trajex/totg/test/uniform_sampler.cpp
      src/viam/trajex/totg/test/integration.cpp
      src/viam/trajex/totg/test/observers.cpp
      src/viam/trajex/totg/test/tcp_limits.cpp
      src/viam/trajex/kinematics/test/chain.cpp
    )

    target_link_libraries(viam-trajex-totg-test
      PRIVATE
      viam::trajex::totg::tools
      Boost::boost
      jsoncpp_lib
    )

    target_compile_definitions(viam-trajex-totg-test PRIVATE
      VIAM_TRAJEX_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/viam/trajex/totg/test/data"
    )

    apply_wall_werror(viam-trajex-totg-test)

    add_test(
      NAME viam-trajex-totg-test
      COMMAND viam-trajex-totg-test
    )


    # Separate binary for streaming session tests so they can be looped fast
    # without paying the integration suite's runtime.
    add_executable(viam-trajex-totg-streaming-test)

    target_sources(viam-trajex-totg-streaming-test
      PRIVATE
      src/viam/trajex/totg/streaming/test/main.cpp
      src/viam/trajex/totg/streaming/test/session.cpp
      src/viam/trajex/totg/streaming/test/waypoint_store.cpp
    )

    target_link_libraries(viam-trajex-totg-streaming-test
      PRIVATE
      viam::trajex::totg
      Boost::boost
    )

    target_compile_definitions(viam-trajex-totg-streaming-test PRIVATE
      VIAM_TRAJEX_STREAMING_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/viam/trajex/totg/streaming/test/data"
    )

    apply_wall_werror(viam-trajex-totg-streaming-test)

    add_test(
      NAME viam-trajex-totg-streaming-test
      COMMAND viam-trajex-totg-streaming-test
    )


    # Streaming-session simulator. Standalone binary, not wired into ctest --
    # see orbsanding-streaming-session-simulation-plan.md.
    add_executable(viam-trajex-totg-streaming-sim-exe)

    target_sources(viam-trajex-totg-streaming-sim-exe
      PRIVATE
      src/viam/trajex/totg/streaming/test/sim/streaming_sim_main.cpp
    )

    set_target_properties(viam-trajex-totg-streaming-sim-exe PROPERTIES
      OUTPUT_NAME viam-trajex-totg-streaming-sim
    )

    target_link_libraries(viam-trajex-totg-streaming-sim-exe
      PRIVATE
      viam::trajex::totg::tools
    )

    apply_wall_werror(viam-trajex-totg-streaming-sim-exe)


    add_executable(viam-trajex-totg-tools-test)

    target_sources(viam-trajex-totg-tools-test
      PRIVATE
      src/viam/trajex/totg/tools/test/main.cpp
      src/viam/trajex/totg/tools/test/planner.cpp
    )

    if (VIAM_TRAJEX_ENABLE_LEGACY)
      target_sources(viam-trajex-totg-tools-test
        PRIVATE
        src/viam/trajex/totg/tools/test/legacy.cpp
      )
    endif()

    target_link_libraries(viam-trajex-totg-tools-test
      PRIVATE
      viam::trajex::totg::tools
      Boost::boost
    )

    apply_wall_werror(viam-trajex-totg-tools-test)

    add_test(
      NAME viam-trajex-totg-tools-test
      COMMAND viam-trajex-totg-tools-test
    )
  endif()


  add_executable(viam-trajex-types-test)

  target_sources(viam-trajex-types-test
    PRIVATE
    src/viam/trajex/types/test.cpp
  )

  target_include_directories(viam-trajex-types-test
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  )

  # The headers under types/ include xtensor, so this needs the include path. It does not link
  # the trajex library, which is where every other test target picks that up.
  target_link_libraries(viam-trajex-types-test
    PRIVATE
    Boost::boost
    xtensor
    xtl
  )

  apply_wall_werror(viam-trajex-types-test)

  add_test(
    NAME viam-trajex-types-test
    COMMAND viam-trajex-types-test
  )


  if (VIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT)
    # Links the reference implementation too when it is enabled, so the suite can run the same
    # inputs through both.
    add_executable(viam-trajex-totg-compat-test)

    target_sources(viam-trajex-totg-compat-test
      PRIVATE
      src/viam/trajex/totg/compat/test/main.cpp
      src/viam/trajex/totg/compat/test/compat.cpp
    )

    target_link_libraries(viam-trajex-totg-compat-test
      PRIVATE
      viam::trajex::totg::compat
      viam::trajex::totg
      viam::trajex::totg::legacy
      Boost::boost
    )

    apply_wall_werror(viam-trajex-totg-compat-test)

    add_test(
      NAME viam-trajex-totg-compat-test
      COMMAND viam-trajex-totg-compat-test
    )

    # Source compatibility tripwires: the reference implementation's own programs, unmodified,
    # built against the shim. They are copied out of their directory first, because their quoted
    # includes would otherwise find the original headers sitting beside them. They are built
    # but not registered with ctest: both always exit zero, and the example waits on stdin.
    #
    # Built at the oldest standard the compat headers claim. CMake raises it to whatever the
    # Eigen target requires, so with Eigen 5 (C++14) this checks less than it says.
    foreach (program Example Test)
      string(TOLOWER ${program} program_lower)
      set(target viam-trajex-totg-compat-shim-${program_lower})

      configure_file(
        src/third_party/trajectories/${program}.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/compat-shim/${program}.cpp
        COPYONLY
      )

      add_executable(${target})

      target_sources(${target}
        PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/compat-shim/${program}.cpp
      )

      set_target_properties(${target} PROPERTIES
        CXX_STANDARD 98
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
      )

      target_link_libraries(${target}
        PRIVATE
        viam::trajex::totg::compat::shim
      )
    endforeach()

    # Test.cpp uses std::cout without including <iostream>, relying on older Eigen to provide
    # it. It fails the same way against the original headers with current Eigen, so supply the
    # include rather than edit the file.
    target_compile_options(viam-trajex-totg-compat-shim-test PRIVATE "SHELL:-include iostream")
  endif()


  if (VIAM_TRAJEX_ENABLE_MOVEIT)
    add_executable(viam-trajex-moveit-totg-test)

    target_sources(viam-trajex-moveit-totg-test
      PRIVATE
      src/viam/trajex/moveit/totg/test/main.cpp
      src/viam/trajex/moveit/totg/test/time_parameterization.cpp
    )

    target_link_libraries(viam-trajex-moveit-totg-test
      PRIVATE
      viam::trajex::moveit::totg
      moveit_core::moveit_robot_model
      moveit_core::moveit_robot_state
      moveit_core::moveit_test_utils
      Boost::boost
    )

    apply_wall_werror(viam-trajex-moveit-totg-test)

    add_test(
      NAME viam-trajex-moveit-totg-test
      COMMAND viam-trajex-moveit-totg-test
    )
  endif()


  if (VIAM_TRAJEX_ENABLE_SERVICE)
    add_executable(viam-trajex-service-test)

    target_sources(viam-trajex-service-test
      PRIVATE
      src/viam/trajex/service/test/mlmodel.cpp
    )

    target_link_libraries(viam-trajex-service-test
      PRIVATE
      viam::trajex::service
      Boost::boost
    )

    apply_wall_werror(viam-trajex-service-test)

    add_test(
      NAME viam-trajex-service-test
      COMMAND viam-trajex-service-test
    )
  endif()

  # C ABI compile-smoke: a tiny .c file that references every exported
  # symbol via address-take. Build failure of this target is the early
  # tripwire for the C-ABI contract -- either the header stopped being
  # C-compilable, or a symbol declared in the header is no longer
  # exported by the library.
  if (VIAM_TRAJEX_ENABLE_CAPI)
    add_executable(viam-trajex-capi-compile-smoke)

    target_sources(viam-trajex-capi-compile-smoke
      PRIVATE
      src/viam/trajex/capi/test/capi_compile_smoke.c
    )

    set_target_properties(viam-trajex-capi-compile-smoke PROPERTIES
      C_STANDARD 99
      C_STANDARD_REQUIRED ON
      C_EXTENSIONS OFF
    )

    target_link_libraries(viam-trajex-capi-compile-smoke
      PRIVATE
      viam::trajex::capi
    )

    apply_wall_werror(viam-trajex-capi-compile-smoke)

    add_test(
      NAME viam-trajex-capi-compile-smoke
      COMMAND viam-trajex-capi-compile-smoke
    )

    add_executable(viam-trajex-capi-test)

    target_sources(viam-trajex-capi-test
      PRIVATE
      src/viam/trajex/capi/test/main.cpp
    )

    target_link_libraries(viam-trajex-capi-test
      PRIVATE
      viam::trajex::capi
      Boost::boost
    )

    apply_wall_werror(viam-trajex-capi-test)

    add_test(
      NAME viam-trajex-capi-test
      COMMAND viam-trajex-capi-test
    )
  endif()

  # The find_package consumer's programs, also built here against this build's own targets. That puts
  # their sources in the compile database for the analysers, and breaks the build as soon as they stop
  # compiling. Only the consumer project itself, built by .github/scripts/verify-package.sh against an
  # install, tests the package. Its directory is never added here: its project finds trajex with
  # find_package, which must only run outside this build.
  function(viam_trajex_add_consumer_test name source library)
    add_executable(viam-trajex-consumer-${name})
    target_sources(viam-trajex-consumer-${name} PRIVATE src/viam/trajex/test/cmake/${source})
    target_link_libraries(viam-trajex-consumer-${name} PRIVATE ${library})
    apply_wall_werror(viam-trajex-consumer-${name})
    add_test(NAME viam-trajex-consumer-${name} COMMAND viam-trajex-consumer-${name})
  endfunction()

  viam_trajex_add_consumer_test(totg totg.cpp viam::trajex::totg)

  if (VIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT)
    viam_trajex_add_consumer_test(compat compat.cpp viam::trajex::totg::compat::shim)
  endif()

  if (VIAM_TRAJEX_ENABLE_CAPI)
    viam_trajex_add_consumer_test(capi capi.c viam::trajex::capi)
    set_target_properties(viam-trajex-consumer-capi PROPERTIES
      C_STANDARD 99
      C_STANDARD_REQUIRED ON
      C_EXTENSIONS OFF
    )
  endif()

  if (VIAM_TRAJEX_ENABLE_MOVEIT)
    viam_trajex_add_consumer_test(moveit moveit.cpp viam::trajex::moveit::totg)
  endif()

endif()

#
# Installation
#

# Every install rule names its component, in two roles: `<component>-runtime` for what a program
# needs to run (executables and shared libraries) and `<component>-dev` for what is needed to build
# against it (headers, static archives, and shared library namelinks). Static archives go to runtime
# instead when VIAM_TRAJEX_INSTALL_ARCHIVES_AS_RUNTIME is set.
#
# `cmake --install --component <name>` installs exactly one component. CMake has no notion of one
# component depending on another, so a consumer names everything it needs: dev implies runtime, and
# a component implies the components of the libraries it links. A full install, with no
# --component, installs every component, as do the module package and the ament package. When xtl
# and xtensor are fetched rather than found, their own install rules are in the
# trajex-vendored-dependencies-dev component.
#
# Each library's public headers are its HEADERS file set, installed with their paths under src
# preserved, so that `#include <viam/trajex/...>` works against the install tree as it does against
# the source tree. A target without one installs no headers.
#
# With EXPORT, and VIAM_TRAJEX_INSTALL_CMAKE_PACKAGE set, the targets are also exported for the
# package config, into one export file per component, installed in the component's dev role. The
# exported names are the alias names: `viam-trajex-totg-compat` exports as
# `viam::trajex::totg::compat`. Only libraries with installed public headers are worth exporting.
set(VIAM_TRAJEX_CMAKE_PACKAGE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/trajex)

function(viam_trajex_install component)
  cmake_parse_arguments(PARSE_ARGV 1 arg "EXPORT" "" "")
  set(targets ${arg_UNPARSED_ARGUMENTS})

  if (VIAM_TRAJEX_INSTALL_ARCHIVES_AS_RUNTIME)
    set(archive_role runtime)
  else()
    set(archive_role dev)
  endif()

  set(export_args)
  if (arg_EXPORT AND VIAM_TRAJEX_INSTALL_CMAKE_PACKAGE)
    foreach(target IN LISTS targets)
      string(REGEX REPLACE "^viam-trajex-" "" export_name ${target})
      string(REPLACE "-" "::" export_name ${export_name})
      set_target_properties(${target} PROPERTIES EXPORT_NAME ${export_name})
    endforeach()
    set(export_args EXPORT ${component}-targets)
  endif()

  install(
    TARGETS ${targets} ${export_args}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${component}-runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${component}-runtime NAMELINK_COMPONENT ${component}-dev
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${component}-${archive_role}
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT ${component}-dev
  )

  if (export_args)
    install(
      EXPORT ${component}-targets
      NAMESPACE viam::trajex::
      DESTINATION ${VIAM_TRAJEX_CMAKE_PACKAGE_INSTALL_DIR}
      COMPONENT ${component}-dev
    )
  endif()
endfunction()

# The legacy target is always installed; when VIAM_TRAJEX_ENABLE_LEGACY is OFF it is an empty
# INTERFACE library and installs nothing. It has no public headers.
viam_trajex_install(trajex-legacy viam-trajex-totg-legacy)

viam_trajex_install(trajex-totg EXPORT viam-trajex-totg)

if (VIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT)
  viam_trajex_install(trajex-compat EXPORT viam-trajex-totg-compat viam-trajex-totg-compat-shim)
endif()

if (VIAM_TRAJEX_ENABLE_TOOLS)
  viam_trajex_install(trajex-tools viam-trajex-totg-tools-legacy viam-trajex-totg-tools viam-trajex-totg-replay-exe)
endif()

if (VIAM_TRAJEX_ENABLE_CAPI)
  viam_trajex_install(trajex-capi EXPORT viam-trajex-capi)
endif()

if (VIAM_TRAJEX_ENABLE_MOVEIT)
  viam_trajex_install(trajex-moveit EXPORT viam-trajex-moveit-totg)
endif()

# The package config, which sits beside the totg export in trajex-totg-dev, since every component
# needs totg. It finds xtl and xtensor for its consumers the way this build did. Found ones are
# found again, at the version found here or newer, or failing a reported version, at our minimum.
# Fetched ones were installed alongside trajex, so it finds exactly those, in the same prefix.
if (VIAM_TRAJEX_INSTALL_CMAKE_PACKAGE)
  include(CMakePackageConfigHelpers)

  foreach(dependency IN ITEMS xtl xtensor)
    string(TOUPPER ${dependency} dependency_upper)
    get_target_property(dependency_imported ${dependency} IMPORTED)
    if (dependency_imported)
      set(dependency_version ${${dependency}_VERSION})
      if (NOT dependency_version)
        set(dependency_version ${VIAM_TRAJEX_${dependency_upper}_VERSION_MINIMUM})
      endif()
      set(find_args "${dependency_version} CONFIG")
    else()
      get_directory_property(dependency_version DIRECTORY ${${dependency}_SOURCE_DIR} DEFINITION ${dependency}_VERSION)
      set(dependency_config_dir ${${dependency_upper}_CMAKECONFIG_INSTALL_DIR})
      if (NOT dependency_version OR NOT dependency_config_dir)
        message(FATAL_ERROR "Cannot tell which version of the fetched ${dependency} is installed, or where its package config goes")
      endif()
      if (NOT IS_ABSOLUTE "${dependency_config_dir}")
        set(dependency_config_dir "\${PACKAGE_PREFIX_DIR}/${dependency_config_dir}")
      endif()
      set(find_args "${dependency_version} EXACT CONFIG PATHS \"${dependency_config_dir}\" NO_DEFAULT_PATH")
    endif()
    set(VIAM_TRAJEX_${dependency_upper}_FIND_ARGS ${find_args})
  endforeach()

  configure_package_config_file(
    cmake/trajexConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/trajexConfig.cmake
    INSTALL_DESTINATION ${VIAM_TRAJEX_CMAKE_PACKAGE_INSTALL_DIR}
  )

  # trajex is at major version 0, where a minor version may break compatibility.
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/trajexConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
  )

  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/trajexConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/trajexConfigVersion.cmake
    DESTINATION ${VIAM_TRAJEX_CMAKE_PACKAGE_INSTALL_DIR}
    COMPONENT trajex-totg-dev
  )
endif()

if (VIAM_TRAJEX_ENABLE_SERVICE)
  # The service library and executable, and the module packaging artifacts (meta.json + CPack).
  viam_trajex_install(trajex-service viam-trajex-service viam-trajex-service-exe)

  # Configure meta.json from template
  configure_file(
    meta.json.in
    ${CMAKE_CURRENT_BINARY_DIR}/meta.json
    @ONLY
  )

  # Install meta.json at module root
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/meta.json
    DESTINATION .
    COMPONENT trajex-service-runtime
  )

  # Set up CPack for trajex module packaging
  set(CPACK_PACKAGE_NAME "viam-trajex-service")
  set(CPACK_PACKAGE_FILE_NAME "module")
  set(CPACK_GENERATOR "TGZ")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY 0)
  include(CPack)
endif()
