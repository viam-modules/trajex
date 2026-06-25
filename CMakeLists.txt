cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(trajex VERSION 0.1.0
  DESCRIPTION "Viam trajectory execution library"
  HOMEPAGE_URL https://github.com/viam-modules/trajex
  LANGUAGES C CXX
)

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

# Enable the third-party reference implementation (`third_party/trajectories`) and the
# `viam::trajex::totg::legacy::` colinearization helpers used by the legacy generator
# path. Pulls in Eigen. Useful for cross-validating the new TOTG implementation.
option(VIAM_TRAJEX_ENABLE_LEGACY "Enable the reference implementation" ON)

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


#
# Dependencies
#

# Minimum required version for each external dependency. An empty value means
# we do not enforce a minimum; the package's `find_package` (or its
# `FIND_PACKAGE_ARGS`-mediated equivalent) accepts whatever is found.
set(VIAM_TRAJEX_BOOST_VERSION_MINIMUM "")    # TODO: Decide this
set(VIAM_TRAJEX_EIGEN3_VERSION_MINIMUM "")   # TODO: Decide this
set(VIAM_TRAJEX_JSONCPP_VERSION_MINIMUM "")  # TODO: Decide this
set(VIAM_TRAJEX_VIAMCPPSDK_VERSION_MINIMUM 0.38.0)
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
#   Eigen3         ENABLE_LEGACY (reference impl + colinearization helpers).
#   Boost          BUILD_TESTS (boost.test framework, consumed in header-only mode
#                  via `<boost/test/included/unit_test.hpp>`; we still declare the
#                  `unit_test_framework` component to validate it is installed.
#                  The service code path uses Boost headers too but receives them
#                  transitively through viam-cpp-sdk).
#   jsoncpp        ENABLE_TOOLS (planner / replay / serialization sources).

include(FetchContent)

if (VIAM_TRAJEX_ENABLE_SERVICE)
  FetchContent_Declare(
    viam-cpp-sdk
    GIT_REPOSITORY https://github.com/viamrobotics/viam-cpp-sdk
    GIT_TAG        releases/v0.38.0
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
  GIT_TAG        e0f00666d90086bb245ae73abb6123d0e2c1b30b # 0.7.2
  FIND_PACKAGE_ARGS ${VIAM_TRAJEX_XTL_VERSION_MINIMUM} CONFIG
)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
  GIT_TAG        9e8662ddad7fcd1a96969b96bf8955e59cf44978 # 0.24.3
  FIND_PACKAGE_ARGS ${VIAM_TRAJEX_XTENSOR_VERSION_MINIMUM} CONFIG
)

FetchContent_MakeAvailable(xtl xtensor)

if (VIAM_TRAJEX_ENABLE_LEGACY)
  find_package(Eigen3 ${VIAM_TRAJEX_EIGEN3_VERSION_MINIMUM} CONFIG REQUIRED)
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
    $<INSTALL_INTERFACE:include>
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
    src/viam/trajex/totg/observers.cpp
    src/viam/trajex/totg/path.cpp
    src/viam/trajex/totg/trajectory.cpp
    src/viam/trajex/totg/uniform_sampler.cpp
    src/viam/trajex/totg/waypoint_accumulator.cpp
    src/viam/trajex/totg/waypoint_utils.cpp
)

target_include_directories(viam-trajex-totg
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(viam-trajex-totg
  PUBLIC
    xtensor
    xtl
)

apply_wall_werror(viam-trajex-totg)

add_library(viam::trajex::totg ALIAS viam-trajex-totg)


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
    $<INSTALL_INTERFACE:include>
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
      $<INSTALL_INTERFACE:include>
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
  )
  target_include_directories(viam-trajex-capi
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<INSTALL_INTERFACE:include>
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

  target_link_libraries(viam-trajex-types-test
    PRIVATE
    Boost::boost
  )

  apply_wall_werror(viam-trajex-types-test)

  add_test(
    NAME viam-trajex-types-test
    COMMAND viam-trajex-types-test
  )


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

endif()

#
# Installation
#

include(GNUInstallDirs)

# Always install core libraries. The legacy alias is always installed; when
# VIAM_TRAJEX_ENABLE_LEGACY is OFF it is an empty INTERFACE library and install() is a no-op.
install(
  TARGETS
  viam-trajex-totg-legacy
  viam-trajex-totg
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if (VIAM_TRAJEX_ENABLE_TOOLS)
  install(
    TARGETS
    viam-trajex-totg-tools-legacy
    viam-trajex-totg-tools
    viam-trajex-totg-replay-exe
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

if (VIAM_TRAJEX_ENABLE_SERVICE)
  install(
    TARGETS
    viam-trajex-service
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

if (VIAM_TRAJEX_ENABLE_CAPI)
  install(
    TARGETS
    viam-trajex-capi
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

if (VIAM_TRAJEX_ENABLE_SERVICE)
  # Install the service executable and the module packaging artifacts (meta.json + CPack).
  install(
    TARGETS
    viam-trajex-service-exe
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

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
  )

  # Set up CPack for trajex module packaging
  set(CPACK_PACKAGE_NAME "viam-trajex-service")
  set(CPACK_PACKAGE_FILE_NAME "module")
  set(CPACK_GENERATOR "TGZ")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY 0)
  include(CPack)
endif()
