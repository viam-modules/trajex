# A project outside the trajex build that finds an installed trajex as any consumer would, and runs a small program
# against each component it asks for. `.github/scripts/verify-package.sh` builds and runs it against fresh installs.
# The trajex build compiles the same programs against its own targets, but never adds this directory.

cmake_minimum_required(VERSION 3.25)

project(trajex-find-package LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(TRAJEX_COMPONENTS "" CACHE STRING "trajex components to require and exercise, besides totg (compat, capi, moveit)")
set(TRAJEX_VENDORED_PREFIX "" CACHE PATH "If set, xtl and xtensor must be found under this prefix, as vendored by trajex")

find_package(trajex REQUIRED COMPONENTS ${TRAJEX_COMPONENTS})
message(STATUS "trajex ${trajex_VERSION} from ${trajex_DIR}")
message(STATUS "xtl ${xtl_VERSION} from ${xtl_DIR}")
message(STATUS "xtensor ${xtensor_VERSION} from ${xtensor_DIR}")

# A vendored install must hand its consumers the xtl and xtensor installed with it, even when others are on the
# default search path.
if (TRAJEX_VENDORED_PREFIX)
  foreach(dependency IN ITEMS xtl xtensor)
    cmake_path(IS_PREFIX TRAJEX_VENDORED_PREFIX "${${dependency}_DIR}" NORMALIZE dependency_is_vendored)
    if (NOT dependency_is_vendored)
      message(FATAL_ERROR "${dependency} was found at ${${dependency}_DIR}, not under ${TRAJEX_VENDORED_PREFIX}")
    endif()
  endforeach()
endif()

enable_testing()

add_executable(totg totg.cpp)
target_link_libraries(totg PRIVATE viam::trajex::totg)
add_test(NAME totg COMMAND totg)

if ("compat" IN_LIST TRAJEX_COMPONENTS)
  add_executable(compat compat.cpp)
  target_link_libraries(compat PRIVATE viam::trajex::totg::compat::shim)
  add_test(NAME compat COMMAND compat)
endif()

# In C, to show that the C API's header and archives serve a C program. The archives' C++ dependencies come through
# the export, which also selects the C++ linker.
if ("capi" IN_LIST TRAJEX_COMPONENTS)
  add_executable(capi capi.c)
  target_link_libraries(capi PRIVATE viam::trajex::capi)
  add_test(NAME capi COMMAND capi)
endif()

if ("moveit" IN_LIST TRAJEX_COMPONENTS)
  add_executable(moveit moveit.cpp)
  target_link_libraries(moveit PRIVATE viam::trajex::moveit::totg)
  add_test(NAME moveit COMMAND moveit)
endif()
