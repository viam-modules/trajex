# Registers every test case of a Boost.Test executable as its own ctest test, so that `ctest -j`
# can spread a slow suite across cores and schedule the slowest cases first from the costs it
# records. Data test case samples are registered individually, since a single data test case can
# carry most of a suite's runtime.
#
# This file plays two roles. Included from a CMakeLists.txt, it defines
# `boost_test_discover_tests(target)`. Run with `cmake -P`, which that function arranges after each
# link of the target, it lists the executable's test tree with `--list_content` and writes one
# `add_test` per leaf test unit, invoking the executable with `--run_test=<path>` for that leaf.
# Discovering after the build rather than at configure time means adding a test never requires a
# reconfigure, and the registrations never disagree with the binary they run.

if (NOT CMAKE_SCRIPT_MODE_FILE)
  set(_boost_test_discovery_script "${CMAKE_CURRENT_LIST_FILE}")

  function(boost_test_discover_tests target)
    set(tests_file "${CMAKE_CURRENT_BINARY_DIR}/${target}_tests.cmake")
    set(include_file "${CMAKE_CURRENT_BINARY_DIR}/${target}_include.cmake")

    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND "${CMAKE_COMMAND}"
        -D "TEST_EXECUTABLE=$<TARGET_FILE:${target}>"
        -D "TEST_PREFIX=${target}"
        -D "TESTS_FILE=${tests_file}"
        -P "${_boost_test_discovery_script}"
      VERBATIM
    )

    # ctest reads this at test time. Until the first build has produced the discovered tests, a
    # placeholder test that cannot run makes the missing build visible rather than silently
    # running nothing.
    file(WRITE "${include_file}"
      "if (EXISTS [==[${tests_file}]==])\n"
      "  include([==[${tests_file}]==])\n"
      "else()\n"
      "  add_test([==[${target}_NOT_BUILT]==] [==[${target}_NOT_BUILT]==])\n"
      "endif()\n"
    )
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${include_file}")
  endfunction()

  return()
endif()

# Script mode starts with every policy unset, so without this an older CMake reads the script with its
# oldest behaviors, under which `IN_LIST` below is not an operator. Match the project's minimum.
cmake_minimum_required(VERSION 3.25)

execute_process(
  COMMAND "${TEST_EXECUTABLE}" --list_content
  OUTPUT_VARIABLE listing
  ERROR_VARIABLE listing
  RESULT_VARIABLE result
)
if (NOT result EQUAL 0)
  message(FATAL_ERROR "${TEST_EXECUTABLE} --list_content failed (${result}):\n${listing}")
endif()

# Each line of the listing is one test unit, indented four spaces per level of nesting, followed by
# `*` if the unit is enabled. A unit is a leaf when the line after it is not nested beneath it.
string(REPLACE ";" "\\;" listing "${listing}")
string(REPLACE "\n" ";" lines "${listing}")
set(depths)
set(names)
set(enables)
foreach (line IN LISTS lines)
  if (NOT line MATCHES "^( *)([^ *]+)([* ]?)$")
    continue()
  endif()
  string(LENGTH "${CMAKE_MATCH_1}" indent)
  math(EXPR depth "${indent} / 4")
  list(APPEND depths ${depth})
  list(APPEND names "${CMAKE_MATCH_2}")
  if (CMAKE_MATCH_3 STREQUAL "*")
    list(APPEND enables 1)
  else()
    list(APPEND enables 0)
  endif()
endforeach()

list(LENGTH names count)
if (count EQUAL 0)
  message(FATAL_ERROR "${TEST_EXECUTABLE} --list_content listed no test units:\n${listing}")
endif()

# A unit runs only if it and all of its ancestors are enabled, so carry the path and the
# enablement of the ancestors down as we walk.
set(content "")
set(path_stack)
set(enabled_stack)
math(EXPR last "${count} - 1")
foreach (i RANGE ${last})
  list(GET depths ${i} depth)
  list(GET names ${i} name)
  list(GET enables ${i} enabled)

  if (depth GREATER 0)
    list(SUBLIST path_stack 0 ${depth} path_stack)
    list(SUBLIST enabled_stack 0 ${depth} enabled_stack)
  else()
    set(path_stack)
    set(enabled_stack)
  endif()
  list(APPEND path_stack "${name}")
  list(APPEND enabled_stack ${enabled})

  set(is_leaf TRUE)
  if (i LESS last)
    math(EXPR next "${i} + 1")
    list(GET depths ${next} next_depth)
    if (next_depth GREATER depth)
      set(is_leaf FALSE)
    endif()
  endif()
  if (NOT is_leaf)
    continue()
  endif()

  list(JOIN path_stack "/" test_path)
  set(test_name "${TEST_PREFIX}/${test_path}")
  string(APPEND content
    "add_test([==[${test_name}]==] [==[${TEST_EXECUTABLE}]==] [==[--run_test=${test_path}]==])\n")
  if (0 IN_LIST enabled_stack)
    string(APPEND content "set_tests_properties([==[${test_name}]==] PROPERTIES DISABLED TRUE)\n")
  endif()
endforeach()

file(WRITE "${TESTS_FILE}" "${content}")
