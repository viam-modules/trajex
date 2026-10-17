cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# The ament package for trajex's MoveIt integration. Build it with colcon, pointed at this directory. colcon and
# ament expect to own the build, so rather than turning trajex into a ROS project, we keep the ROS specific parts
# here and pull in trajex from the repository root.

project(viam_trajex_moveit VERSION 0.1.0 LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)

# trajex only picks a C++ standard when its parent hasn't, and ROS packages usually pick C++17, so we ask for C++20
# here.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ROS packages install shared libraries, which are found at run time through the workspace environment.
set(BUILD_SHARED_LIBS ON)

# Build trajex with its MoveIt integration and nothing else optional, and install its CMake package into the
# workspace. Plain variables are enough, since trajex's options honor them.
set(VIAM_TRAJEX_INSTALL_CMAKE_PACKAGE ON)
set(VIAM_TRAJEX_ENABLE_MOVEIT ON)
set(VIAM_TRAJEX_ENABLE_LEGACY OFF)
set(VIAM_TRAJEX_TOTG_ENABLE_LEGACY_COMPAT OFF)
set(VIAM_TRAJEX_ENABLE_CAPI OFF)
set(VIAM_TRAJEX_ENABLE_TOOLS OFF)
set(VIAM_TRAJEX_ENABLE_SERVICE OFF)
set(VIAM_TRAJEX_BUILD_TESTS OFF)
set(VIAM_TRAJEX_BUILD_BENCHMARKS OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../.. trajex)

add_library(viam_trajex_moveit_plugins
  totg/response_adapter.cpp
)

target_link_libraries(viam_trajex_moveit_plugins
  PRIVATE
  viam::trajex::moveit::totg
  moveit_core::moveit_planning_interface
  moveit_core::moveit_utils
  pluginlib::pluginlib
  rclcpp::rclcpp
)

install(
  TARGETS viam_trajex_moveit_plugins
  LIBRARY DESTINATION lib
)

pluginlib_export_plugin_description_file(moveit_core plugins.xml)

if (BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # This test loads the adapters by name, as MoveIt does, so it runs against the installed package and its ament
  # index entry.
  ament_add_gtest(viam_trajex_moveit_response_adapter_test
    totg/test/response_adapter.cpp
  )

  target_link_libraries(viam_trajex_moveit_response_adapter_test
    viam::trajex::totg
    moveit_core::moveit_planning_interface
    moveit_core::moveit_robot_model
    moveit_core::moveit_robot_state
    moveit_core::moveit_robot_trajectory
    moveit_core::moveit_test_utils
    pluginlib::pluginlib
    rclcpp::rclcpp
  )

  # This test runs end to end through MoveIt's planning pipeline, with OMPL as the planner, configured from a params
  # file.
  find_package(moveit_ros_planning REQUIRED)

  ament_add_gtest(viam_trajex_moveit_pipeline_test
    totg/test/pipeline.cpp
  )

  target_compile_definitions(viam_trajex_moveit_pipeline_test PRIVATE
    VIAM_TRAJEX_MOVEIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/totg/test"
  )

  target_link_libraries(viam_trajex_moveit_pipeline_test
    moveit_ros_planning::moveit_planning_pipeline
    moveit_core::moveit_kinematic_constraints
    moveit_core::moveit_planning_interface
    moveit_core::moveit_planning_scene
    moveit_core::moveit_robot_model
    moveit_core::moveit_robot_state
    moveit_core::moveit_test_utils
    rclcpp::rclcpp
  )
endif()

# trajex installs its own CMake package into the workspace. Exporting it as a dependency lets
# `find_package(viam_trajex_moveit)` find it too, so downstream packages can link targets like
# `viam::trajex::moveit::totg` directly.
ament_export_dependencies(trajex)

ament_package()
