cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# The ament package for trajex's MoveIt integration. colcon and ament want to own the build they run, so the ROS
# shaped part of the repository lives here and pulls in trajex from the repository root, rather than trajex
# becoming a ROS project. Build with colcon, pointing it at this directory.

project(viam_trajex_moveit VERSION 0.1.0 LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)

# trajex is C++20. Set it here: trajex only chooses a standard when its parent has not, and a ROS parent typically
# chooses C++17.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ROS packages install shared libraries, found at run time through the workspace environment.
set(BUILD_SHARED_LIBS ON)

# trajex, with its MoveIt integration and nothing else optional, and its CMake package installed into the workspace.
# These are plain variables, which trajex's options honor.
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

# Loads the adapters by name, as MoveIt does, so it runs against the installed package and its ament index entry.
if (BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

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

  # End to end through MoveIt's planning pipeline, with OMPL as the planner, configured from a params file.
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

# trajex installs its own CMake package into the workspace, so `find_package(viam_trajex_moveit)` finds it too, and
# downstream packages link its targets, such as `viam::trajex::moveit::totg`, directly.
ament_export_dependencies(trajex)

ament_package()
