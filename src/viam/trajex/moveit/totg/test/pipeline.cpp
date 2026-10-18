// Tests the adapters end to end through MoveIt's own planning pipeline. The parameters come from a ROS params file
// laid out like a user's pipeline configuration. The pipeline loads OMPL and a trajex response adapter by name, OMPL
// plans a motion, and the adapter times it.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_interface/planning_response.hpp>
#include <moveit/planning_pipeline/planning_pipeline.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/utils/robot_model_test_utils.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <rclcpp/rclcpp.hpp>

namespace {

namespace mc = ::moveit::core;
using error_codes = moveit_msgs::msg::MoveItErrorCodes;

class pipeline_test : public testing::Test {
   protected:
    // MoveIt's testing model has no acceleration limits in its URDF, where a real configuration would supply them
    // through joint_limits.yaml. We add them here, as MoveIt's own TOTG tests do.
    void SetUp() override {
        model_ = mc::loadTestingRobotModel("panda");
        ASSERT_TRUE(model_);
        for (auto* joint : model_->getActiveJointModels()) {
            auto bounds = joint->getVariableBoundsMsg();
            for (auto& bound : bounds) {
                bound.has_acceleration_limits = true;
                bound.max_acceleration = 1.0;
            }
            joint->setVariableBounds(bounds);
        }
        node_ = std::make_shared<rclcpp::Node>(
            "pipeline_test",
            rclcpp::NodeOptions().allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true).arguments(
                {"--ros-args", "--params-file", VIAM_TRAJEX_MOVEIT_TEST_DATA_DIR "/pipelines.yaml"}));
        scene_ = std::make_shared<planning_scene::PlanningScene>(model_);
    }

    // Plans the Panda arm from its `ready` pose to its `extended` pose through the named pipeline.
    //
    // The fixture keeps the pipeline, as move_group keeps its pipelines for the life of the process. Destroying the
    // pipeline unloads its plugin libraries, and since OMPL's library made the response's trajectory, releasing the
    // trajectory after that would run code that is no longer mapped.
    planning_interface::MotionPlanResponse plan(const std::string& pipeline_name) {
        pipeline_ = std::make_shared<planning_pipeline::PlanningPipeline>(model_, node_, pipeline_name);
        const auto& pipeline = *pipeline_;
        const auto* group = model_->getJointModelGroup("panda_arm");

        mc::RobotState start(model_);
        EXPECT_TRUE(start.setToDefaultValues(group, "ready"));
        goal_ = std::make_shared<mc::RobotState>(model_);
        EXPECT_TRUE(goal_->setToDefaultValues(group, "extended"));

        planning_interface::MotionPlanRequest request;
        request.group_name = "panda_arm";
        request.planner_id = "RRTConnectkConfigDefault";
        request.allowed_planning_time = 10.0;
        request.num_planning_attempts = 1;
        request.max_velocity_scaling_factor = 1.0;
        request.max_acceleration_scaling_factor = 1.0;
        mc::robotStateToRobotStateMsg(start, request.start_state);
        request.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(*goal_, group));

        planning_interface::MotionPlanResponse response;
        EXPECT_TRUE(pipeline.generatePlan(scene_, request, response));
        return response;
    }

    // The trajex adapters space waypoints uniformly, no further apart than the configured period. MoveIt's TOTG would
    // space them at its default 0.1 s with a shorter final step, so this tells us that trajex timed the plan.
    void expect_timed_by_trajex(const planning_interface::MotionPlanResponse& response, double period) const {
        ASSERT_EQ(response.error_code.val, error_codes::SUCCESS);
        ASSERT_TRUE(response.trajectory);
        const auto& trajectory = *response.trajectory;
        ASSERT_GE(trajectory.getWayPointCount(), 3U);
        EXPECT_GT(trajectory.getDuration(), 0.0);

        const double spacing = trajectory.getWayPointDurationFromPrevious(1);
        EXPECT_LE(spacing, period + 1e-9);
        EXPECT_GT(spacing, period / 2.0);
        for (std::size_t w = 1; w < trajectory.getWayPointCount(); ++w) {
            EXPECT_NEAR(trajectory.getWayPointDurationFromPrevious(w), spacing, 1e-9);
        }

        // OMPL meets the goal within its joint tolerance, and the adapter keeps the path's final waypoint.
        const auto* group = model_->getJointModelGroup("panda_arm");
        Eigen::VectorXd end;
        Eigen::VectorXd goal;
        trajectory.getLastWayPoint().copyJointGroupPositions(group, end);
        goal_->copyJointGroupPositions(group, goal);
        EXPECT_LE((end - goal).cwiseAbs().maxCoeff(), 1e-3);
        for (const auto& name : group->getVariableNames()) {
            EXPECT_NEAR(trajectory.getLastWayPoint().getVariableVelocity(name), 0.0, 1e-9);
        }
    }

    mc::RobotModelPtr model_;
    rclcpp::Node::SharedPtr node_;
    planning_scene::PlanningScenePtr scene_;
    mc::RobotStatePtr goal_;
    planning_pipeline::PlanningPipelinePtr pipeline_;
};

TEST_F(pipeline_test, trajex_adapter_times_an_ompl_plan_with_parameters_from_yaml) {
    const auto response = plan("ompl_trajex");
    expect_timed_by_trajex(response, 1.0 / 37.0);
    EXPECT_EQ(node_->get_parameter("ompl_trajex.trajex.totg.sampling.frequency").as_double(), 37.0);
    EXPECT_TRUE(node_->has_parameter("ompl_trajex.trajex.totg.path.max_blend_deviation"));
}

TEST_F(pipeline_test, compat_adapter_times_an_ompl_plan_with_moveits_parameters_from_yaml) {
    const auto response = plan("ompl_trajex_compat");
    expect_timed_by_trajex(response, 0.05);
    EXPECT_EQ(node_->get_parameter("ompl_trajex_compat.totg.resample_dt").as_double(), 0.05);
    EXPECT_TRUE(node_->has_parameter("ompl_trajex_compat.totg.path_tolerance"));
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    const int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
