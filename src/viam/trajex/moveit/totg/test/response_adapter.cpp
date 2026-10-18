// Loads the response adapters the way MoveIt does, by name through pluginlib, and runs them over a plan.

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_interface/planning_response.hpp>
#include <moveit/planning_interface/planning_response_adapter.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/utils/robot_model_test_utils.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>

#include <viam/trajex/totg/path.hpp>

namespace {

namespace mc = ::moveit::core;
using adapter = planning_interface::PlanningResponseAdapter;
using error_codes = moveit_msgs::msg::MoveItErrorCodes;

constexpr auto k_trajex = "viam_trajex_moveit/AddTotgTimeParameterization";
constexpr auto k_compat = "viam_trajex_moveit/AddTotgTimeParameterizationCompat";
constexpr auto k_moveit = "default_planning_response_adapters/AddTimeOptimalParameterization";
constexpr auto k_namespace = "pipeline";

class response_adapter_test : public testing::Test {
   protected:
    // MoveIt's testing model has no acceleration limits in its URDF, so we add them, as MoveIt's own TOTG tests do.
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
    }

    static rclcpp::Node::SharedPtr node(const std::string& name, std::vector<rclcpp::Parameter> overrides = {}) {
        // These are the node options move_group uses, so overrides behave the way values from YAML do there.
        return std::make_shared<rclcpp::Node>(name,
                                              rclcpp::NodeOptions()
                                                  .allow_undeclared_parameters(true)
                                                  .automatically_declare_parameters_from_overrides(true)
                                                  .parameter_overrides(std::move(overrides)));
    }

    std::shared_ptr<adapter> load(const std::string& name, const rclcpp::Node::SharedPtr& node) {
        auto instance = loader_.createSharedInstance(name);
        instance->initialize(node, k_namespace);
        return instance;
    }

    planning_interface::MotionPlanResponse plan() const {
        const auto* group = model_->getJointModelGroup("panda_arm");
        auto trajectory = std::make_shared<robot_trajectory::RobotTrajectory>(model_, group);
        mc::RobotState state(model_);
        state.setToDefaultValues();
        for (const auto& waypoint : std::vector<std::vector<double>>{{-0.5, -1.5, 1.35, -2.51, -0.88, 0.63, 0.0},
                                                                     {0.0, -1.5, 1.4, -1.2, -1.0, -0.2, 0.0},
                                                                     {-0.5, -1.0, 1.35, -2.51, -0.88, 0.63, 0.0}}) {
            state.setJointGroupPositions(group, waypoint);
            trajectory->addSuffixWayPoint(state, 0.0);
        }
        planning_interface::MotionPlanResponse response;
        response.trajectory = trajectory;
        return response;
    }

    static planning_interface::MotionPlanRequest request() {
        planning_interface::MotionPlanRequest request;
        request.max_velocity_scaling_factor = 1.0;
        request.max_acceleration_scaling_factor = 1.0;
        return request;
    }

    // Runs an adapter over a fresh plan and returns the response.
    planning_interface::MotionPlanResponse run(const adapter& instance) const {
        auto response = plan();
        instance.adapt(nullptr, request(), response);
        return response;
    }

    static double max_spacing(const planning_interface::MotionPlanResponse& response) {
        double spacing = 0.0;
        for (std::size_t w = 0; w < response.trajectory->getWayPointCount(); ++w) {
            spacing = std::max(spacing, response.trajectory->getWayPointDurationFromPrevious(w));
        }
        return spacing;
    }

    mc::RobotModelPtr model_;
    pluginlib::ClassLoader<adapter> loader_{"moveit_core", "planning_interface::PlanningResponseAdapter"};
};

TEST_F(response_adapter_test, both_adapters_are_discoverable) {
    const auto classes = loader_.getDeclaredClasses();
    EXPECT_NE(std::find(classes.begin(), classes.end(), k_trajex), classes.end());
    EXPECT_NE(std::find(classes.begin(), classes.end(), k_compat), classes.end());
}

TEST_F(response_adapter_test, trajex_adapter_declares_trajex_defaults_and_times_a_plan) {
    const auto n = node("trajex_defaults");
    const auto instance = load(k_trajex, n);
    EXPECT_EQ(instance->getDescription(), "AddTotgTimeParameterization");

    const viam::trajex::totg::path::options defaults;
    EXPECT_EQ(n->get_parameter("pipeline.trajex.totg.path.max_blend_deviation").as_double(), defaults.max_blend_deviation());
    EXPECT_EQ(n->get_parameter("pipeline.trajex.totg.path.max_linear_deviation").as_double(), defaults.max_linear_deviation());
    EXPECT_TRUE(n->has_parameter("pipeline.trajex.totg.path.deduplication_tolerance"));
    EXPECT_TRUE(n->has_parameter("pipeline.trajex.totg.trajectory.delta"));
    EXPECT_TRUE(n->has_parameter("pipeline.trajex.totg.trajectory.epsilon"));
    EXPECT_EQ(n->get_parameter("pipeline.trajex.totg.sampling.frequency").as_double(), 10.0);

    const auto response = run(*instance);
    EXPECT_EQ(response.error_code.val, error_codes::SUCCESS);
    EXPECT_GT(response.trajectory->getDuration(), 0.0);
    EXPECT_LE(max_spacing(response), 0.1 + 1e-9);
}

TEST_F(response_adapter_test, compat_adapter_declares_moveit_names_and_defaults_and_times_a_plan) {
    const auto n = node("compat_defaults");
    const auto instance = load(k_compat, n);
    EXPECT_EQ(instance->getDescription(), "AddTotgTimeParameterizationCompat");
    EXPECT_EQ(n->get_parameter("pipeline.totg.path_tolerance").as_double(), 0.1);
    EXPECT_EQ(n->get_parameter("pipeline.totg.resample_dt").as_double(), 0.1);
    EXPECT_EQ(n->get_parameter("pipeline.totg.min_angle_change").as_double(), 0.001);

    const auto response = run(*instance);
    EXPECT_EQ(response.error_code.val, error_codes::SUCCESS);
    EXPECT_LE(max_spacing(response), 0.1 + 1e-9);
}

// Overrides arrive already declared, the way move_group receives values from YAML, and may be integers.
TEST_F(response_adapter_test, overrides_are_honored) {
    const auto compat_node = node("compat_overrides", {rclcpp::Parameter("pipeline.totg.resample_dt", 0.02)});
    const auto compat = load(k_compat, compat_node);
    const auto compat_response = run(*compat);
    ASSERT_EQ(compat_response.error_code.val, error_codes::SUCCESS);
    EXPECT_LE(max_spacing(compat_response), 0.02 + 1e-9);

    const auto trajex_node = node("trajex_overrides", {rclcpp::Parameter("pipeline.trajex.totg.sampling.frequency", 100)});
    const auto trajex_adapter = load(k_trajex, trajex_node);
    const auto trajex_response = run(*trajex_adapter);
    ASSERT_EQ(trajex_response.error_code.val, error_codes::SUCCESS);
    EXPECT_LE(max_spacing(trajex_response), 0.01 + 1e-9);
}

// Changes made after initialization apply to the next plan.
TEST_F(response_adapter_test, parameter_changes_apply_to_later_plans) {
    const auto n = node("trajex_runtime");
    const auto instance = load(k_trajex, n);
    EXPECT_LE(max_spacing(run(*instance)), 0.1 + 1e-9);
    n->set_parameter(rclcpp::Parameter("pipeline.trajex.totg.sampling.frequency", 50.0));
    EXPECT_LE(max_spacing(run(*instance)), 0.02 + 1e-9);
}

TEST_F(response_adapter_test, a_response_without_a_path_is_an_invalid_plan) {
    const auto n = node("trajex_no_path");
    const auto instance = load(k_trajex, n);
    planning_interface::MotionPlanResponse response;
    instance->adapt(nullptr, request(), response);
    EXPECT_EQ(response.error_code.val, error_codes::INVALID_MOTION_PLAN);
}

TEST_F(response_adapter_test, invalid_parameters_fail_the_plan_without_throwing) {
    const auto n = node("compat_invalid");
    const auto instance = load(k_compat, n);
    ASSERT_EQ(run(*instance).error_code.val, error_codes::SUCCESS);

    n->set_parameter(rclcpp::Parameter("pipeline.totg.resample_dt", 0.0));
    auto response = plan();
    EXPECT_NO_THROW(instance->adapt(nullptr, request(), response));
    EXPECT_EQ(response.error_code.val, error_codes::FAILURE);
}

// The adapter holds its node weakly, since the node owns the pipeline that owns the adapter. Once the node is gone,
// the adapter should report failure rather than try to use it.
TEST_F(response_adapter_test, an_adapter_whose_node_is_gone_fails_cleanly) {
    auto n = node("trajex_orphaned");
    const auto instance = load(k_trajex, n);
    n.reset();
    auto response = plan();
    EXPECT_NO_THROW(instance->adapt(nullptr, request(), response));
    EXPECT_EQ(response.error_code.val, error_codes::FAILURE);
}

// MoveIt's own adapter and the compat adapter declare the same parameters, so either should be able to initialize
// first on a shared node.
TEST_F(response_adapter_test, compat_adapter_coexists_with_moveits_own) {
    {
        const auto n = node("moveit_first");
        const auto moveit = load(k_moveit, n);
        const auto ours = load(k_compat, n);
        EXPECT_EQ(run(*moveit).error_code.val, error_codes::SUCCESS);
        EXPECT_EQ(run(*ours).error_code.val, error_codes::SUCCESS);
    }
    {
        const auto n = node("ours_first");
        const auto ours = load(k_compat, n);
        const auto moveit = load(k_moveit, n);
        EXPECT_EQ(run(*ours).error_code.val, error_codes::SUCCESS);
        EXPECT_EQ(run(*moveit).error_code.val, error_codes::SUCCESS);
    }
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    const int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
