// MoveIt planning response adapters that time parameterize with trajex TOTG. pluginlib finds them through
// plugins.xml in the ament package, and nothing else refers to them, so there's no header.

#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_interface/planning_response_adapter.hpp>
#include <moveit/utils/logger.hpp>
#include <moveit/utils/moveit_error_code.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include <viam/trajex/moveit/totg/time_parameterization.hpp>

namespace viam::trajex::moveit::totg {

namespace {

rclcpp::Logger logger() {
    return ::moveit::getLogger("viam.trajex.moveit.totg.response_adapter");
}

// Declares and reads one of an adapter's double parameters. The hosting node may already have declared it, for
// instance from a YAML override in a node that declares overrides automatically, in which case we leave it alone.
// Otherwise we declare it with its default, its description, and a non-negative range. Values that must be strictly
// positive are checked later, when the time parameterization is built.
class double_parameter {
   public:
    double_parameter(std::string name, double default_value, std::string description)
        : name_(std::move(name)), default_(default_value), description_(std::move(description)) {}

    void declare(rclcpp::Node& node, const std::string& prefix) {
        full_name_ = prefix + name_;
        if (node.has_parameter(full_name_)) {
            return;
        }
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        descriptor.description = description_;
        rcl_interfaces::msg::FloatingPointRange range;
        range.from_value = 0.0;
        range.to_value = std::numeric_limits<double>::max();
        descriptor.floating_point_range.push_back(range);
        node.declare_parameter(full_name_, default_, descriptor);
    }

    // A `1` in YAML is an integer, and a parameter declared from it stays an integer, so we accept either type.
    double read(rclcpp::Node& node) const {
        const auto parameter = node.get_parameter(full_name_);
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
            return static_cast<double>(parameter.as_int());
        }
        return parameter.as_double();
    }

   private:
    std::string name_;
    double default_;
    std::string description_;
    std::string full_name_;
};

std::string join(const std::string& parameter_namespace, const std::string& group) {
    return parameter_namespace.empty() ? group + "." : parameter_namespace + "." + group + ".";
}

// The two adapters differ only in their parameters and in how those become a time parameterization, so everything
// else lives here. We hold the node weakly, since the node owns the planning pipeline, which owns the adapter, and a
// strong reference would make a cycle.
class adapter_base : public planning_interface::PlanningResponseAdapter {
   public:
    void initialize(const rclcpp::Node::SharedPtr& node, const std::string& parameter_namespace) override {
        node_ = node;
        declare_(*node, parameter_namespace);
    }

    void adapt(const planning_scene::PlanningSceneConstPtr&,
               const planning_interface::MotionPlanRequest& request,
               planning_interface::MotionPlanResponse& response) const override {
        if (!response.trajectory) {
            RCLCPP_ERROR(logger(), "%s: the motion plan response contains no path.", getDescription().c_str());
            response.error_code = ::moveit::core::MoveItErrorCode::INVALID_MOTION_PLAN;
            return;
        }

        bool succeeded = false;
        try {
            const auto node = node_.lock();
            if (!node) {
                throw std::logic_error{"adapter used without initialization, or after its node was destroyed"};
            }
            // We build a new time parameterization for each request, from the parameters' current values, so that
            // changes made while the node runs take effect.
            succeeded = make_(*node).computeTimeStamps(
                *response.trajectory, request.max_velocity_scaling_factor, request.max_acceleration_scaling_factor);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(logger(), "%s: %s", getDescription().c_str(), e.what());
        }

        if (!succeeded) {
            RCLCPP_ERROR(logger(), "%s failed to time parameterize the trajectory.", getDescription().c_str());
        }
        response.error_code = succeeded ? ::moveit::core::MoveItErrorCode::SUCCESS : ::moveit::core::MoveItErrorCode::FAILURE;
    }

   private:
    virtual void declare_(rclcpp::Node& node, const std::string& parameter_namespace) = 0;
    virtual time_parameterization make_(rclcpp::Node& node) const = 0;

    std::weak_ptr<rclcpp::Node> node_;
};

}  // namespace

// Configured by trajex's own options, under `trajex.totg`, with trajex's defaults.
class response_adapter final : public adapter_base {
   public:
    std::string getDescription() const override {
        return "AddTotgTimeParameterization";
    }

   private:
    void declare_(rclcpp::Node& node, const std::string& parameter_namespace) override {
        const auto prefix = join(parameter_namespace, "trajex.totg");
        for (auto* parameter : {&max_blend_deviation_,
                                &max_linear_deviation_,
                                &min_blend_curvature_,
                                &max_blend_curvature_,
                                &deduplication_tolerance_,
                                &delta_,
                                &epsilon_,
                                &sampling_frequency_}) {
            parameter->declare(node, prefix);
        }
    }

    time_parameterization make_(rclcpp::Node& node) const override {
        trajex::totg::path::options path_options;
        path_options.set_max_blend_deviation(max_blend_deviation_.read(node))
            .set_max_linear_deviation(max_linear_deviation_.read(node))
            .set_min_blend_curvature(min_blend_curvature_.read(node))
            .set_max_blend_curvature(max_blend_curvature_.read(node));

        trajex::totg::trajectory::options trajectory_options;
        trajectory_options.delta = trajex::totg::trajectory::seconds{delta_.read(node)};
        trajectory_options.epsilon = trajex::epsilon{epsilon_.read(node)};

        return time_parameterization{std::move(path_options),
                                     deduplication_tolerance_.read(node),
                                     std::move(trajectory_options),
                                     types::hertz{sampling_frequency_.read(node)}};
    }

    // We take the defaults from trajex, so that they're only stated in one place.
    inline static const trajex::totg::path::options k_path_defaults{};

    double_parameter max_blend_deviation_{
        "path.max_blend_deviation",
        k_path_defaults.max_blend_deviation(),
        "Maximum distance a blend may deviate from its corner, in configuration space units (rad for revolute joints, "
        "m for prismatic). 0 disables blending."};
    double_parameter max_linear_deviation_{
        "path.max_linear_deviation",
        k_path_defaults.max_linear_deviation(),
        "Maximum deviation for merging nearly collinear waypoints, in configuration space units. 0 disables merging."};
    double_parameter min_blend_curvature_{"path.min_blend_curvature",
                                          k_path_defaults.min_blend_curvature(),
                                          "Minimum blend curvature, in 1 / configuration space unit. Must be positive."};
    double_parameter max_blend_curvature_{"path.max_blend_curvature",
                                          k_path_defaults.max_blend_curvature(),
                                          "Maximum blend curvature, in 1 / configuration space unit. Must be at least "
                                          "path.min_blend_curvature."};
    double_parameter deduplication_tolerance_{
        "path.deduplication_tolerance",
        time_parameterization::k_default_deduplication_tolerance,
        "Waypoints within this distance of the previous kept waypoint in every joint are dropped, in configuration "
        "space units."};
    double_parameter delta_{"trajectory.delta",
                            trajex::totg::trajectory::options::k_default_delta.count(),
                            "Phase plane integration time step, in s. Must be positive."};
    double_parameter epsilon_{"trajectory.epsilon",
                              static_cast<double>(trajex::totg::trajectory::options::k_default_epsilon),
                              "Tolerance for numerical comparisons, dimensionless. Must be positive."};
    double_parameter sampling_frequency_{"sampling.frequency",
                                         time_parameterization::k_default_sampling_frequency_hz,
                                         "Minimum rate at which waypoints are written into the trajectory, in Hz. "
                                         "Must be positive."};
};

// Configured by the same parameters as MoveIt's own AddTimeOptimalParameterization, so that an existing `totg`
// configuration works unchanged.
class compat_response_adapter final : public adapter_base {
   public:
    std::string getDescription() const override {
        return "AddTotgTimeParameterizationCompat";
    }

   private:
    void declare_(rclcpp::Node& node, const std::string& parameter_namespace) override {
        const auto prefix = join(parameter_namespace, "totg");
        for (auto* parameter : {&path_tolerance_, &resample_dt_, &min_angle_change_}) {
            parameter->declare(node, prefix);
        }
    }

    // MoveIt's `path_tolerance` maps to the blend deviation, `min_angle_change` to the deduplication tolerance, and
    // `resample_dt` to the sampling period. MoveIt doesn't merge collinear waypoints, so we turn that off.
    time_parameterization make_(rclcpp::Node& node) const override {
        trajex::totg::path::options path_options;
        path_options.set_max_blend_deviation(path_tolerance_.read(node)).set_max_linear_deviation(0.0);
        return time_parameterization{
            std::move(path_options), min_angle_change_.read(node), {}, types::hertz{1.0 / resample_dt_.read(node)}};
    }

    double_parameter path_tolerance_{"path_tolerance",
                                     0.1,
                                     "Tolerance per joint in which the time parameterization is allowed to deviate from the "
                                     "original path."};
    double_parameter resample_dt_{"resample_dt", 0.1, "Time steps between two adjacent waypoints of the parameterized trajectory."};
    double_parameter min_angle_change_{"min_angle_change", 0.001, "Minimum joint value change to consider two waypoints unique."};
};

}  // namespace viam::trajex::moveit::totg

PLUGINLIB_EXPORT_CLASS(viam::trajex::moveit::totg::response_adapter, planning_interface::PlanningResponseAdapter)
PLUGINLIB_EXPORT_CLASS(viam::trajex::moveit::totg::compat_response_adapter, planning_interface::PlanningResponseAdapter)
