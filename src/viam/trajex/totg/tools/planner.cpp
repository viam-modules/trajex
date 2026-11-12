#include <viam/trajex/totg/tools/planner.hpp>

#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <json/json.h>

namespace viam::trajex::totg {

namespace {

Json::Value json_array(const auto& values) {
    Json::Value array(Json::arrayValue);
    for (const double v : values) {
        array.append(v);
    }
    return array;
}

// Writes each joint of a kinematic chain as parse_replay_record reads it back. A moving
// joint's axis key names how it moves, as the builder overload does, so no kind is recorded.
struct chain_writer {
    using chain = kinematics::chain;

    Json::Value joints{Json::arrayValue};

    static Json::Value entry(const chain::joint_origin& origin) {
        Json::Value joint(Json::objectValue);
        joint["xyz"] = json_array(origin.translation);
        joint["wxyz"] = json_array(origin.rotation);
        return joint;
    }

    void operator()(const chain::joint_origin& origin) {
        joints.append(entry(origin));
    }

    void operator()(const chain::joint_origin& origin, const chain::rotation_axis& axis) {
        Json::Value joint = entry(origin);
        joint["rotation_axis"] = json_array(axis.direction);
        joints.append(std::move(joint));
    }

    void operator()(const chain::joint_origin& origin, const chain::translation_axis& axis) {
        Json::Value joint = entry(origin);
        joint["translation_axis"] = json_array(axis.direction);
        joints.append(std::move(joint));
    }
};

}  // namespace

planner_base::planner_base(struct config cfg) : config_(std::move(cfg)) {
    // Checked here rather than when the limit is built, so a bad config fails where it is
    // accepted. serialize_for_replay relies on it too, since it runs on failure paths to record
    // diagnostics and must not throw.
    if (config_.tcp_max_linear_speed && !config_.kinematics_chain) {
        throw std::invalid_argument("planner config tcp_max_linear_speed requires kinematics_chain");
    }

    // As in the C ABI, a chain must match the joint limits whether or not a cap is set, so that a
    // mismatch is reported by the config that made it rather than partway through the first
    // trajectory that caps, and so that a replay record never carries a chain it could not apply.
    if (config_.kinematics_chain && config_.kinematics_chain->dof() != config_.velocity_limits->size()) {
        throw std::invalid_argument("planner config kinematics_chain has " + std::to_string(config_.kinematics_chain->dof()) +
                                    " rotating or translating joints, but the joint limits describe " +
                                    std::to_string(config_.velocity_limits->size()));
    }
}

const struct planner_base::config& planner_base::get_config() const noexcept {
    return config_;
}

const planner_base::timing_stats& planner_base::timing() const noexcept {
    return timing_;
}

std::size_t planner_base::processed_waypoint_count() const noexcept {
    return processed_waypoint_count_;
}

path::options planner_base::path_options() const {
    auto opts = path::options{}.set_max_blend_deviation(config_.path_blend_tolerance);
    if (config_.colinearization_ratio) {
        opts.set_max_linear_deviation(config_.path_blend_tolerance * *config_.colinearization_ratio);
    }
    if (config_.min_blend_curvature) {
        opts.set_min_blend_curvature(*config_.min_blend_curvature);
    }
    if (config_.max_blend_curvature) {
        opts.set_max_blend_curvature(*config_.max_blend_curvature);
    }
    return opts;
}

struct planner_base::config& planner_base::mutable_config() noexcept {
    return config_;
}

planner_base::timing_stats& planner_base::mutable_timing() noexcept {
    return timing_;
}

std::size_t& planner_base::mutable_processed_waypoint_count() noexcept {
    return processed_waypoint_count_;
}

std::string planner_base::serialize_for_replay(const waypoint_accumulator& waypoints, std::optional<std::string_view> error_message) const {
    // Build ISO 8601 timestamp with microsecond precision.
    const auto now = std::chrono::system_clock::now();
    const auto seconds_part = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
    const auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::time_point{seconds_part});
    const auto delta_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch() - seconds_part);

    struct tm buf;
    if (gmtime_r(&tt, &buf) == nullptr) {
        throw std::runtime_error("failed to convert time to iso8601");
    }
    std::ostringstream ts;
    ts << std::put_time(&buf, "%FT%T") << "." << std::setw(6) << std::setfill('0') << delta_us.count() << "Z";

    // Reconstruct path::options the same way run_totg_ does so the curvature bounds
    // recorded into the replay match what the totg run actually used: configured values
    // when get_config() carries them, defaults otherwise.
    auto path_opts = path::options{}.set_max_blend_deviation(get_config().path_blend_tolerance);
    if (get_config().colinearization_ratio) {
        path_opts.set_max_linear_deviation(get_config().path_blend_tolerance * *get_config().colinearization_ratio);
    }
    if (get_config().min_blend_curvature) {
        path_opts.set_min_blend_curvature(*get_config().min_blend_curvature);
    }
    if (get_config().max_blend_curvature) {
        path_opts.set_max_blend_curvature(*get_config().max_blend_curvature);
    }

    Json::Value root;
    root["schema_version"] = 3;
    root["timestamp"] = ts.str();
    if (error_message) {
        root["error_message"] = std::string(*error_message);
    }
    root["path_tolerance_delta_rads"] = get_config().path_blend_tolerance;
    if (get_config().colinearization_ratio) {
        root["path_colinearization_ratio"] = *get_config().colinearization_ratio;
    }

    Json::Value vel_array(Json::arrayValue);
    for (const double v : *get_config().velocity_limits) {
        vel_array.append(v);
    }
    root["max_velocity_vec_rads_per_sec"] = std::move(vel_array);

    Json::Value acc_array(Json::arrayValue);
    for (const double a : *get_config().acceleration_limits) {
        acc_array.append(a);
    }
    root["max_acceleration_vec_rads_per_sec2"] = std::move(acc_array);

    root["min_blend_curvature"] = path_opts.min_blend_curvature();
    root["max_blend_curvature"] = path_opts.max_blend_curvature();

    if (get_config().kinematics_chain) {
        chain_writer writer;
        get_config().kinematics_chain->visit_joints(writer);
        root["kinematics_chain"] = std::move(writer.joints);
    }
    if (get_config().tcp_max_linear_speed) {
        root["tcp_max_linear_speed"] = *get_config().tcp_max_linear_speed;
    }

    Json::Value waypoints_array(Json::arrayValue);
    for (const auto& waypoint : waypoints) {
        Json::Value wp(Json::arrayValue);
        for (const double val : waypoint) {
            wp.append(val);
        }
        waypoints_array.append(std::move(wp));
    }
    root["waypoints_rads"] = std::move(waypoints_array);

    Json::StreamWriterBuilder writer;
    writer["precision"] = static_cast<int>(std::numeric_limits<double>::max_digits10);
    writer["indentation"] = " ";
    return Json::writeString(writer, root);
}

}  // namespace viam::trajex::totg
