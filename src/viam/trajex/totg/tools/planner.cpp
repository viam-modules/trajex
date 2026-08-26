#include <viam/trajex/totg/tools/planner.hpp>

#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <json/json.h>

namespace viam::trajex::totg {

planner_base::planner_base(struct config cfg) : config_(std::move(cfg)) {
    // Validate model_table shape here rather than in serialize_for_replay: serialization runs
    // on failure paths to record diagnostics, and a throw there would destroy the replay record
    // for the original error.
    if (config_.model_table && (config_.model_table->dimension() != 2 || config_.model_table->shape(1) != 10)) {
        throw std::invalid_argument("planner config model_table must be an (n, 10) tensor");
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
    root["schema_version"] = 2;
    root["timestamp"] = ts.str();
    if (error_message) {
        root["error_message"] = std::string(*error_message);
    }
    root["path_tolerance_delta_rads"] = get_config().path_blend_tolerance;
    if (get_config().colinearization_ratio) {
        root["path_colinearization_ratio"] = *get_config().colinearization_ratio;
    }

    Json::Value vel_array(Json::arrayValue);
    for (const double v : get_config().velocity_limits) {
        vel_array.append(v);
    }
    root["max_velocity_vec_rads_per_sec"] = std::move(vel_array);

    Json::Value acc_array(Json::arrayValue);
    for (const double a : get_config().acceleration_limits) {
        acc_array.append(a);
    }
    root["max_acceleration_vec_rads_per_sec2"] = std::move(acc_array);

    root["min_blend_curvature"] = path_opts.min_blend_curvature();
    root["max_blend_curvature"] = path_opts.max_blend_curvature();

    // TCP Cartesian speed limit. The jacobian callback in get_config().tcp cannot be serialized, so
    // record the scalar cap plus the (n, 10) model-table tensor it was built from; replay rebuilds the
    // callback via make_tcp_jacobian. Both fields are written together and only when the model-table
    // provenance is available, since neither alone reproduces the limit.
    if (get_config().tcp && get_config().model_table) {
        // The (n, 10) shape was validated at planner construction, so writing the table here
        // cannot fail on the failure paths that call this to record diagnostics.
        const auto& table = *get_config().model_table;
        root["max_tcp_speed_m_per_sec"] = get_config().tcp->max_velocity;

        Json::Value model_table_array(Json::arrayValue);
        for (std::size_t i = 0; i < table.shape(0); ++i) {
            Json::Value row(Json::arrayValue);
            for (std::size_t j = 0; j < 10; ++j) {
                row.append(table(i, j));
            }
            model_table_array.append(std::move(row));
        }
        root["model_table"] = std::move(model_table_array);
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
