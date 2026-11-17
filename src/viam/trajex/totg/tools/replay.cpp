#include <viam/trajex/totg/tools/replay.hpp>

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <json/json.h>

#if __has_include(<xtensor/core/xmath.hpp>)
#include <xtensor/core/xmath.hpp>
#else
#include <xtensor/xmath.hpp>
#endif

namespace viam::trajex::totg {

namespace {

// One fixed-length numeric field of a kinematics_chain joint in a replay record.
template <std::size_t N>
std::array<double, N> chain_joint_field(const Json::Value& joint, Json::ArrayIndex index, const char* key) {
    const auto& value = joint[key];
    if (!value.isArray() || value.size() != N) {
        throw std::runtime_error("kinematics_chain joint " + std::to_string(index) + " field " + key + " must be an array of " +
                                 std::to_string(N) + " numbers");
    }
    std::array<double, N> out{};
    for (Json::ArrayIndex k = 0; k < value.size(); ++k) {
        out[k] = value[k].asDouble();
    }
    return out;
}

}  // namespace

std::pair<planner_base::config, xmatrix<>> parse_replay_record(std::istream& in) {
    Json::Value root;
    const Json::CharReaderBuilder reader;
    std::string errs;
    if (!Json::parseFromStream(reader, in, &root, &errs)) {
        throw std::runtime_error("failed to parse replay record JSON: " + errs);
    }

    auto require = [&](const char* field) -> const Json::Value& {
        if (!root.isMember(field)) {
            throw std::runtime_error(std::string("replay record missing required field: ") + field);
        }
        return root[field];
    };

    const auto& vel_json = require("max_velocity_vec_rads_per_sec");
    const auto& acc_json = require("max_acceleration_vec_rads_per_sec2");
    const auto& wps_json = require("waypoints_rads");

    if (!vel_json.isArray() || vel_json.empty()) {
        throw std::runtime_error("max_velocity_vec_rads_per_sec must be a non-empty array");
    }
    if (!acc_json.isArray() || acc_json.empty()) {
        throw std::runtime_error("max_acceleration_vec_rads_per_sec2 must be a non-empty array");
    }
    if (!wps_json.isArray() || wps_json.empty()) {
        throw std::runtime_error("waypoints_rads must be a non-empty array");
    }

    const auto dof = static_cast<std::size_t>(vel_json.size());
    const auto num_waypoints = static_cast<std::size_t>(wps_json.size());

    xvector<> velocity_limits = xt::zeros<double>(std::vector<std::size_t>{dof});
    for (Json::ArrayIndex i = 0; i < vel_json.size(); ++i) {
        velocity_limits(i) = vel_json[i].asDouble();
    }

    xvector<> acceleration_limits = xt::zeros<double>(std::vector<std::size_t>{static_cast<std::size_t>(acc_json.size())});
    for (Json::ArrayIndex i = 0; i < acc_json.size(); ++i) {
        acceleration_limits(i) = acc_json[i].asDouble();
    }

    xmatrix<> waypoints = xt::zeros<double>(std::vector<std::size_t>{num_waypoints, dof});
    for (Json::ArrayIndex i = 0; i < wps_json.size(); ++i) {
        const auto& wp = wps_json[i];
        if (!wp.isArray() || static_cast<std::size_t>(wp.size()) != dof) {
            throw std::runtime_error("waypoint has wrong number of joints");
        }
        for (Json::ArrayIndex j = 0; j < wp.size(); ++j) {
            waypoints(i, static_cast<std::size_t>(j)) = wp[j].asDouble();
        }
    }

    planner_base::config cfg;
    cfg.velocity_limits = std::move(velocity_limits);
    cfg.acceleration_limits = std::move(acceleration_limits);
    if (root.isMember("path_tolerance_delta_rads")) {
        cfg.path_blend_tolerance = root["path_tolerance_delta_rads"].asDouble();
    }
    if (root.isMember("path_colinearization_ratio")) {
        cfg.colinearization_ratio = root["path_colinearization_ratio"].asDouble();
    }
    if (root.isMember("min_blend_curvature")) {
        cfg.min_blend_curvature = root["min_blend_curvature"].asDouble();
    }
    if (root.isMember("max_blend_curvature")) {
        cfg.max_blend_curvature = root["max_blend_curvature"].asDouble();
    }

    // Optional kinematics and TCP speed limit (schema v3). Each joint drives the builder overload
    // its axis key names, which rebuilds the chain the record was written from.
    if (root.isMember("kinematics_chain")) {
        const auto& chain_json = root["kinematics_chain"];
        if (!chain_json.isArray() || chain_json.empty()) {
            throw std::runtime_error("kinematics_chain must be a non-empty array of joints");
        }
        using chain = kinematics::chain;
        chain::builder builder;
        for (Json::ArrayIndex i = 0; i < chain_json.size(); ++i) {
            const auto& joint = chain_json[i];
            const chain::joint_origin origin{.translation = chain_joint_field<3>(joint, i, "xyz"),
                                             .rotation = chain_joint_field<4>(joint, i, "wxyz")};
            const bool rotates = joint.isMember("rotation_axis");
            const bool translates = joint.isMember("translation_axis");
            if (rotates && translates) {
                throw std::runtime_error("kinematics_chain joint " + std::to_string(i) +
                                         " has both a rotation_axis and a translation_axis");
            }
            if (rotates) {
                builder.extend(origin, chain::rotation_axis{chain_joint_field<3>(joint, i, "rotation_axis")});
            } else if (translates) {
                builder.extend(origin, chain::translation_axis{chain_joint_field<3>(joint, i, "translation_axis")});
            } else {
                builder.extend(origin);
            }
        }
        cfg.kinematics_chain = std::move(builder).build();
    }

    // A v2 record that carried a TCP limit did so with a model_table, which nothing reads any
    // more. Its cap arrives without kinematics and is refused here rather than silently dropped.
    if (root.isMember("tcp_max_linear_speed")) {
        if (!cfg.kinematics_chain) {
            throw std::runtime_error("tcp_max_linear_speed given without a kinematics_chain to apply it to");
        }
        cfg.tcp_max_linear_speed = root["tcp_max_linear_speed"].asDouble();
    }

    return {std::move(cfg), std::move(waypoints)};
}

std::pair<planner_base::config, xmatrix<>> parse_replay_record(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open replay record file: " + path.string());
    }
    return parse_replay_record(in);
}

replay_planner::replay_planner(config cfg, std::unique_ptr<trajectory_integration_event_collector> collector)
    : planner<replay_receiver>(std::move(cfg)), collector_(std::move(collector)) {
    mutable_config().observer = collector_.get();
}

replay_planner replay_planner::create(std::istream& in, std::optional<std::size_t> prefix_waypoint_count) {
    auto [cfg, waypoints] = parse_replay_record(in);

    // Validate prefix request up-front so callers see a clear error before the planner is constructed.
    // When a prefix shorter than the full set is requested, materialize the leading rows into a fresh
    // matrix and drop the original. The remainder of this function then runs unchanged against `waypoints`.
    const auto total_waypoints = waypoints.shape(0);
    if (prefix_waypoint_count.has_value()) {
        if (*prefix_waypoint_count == 0 || *prefix_waypoint_count > total_waypoints) {
            throw std::out_of_range("replay_planner prefix_waypoint_count " + std::to_string(*prefix_waypoint_count) +
                                    " is out of range for a record with " + std::to_string(total_waypoints) + " waypoints");
        }
        if (*prefix_waypoint_count < total_waypoints) {
            xmatrix<> prefix = xt::view(waypoints, xt::range(std::size_t{0}, *prefix_waypoint_count), xt::all());
            waypoints = std::move(prefix);
        }
    }

    auto collector = std::make_unique<trajectory_integration_event_collector>();
    replay_planner p(std::move(cfg), std::move(collector));

    // Stash the waypoints array and provision it as a single unsegmented waypoint set.
    auto data = p.stash(std::move(waypoints));
    p.with_waypoint_provider([data](auto&) { return waypoint_accumulator{*data}; });

    p.with_totg([](const auto&, replay_receiver& recv, const waypoint_accumulator& waypoints, trajectory&& traj, auto) {
        recv.waypoints.emplace(waypoints);
        recv.traj = std::move(traj);
    });

    return p;
}

replay_planner replay_planner::create(const std::filesystem::path& path, std::optional<std::size_t> prefix_waypoint_count) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open replay record file: " + path.string());
    }
    return create(in, prefix_waypoint_count);
}

const trajectory_integration_event_collector& replay_planner::collector() const noexcept {
    return *collector_;
}

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)

legacy_replay_planner legacy_replay_planner::create(std::istream& in) {
    auto [cfg, waypoints] = parse_replay_record(in);

    // The legacy generator cannot enforce a TCP limit, and the planner refuses to register it
    // while one is set. A legacy replay of a TCP-carrying record is a deliberate uncapped
    // comparison run, so drop the cap explicitly; the kinematics are kept.
    cfg.tcp_max_linear_speed.reset();

    legacy_replay_planner p(std::move(cfg));

    auto data = p.stash(std::move(waypoints));
    p.with_waypoint_provider([data](auto&) { return waypoint_accumulator{*data}; });

    p.with_legacy([](const auto&, legacy_replay_receiver& recv, const waypoint_accumulator&, Path&& path, Trajectory&& traj, auto) {
        recv.result.emplace(std::move(path), std::move(traj));
    });

    return p;
}

legacy_replay_planner legacy_replay_planner::create(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open replay record file: " + path.string());
    }
    return create(in);
}

#endif

}  // namespace viam::trajex::totg
