#pragma once

#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <utility>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/totg/observers.hpp>
#include <viam/trajex/totg/tools/planner.hpp>
#include <viam/trajex/totg/trajectory.hpp>

namespace viam::trajex::totg {

///
/// Parses a canonical replay JSON record from `in` into a planner config and
/// a (num_waypoints, dof) xarray of waypoints. Exposed for use by tools and
/// tests that want the raw record without instantiating a planner.
///
/// @param in Stream containing a canonical JSON replay record
/// @return Pair of (config, waypoints)
/// @throws std::runtime_error if the stream cannot be parsed or required fields are missing
///
std::pair<planner_base::config, xt::xarray<double>> parse_replay_record(std::istream& in);

///
/// File-path overload of parse_replay_record.
///
/// @param path Path to a canonical JSON replay record file
/// @return Pair of (config, waypoints)
/// @throws std::runtime_error if the file cannot be opened or parsed
///
std::pair<planner_base::config, xt::xarray<double>> parse_replay_record(const std::filesystem::path& path);

///
/// Receiver for replay_planner. Holds the most recently generated
/// trajectory so callers can access it after execute() completes.
///
struct replay_receiver {
    std::optional<trajectory> traj;
};

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
///
/// Receiver for legacy_replay_planner. Holds the path and trajectory produced
/// by the legacy generator so callers can access them after execute() completes.
///
struct legacy_replay_receiver {
    std::optional<std::pair<Path, Trajectory>> result;
};
#endif

///
/// Replay planner for the TOTG algorithm.
///
/// Loads a canonical JSON replay record, runs TOTG with an attached event collector,
/// and provides the collector for downstream diagnostic JSON output.
///
/// Typical usage:
/// @code
///   auto p = replay_planner::create("failed.json");
///   auto outcome = p.execute([](const auto&, auto tx, const auto&) { return tx; });
///   write_trajectory_json(std::cout, p.collector(),
///                         outcome.receiver ? &outcome.receiver->traj.value() : nullptr);
/// @endcode
///
class replay_planner : public planner<replay_receiver> {
   public:
    ///
    /// Constructs a replay planner from a replay record stream.
    ///
    /// When `prefix_waypoint_count` is supplied, the planner runs over only the
    /// first N waypoints of the record instead of the full set. This is intended
    /// for tests that compare a trajectory generated from a full waypoint set
    /// against one generated from a prefix of the same set.
    ///
    /// @param in Stream containing a canonical JSON replay record
    /// @param prefix_waypoint_count Optional cap on the number of leading waypoints to use
    /// @throws std::runtime_error if the stream cannot be parsed or required fields are missing
    /// @throws std::out_of_range if `prefix_waypoint_count` is zero or exceeds the record's waypoint count
    ///
    static replay_planner create(std::istream& in, std::optional<std::size_t> prefix_waypoint_count = std::nullopt);

    ///
    /// Constructs a replay planner from a replay record file path.
    ///
    /// @param path Path to a canonical JSON replay record file
    /// @param prefix_waypoint_count Optional cap on the number of leading waypoints to use; see stream overload
    /// @throws std::runtime_error if the file cannot be opened or parsed
    /// @throws std::out_of_range if `prefix_waypoint_count` is zero or exceeds the record's waypoint count
    ///
    static replay_planner create(const std::filesystem::path& path, std::optional<std::size_t> prefix_waypoint_count = std::nullopt);

    ///
    /// Returns the event collector populated during execute().
    ///
    const trajectory_integration_event_collector& collector() const noexcept;

   private:
    // The collector lives on the heap so its address is stable through moves.
    // mutable_config().observer holds a raw pointer to it; the pointer remains
    // valid as long as the planner is alive.
    std::unique_ptr<trajectory_integration_event_collector> collector_;

    explicit replay_planner(config cfg, std::unique_ptr<trajectory_integration_event_collector> collector);
};

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
///
/// Replay planner for the legacy trajectory algorithm.
///
/// Loads a canonical JSON replay record and runs the legacy generator. Primarily
/// useful for getting a debugger session on a known-bad trajectory.
///
class legacy_replay_planner : public planner<legacy_replay_receiver> {
   public:
    using planner<legacy_replay_receiver>::planner;

    ///
    /// Constructs a replay planner from a replay record stream.
    ///
    static legacy_replay_planner create(std::istream& in);

    ///
    /// Constructs a replay planner from a replay record file path.
    ///
    static legacy_replay_planner create(const std::filesystem::path& path);
};

#endif

}  // namespace viam::trajex::totg
