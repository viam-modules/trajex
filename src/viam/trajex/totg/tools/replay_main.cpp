// CLI tool: replay a canonical JSON trajectory replay record through the TOTG algorithm and
// emit diagnostic JSON for visualization.
//
// Usage:
//   trajex_replay_trajectory failed.json | python3 visualize.py
//   cat failed.json | trajex_replay_trajectory | python3 visualize.py

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <viam/trajex/totg/tools/json_serialization.hpp>
#include <viam/trajex/totg/tools/replay.hpp>

using viam::trajex::totg::replay_planner;

int main(int argc, char* argv[]) try {
    replay_planner p = (argc >= 2) ? replay_planner::create(std::filesystem::path(argv[1])) : replay_planner::create(std::cin);

    auto outcome = p.execute([](const auto&, auto tx, const auto&) { return tx; });

    // If trajex failed before any observer hook fired (e.g. path::create threw), the
    // collector captured nothing and there is no failure JSON to emit. Rethrow so the
    // outer catch reports the error and we exit non-zero. The trajex-internal failure
    // path (on_failed observer fires, collector retains invalid_trajectory / events) is
    // preserved: in that case we fall through to write_trajectory_json, which uses the
    // collector contents to emit a complete failure record with exit 0.
    if (outcome.error && !p.collector().invalid_trajectory() && !p.collector().invalid_exception()) {
        std::rethrow_exception(outcome.error);
    }

    const viam::trajex::totg::trajectory* traj = nullptr;
    if (outcome.receiver && outcome.receiver->traj) {
        traj = &*outcome.receiver->traj;
    }

    viam::trajex::totg::write_trajectory_json(std::cout, p.collector(), traj);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
