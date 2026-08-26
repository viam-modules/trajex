// Streaming session simulator.
//
// Loads a canonical replay JSON record, sweeps a (W_c, W_r) grid by simulating a
// caller that streams waypoints at maximum rate on top of streaming::session, and
// writes a CSV summarizing each cell.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xview.hpp>
#endif

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/streaming/session.hpp>
#include <viam/trajex/totg/tools/replay.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/hertz.hpp>

namespace {

using viam::trajex::totg::parse_replay_record;
using viam::trajex::totg::path;
using viam::trajex::totg::trajectory;
using viam::trajex::totg::waypoint_accumulator;
namespace streaming = viam::trajex::totg::streaming;
namespace types = viam::trajex::types;

struct sim_config {
    std::filesystem::path replay_path;
    std::filesystem::path output_path;
    double w_c_min = 0.5;
    double w_c_max = 32.0;
    std::size_t w_c_count = 7;
    double w_r_min = 0.01;
    double w_r_max = 1.0;
    std::size_t w_r_count = 7;
    double speed_factor = 1.0;
    std::size_t batch_size = 2;
    double sample_rate_hz = 100.0;
};

void usage_and_exit(const char* argv0, int code) {
    std::cerr << "usage: " << argv0 << " --replay <path> --output <csv> "
              << "[--w-c-min N] [--w-c-max N] [--w-c-count N] "
              << "[--w-r-min N] [--w-r-max N] [--w-r-count N] "
              << "[--speed-factor N] [--batch-size N] [--sample-rate N]\n";
    std::exit(code);  // NOLINT(concurrency-mt-unsafe)
}

sim_config parse_args(int argc, char* argv[]) {
    sim_config cfg;
    auto next = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            usage_and_exit(argv[0], 2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--replay") == 0) {
            cfg.replay_path = next(i);
        } else if (std::strcmp(a, "--output") == 0) {
            cfg.output_path = next(i);
        } else if (std::strcmp(a, "--w-c-min") == 0) {
            cfg.w_c_min = std::stod(next(i));
        } else if (std::strcmp(a, "--w-c-max") == 0) {
            cfg.w_c_max = std::stod(next(i));
        } else if (std::strcmp(a, "--w-c-count") == 0) {
            cfg.w_c_count = std::stoul(next(i));
        } else if (std::strcmp(a, "--w-r-min") == 0) {
            cfg.w_r_min = std::stod(next(i));
        } else if (std::strcmp(a, "--w-r-max") == 0) {
            cfg.w_r_max = std::stod(next(i));
        } else if (std::strcmp(a, "--w-r-count") == 0) {
            cfg.w_r_count = std::stoul(next(i));
        } else if (std::strcmp(a, "--speed-factor") == 0) {
            cfg.speed_factor = std::stod(next(i));
        } else if (std::strcmp(a, "--batch-size") == 0) {
            cfg.batch_size = std::stoul(next(i));
        } else if (std::strcmp(a, "--sample-rate") == 0) {
            cfg.sample_rate_hz = std::stod(next(i));
        } else if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            usage_and_exit(argv[0], 0);
        } else {
            std::cerr << "unknown flag: " << a << "\n";
            usage_and_exit(argv[0], 2);
        }
    }
    if (cfg.replay_path.empty() || cfg.output_path.empty()) {
        usage_and_exit(argv[0], 2);
    }
    if (cfg.batch_size < 2) {
        throw std::invalid_argument("batch-size must be at least 2 (seam point counts)");
    }
    return cfg;
}

// Build a geometric sweep of `count` values from min to max, inclusive at both ends.
// count == 1 yields {min}.
std::vector<double> geometric_grid(double lo, double hi, std::size_t count) {
    std::vector<double> values;
    values.reserve(count);
    if (count == 0) {
        return values;
    }
    if (count == 1) {
        values.push_back(lo);
        return values;
    }
    const double log_lo = std::log(lo);
    const double log_hi = std::log(hi);
    for (std::size_t i = 0; i < count; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(count - 1);
        values.push_back(std::exp(log_lo + (t * (log_hi - log_lo))));
    }
    return values;
}

// Build path::options the same way planner.cpp reconstructs it from a replay record.
path::options path_options_from_replay(const viam::trajex::totg::planner_base::config& cfg) {
    auto opts = path::options{}.set_max_blend_deviation(cfg.path_blend_tolerance);
    if (cfg.colinearization_ratio) {
        opts.set_max_linear_deviation(cfg.path_blend_tolerance * *cfg.colinearization_ratio);
    }
    if (cfg.min_blend_curvature) {
        opts.set_min_blend_curvature(*cfg.min_blend_curvature);
    }
    if (cfg.max_blend_curvature) {
        opts.set_max_blend_curvature(*cfg.max_blend_curvature);
    }
    return opts;
}

trajectory::options trajectory_options_from_replay(const viam::trajex::totg::planner_base::config& cfg) {
    trajectory::options topts;
    topts.max_velocity = cfg.velocity_limits;
    topts.max_acceleration = cfg.acceleration_limits;
    return topts;
}

struct cell_result {
    double w_c{};
    double w_r{};
    int rebases{};
    std::optional<int> starved_at_waypoint;
    // True when the cell failure was a trajex exception during extend()/rebase, not a
    // genuine starve. Currently lumped into starved_at_waypoint for the CSV schema's
    // sake; logged to stderr so the operator can see the distinction. The CSV reader
    // sees a red cell either way.
    bool extend_threw{false};
    std::string extend_throw_message;
};

cell_result simulate_cell(const xt::xarray<double>& workload,
                          const path::options& popts,
                          const trajectory::options& topts,
                          double sample_rate_hz,
                          double w_c,
                          double w_r,
                          double speed_factor,
                          std::size_t batch_size) {
    cell_result result;
    result.w_c = w_c;
    result.w_r = w_r;

    const std::size_t n = workload.shape(0);
    const double sample_period = 1.0 / sample_rate_hz;

    // Declared above the try so the catch can report which waypoint we were processing
    // when trajex threw. A trajex exception during extend() or during the rebase that
    // sample_next() may trigger fails the cell; the catch records the failure point in
    // starved_at_waypoint so the plot paints red. The distinction from a real starve is
    // recorded in extend_threw and logged to stderr.
    std::size_t next_wp_idx = 0;
    std::size_t active_wp_count = 0;
    std::size_t staged_wp_count = 0;

    try {
        streaming::session sess(popts, topts, types::hertz{sample_rate_hz});

        // Bootstrap: the first extend must deliver at least two waypoints to make a valid
        // trajectory. Use exactly two so subsequent batches can default to batch_size=2 with
        // one new waypoint each.
        {
            const xt::xarray<double> bootstrap = xt::view(workload, xt::range(std::size_t{0}, std::size_t{2}), xt::all());
            const waypoint_accumulator acc(bootstrap);
            sess.extend(acc);
            next_wp_idx = 2;
            active_wp_count = 2;
        }

        // Pre-fill the watermark to one replan-budget ahead of t=0, mirroring a real consumer
        // that primes its buffer before letting the arm begin executing. Without this, the
        // very first tick would advance the arm into the future before any samples exist.
        {
            const double target = w_r;
            while (sess.current_time().count() < target) {
                const auto pre = sess.trajectory_generation_count();
                auto s = sess.sample_next(1);
                const auto post = sess.trajectory_generation_count();
                if (post > pre) {
                    result.rebases += static_cast<int>(post - pre);
                    active_wp_count += staged_wp_count;
                    staged_wp_count = 0;
                }
                if (s.empty()) {
                    break;
                }
            }
        }

        double arm_time = 0.0;

        // Steady-state loop. Termination conditions:
        //   - All workload waypoints delivered AND staging empty AND arm reached active terminal.
        //   - Starve detected during an extend() call (returns early).
        //   - Session unexpectedly drained (defensive break).
        auto active_terminal_global = [&]() {
            const auto* a = sess.active_trajectory();
            return sess.active_epoch().count() + (a ? a->duration().count() : 0.0);
        };

        while (true) {
            const bool all_delivered = (next_wp_idx >= n);
            const bool staging_empty = (staged_wp_count == 0);
            if (all_delivered && staging_empty && arm_time >= active_terminal_global()) {
                break;
            }

            // Tick: advance arm by one sample period.
            arm_time += sample_period;

            // Refill the watermark to >= arm_time + w_r. Each sample_next(1) may trigger a
            // rebase internally; track the generation count delta to count rebases.
            while (sess.current_time().count() < arm_time + w_r) {
                const auto pre = sess.trajectory_generation_count();
                auto s = sess.sample_next(1);
                const auto post = sess.trajectory_generation_count();
                if (post > pre) {
                    result.rebases += static_cast<int>(post - pre);
                    active_wp_count += staged_wp_count;
                    staged_wp_count = 0;
                }
                if (s.empty()) {
                    break;
                }
            }

            // Deliver waypoints while the commit window has headroom.
            while (next_wp_idx < n) {
                const auto* active = sess.active_trajectory();
                if (active == nullptr) {
                    break;  // shouldn't happen after bootstrap, but guard for safety
                }
                const double active_dur = active->duration().count();
                const double per_wp_dur = active_dur / static_cast<double>(std::max<std::size_t>(active_wp_count, 1));
                const double est_staged_dur = static_cast<double>(staged_wp_count) * per_wp_dur;
                const double est_commit_horizon_global = sess.active_epoch().count() + active_dur + est_staged_dur;
                const double headroom = est_commit_horizon_global - arm_time;
                if (headroom >= w_c) {
                    break;
                }

                // Build the next batch: [seam_point, new_waypoints...]. Seam is at index
                // (next_wp_idx - 1); batch covers [next_wp_idx - 1, batch_end).
                const std::size_t batch_start = next_wp_idx - 1;
                const std::size_t batch_end = std::min(next_wp_idx + batch_size - 1, n);
                const xt::xarray<double> batch_data = xt::view(workload, xt::range(batch_start, batch_end), xt::all());
                const waypoint_accumulator acc(batch_data);

                const double watermark_before = sess.current_time().count();
                const auto pre_gen = sess.trajectory_generation_count();

                const auto start = std::chrono::steady_clock::now();
                sess.extend(acc);  // may throw; let it propagate up
                const auto stop = std::chrono::steady_clock::now();
                const double elapsed_real = std::chrono::duration<double>(stop - start).count();
                const double arm_advance = elapsed_real * speed_factor;

                // Starve check: if the arm would advance past the watermark during this extend,
                // the consumer is asking for a sample the session hasn't produced. The starve
                // point is recorded as the waypoint index this extend was delivering.
                if (arm_time + arm_advance > watermark_before) {
                    result.starved_at_waypoint = static_cast<int>(next_wp_idx);
                    return result;
                }
                arm_time += arm_advance;

                const auto post_gen = sess.trajectory_generation_count();
                const std::size_t new_wps = batch_end - next_wp_idx;
                if (post_gen > pre_gen) {
                    active_wp_count += new_wps;
                } else {
                    staged_wp_count += new_wps;
                }
                next_wp_idx = batch_end;
            }
        }
    } catch (const std::exception& e) {
        result.extend_threw = true;
        result.extend_throw_message = e.what();
        result.starved_at_waypoint = static_cast<int>(next_wp_idx);
    }

    return result;
}

void write_csv(const sim_config& cfg, std::size_t n_waypoints, const std::vector<cell_result>& cells) {
    std::ofstream out(cfg.output_path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + cfg.output_path.string());
    }
    out << "# workload: " << cfg.replay_path.filename().string() << "\n";
    out << "# n_waypoints: " << n_waypoints << "\n";
    out << "# sample_rate_hz: " << cfg.sample_rate_hz << "\n";
    out << "# speed_factor: " << cfg.speed_factor << "\n";
    out << "# batch_size: " << cfg.batch_size << "\n";
    out << "commit_window,replan_budget,rebases,starved_at_waypoint\n";
    for (const auto& c : cells) {
        out << c.w_c << "," << c.w_r << "," << c.rebases << ",";
        if (c.starved_at_waypoint) {
            out << *c.starved_at_waypoint;
        }
        out << "\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) try {
    const sim_config cfg = parse_args(argc, argv);

    auto [planner_cfg, workload] = parse_replay_record(cfg.replay_path);
    const std::size_t n = workload.shape(0);
    if (n < 2) {
        throw std::runtime_error("replay record must contain at least 2 waypoints");
    }

    const auto popts = path_options_from_replay(planner_cfg);
    const auto topts = trajectory_options_from_replay(planner_cfg);

    const auto w_c_values = geometric_grid(cfg.w_c_min, cfg.w_c_max, cfg.w_c_count);
    const auto w_r_values = geometric_grid(cfg.w_r_min, cfg.w_r_max, cfg.w_r_count);

    std::vector<cell_result> cells;
    cells.reserve(w_c_values.size() * w_r_values.size());

    const std::size_t total = w_c_values.size() * w_r_values.size();
    std::size_t done = 0;
    for (const double w_c : w_c_values) {
        for (const double w_r : w_r_values) {
            ++done;
            std::cerr << "[" << done << "/" << total << "] W_c=" << w_c << " W_r=" << w_r << " ... " << std::flush;
            cell_result r = simulate_cell(workload, popts, topts, cfg.sample_rate_hz, w_c, w_r, cfg.speed_factor, cfg.batch_size);
            cells.push_back(r);
            std::cerr << "rebases=" << r.rebases;
            if (r.extend_threw) {
                std::cerr << " EXTEND-THREW@" << *r.starved_at_waypoint << ": " << r.extend_throw_message;
            } else if (r.starved_at_waypoint) {
                std::cerr << " STARVED@" << *r.starved_at_waypoint;
            }
            std::cerr << "\n";
        }
    }

    write_csv(cfg, n, cells);
    std::cerr << "wrote " << cfg.output_path.string() << "\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
