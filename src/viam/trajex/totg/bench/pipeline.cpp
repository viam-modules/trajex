// Benchmarks for the core trajex pipeline: waypoint accumulation, the store a session keeps
// them in, path construction, trajectory generation, and sampling.
//
// Driven from canonical JSON replay records so the geometry and the limits are ones a real
// arm produced. Records are parsed once into a fixture and only the pipeline stages
// themselves are timed.
//
// Options are built the way `run_totg_` builds them (planner.hpp), minus the observer:
// production attaches none, and an attached observer allocates per event.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xview.hpp>
#endif

#include <viam/trajex/totg/bench/benchmarks.hpp>
#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/streaming/private/waypoint_store.hpp>
#include <viam/trajex/totg/tools/planner.hpp>
#include <viam/trajex/totg/tools/replay.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/hertz.hpp>
#include <viam/trajex/types/xt.hpp>

namespace {

using viam::trajex::xmatrix;
using viam::trajex::xvector;
using viam::trajex::totg::parse_replay_record;
using viam::trajex::totg::path;
using viam::trajex::totg::planner_base;
using viam::trajex::totg::trajectory;
using viam::trajex::totg::uniform_sampler;
using viam::trajex::totg::waypoint_accumulator;

struct workload {
    const char* label;
    const char* filename;
};

constexpr workload k_workloads[] = {
    {"lab_sander", "lab_sander_backward_integration_exceeded-20260507.trajex-totg-replay.json"},
    {"VIK-182-stall", "VIK-182-stall.trajex-totg-replay.json"},
};

// Representative of what the universal-robots module samples at; its configured range is
// 1 to 500 Hz.
constexpr double k_sampling_freq_hz = 100.0;

// Unmeasured iterations run before the measured ones. Process startup, first-touch page
// faults, and allocator warmup otherwise land entirely in whichever benchmark happens to
// run first, which showed up as a 940x inflation of the first cell.
constexpr double k_warmup_seconds = 0.1;

// Prefix lengths to sweep, filtered per record to those smaller than it. Each record's own
// waypoint count is appended, so every record ends its sweep at full size and those columns
// line up across records.
constexpr std::int64_t k_prefix_sizes[] = {64, 256, 1024, 4096};

// The store is swept over batch size as well as total waypoints, because it is the only
// stage whose cost depends on how the waypoints arrived. B = 1 is point-by-point, which is
// common in practice.
constexpr std::int64_t k_store_sizes[] = {1024, 4096};
constexpr std::int64_t k_store_batches[] = {1, 8, 64, 256};

struct record {
    planner_base::config config;
    xmatrix<> waypoints;
};

// Parsed on first request and cached. Registration asks each record for its waypoint count,
// so both are parsed up front; the cache keeps the five benchmarks per record from parsing
// it five times. JSON parsing is not part of any stage we are measuring.
const record& loaded_record(const std::string& filename) {
    static std::map<std::string, record> cache;

    const auto found = cache.find(filename);
    if (found != cache.end()) {
        return found->second;
    }

    auto [config, waypoints] = parse_replay_record(std::filesystem::path{VIAM_TRAJEX_BENCH_DATA_DIR} / filename);
    return cache.emplace(filename, record{std::move(config), std::move(waypoints)}).first->second;
}

std::size_t waypoint_count(const std::string& filename) {
    return loaded_record(filename).waypoints.shape(0);
}

std::vector<std::int64_t> sizes_for(std::span<const std::int64_t> candidates, std::size_t total) {
    std::vector<std::int64_t> sizes;
    for (const auto candidate : candidates) {
        if (std::cmp_less(candidate, total)) {
            sizes.push_back(candidate);
        }
    }
    sizes.push_back(static_cast<std::int64_t>(total));
    return sizes;
}

xmatrix<> slice_of(const xmatrix<>& waypoints, std::size_t first, std::size_t last) {
    return xmatrix<>{xt::view(waypoints, xt::range(first, last), xt::all())};
}

xmatrix<> prefix_of(const xmatrix<>& waypoints, std::size_t count) {
    return slice_of(waypoints, 0, count);
}

path::options path_options_for(const planner_base::config& config) {
    path::options options;
    options.set_max_blend_deviation(config.path_blend_tolerance);
    if (config.colinearization_ratio) {
        options.set_max_linear_deviation(config.path_blend_tolerance * *config.colinearization_ratio);
    }
    if (config.min_blend_curvature) {
        options.set_min_blend_curvature(*config.min_blend_curvature);
    }
    if (config.max_blend_curvature) {
        options.set_max_blend_curvature(*config.max_blend_curvature);
    }
    return options;
}

trajectory::options trajectory_options_for(const planner_base::config& config) {
    trajectory::options options;
    options.max_velocity = config.velocity_limits;
    options.max_acceleration = config.acceleration_limits;
    options.tcp = config.tcp;
    return options;
}

void bm_accumulate(benchmark::State& state, const std::string& filename) {
    const auto& source = loaded_record(filename);
    const auto waypoints = prefix_of(source.waypoints, static_cast<std::size_t>(state.range(0)));

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        waypoint_accumulator accumulator{waypoints};
        benchmark::DoNotOptimize(accumulator);
    }
}

// Models a session accumulating `state.range(0)` waypoints in batches of `state.range(1)`.
//
// A batch arrives carrying the previous batch's last waypoint as a seam, which the session
// strips by passing `from = 1`, so a batch delivering B new waypoints is an accumulator of
// B + 1 rows.
void bm_waypoint_store(benchmark::State& state, const std::string& filename) {
    const auto& source = loaded_record(filename);
    const auto total = static_cast<std::size_t>(state.range(0));
    const auto batch_size = static_cast<std::size_t>(state.range(1));
    const auto waypoints = prefix_of(source.waypoints, total);

    std::vector<xmatrix<>> storage;
    for (std::size_t first = 0; first + 1 < total; first += batch_size) {
        storage.push_back(slice_of(waypoints, first, std::min(total, first + batch_size + 1)));
    }

    std::vector<waypoint_accumulator> batches;
    batches.reserve(storage.size());
    for (const auto& owned : storage) {
        batches.emplace_back(owned);
    }

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        viam::trajex::totg::streaming::waypoint_store store;
        store.append(batches.front(), 0);
        for (std::size_t i = 1; i != batches.size(); ++i) {
            store.append(batches[i], 1);
        }
        benchmark::DoNotOptimize(store.size());
    }
}

void bm_path_create(benchmark::State& state, const std::string& filename) {
    const auto& source = loaded_record(filename);
    const auto waypoints = prefix_of(source.waypoints, static_cast<std::size_t>(state.range(0)));
    const waypoint_accumulator accumulator{waypoints};
    const auto options = path_options_for(source.config);

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        auto p = path::create(accumulator, options);
        benchmark::DoNotOptimize(p);
    }
}

void bm_trajectory_create(benchmark::State& state, const std::string& filename) {
    const auto& source = loaded_record(filename);
    const auto waypoints = prefix_of(source.waypoints, static_cast<std::size_t>(state.range(0)));
    const waypoint_accumulator accumulator{waypoints};
    const auto p = path::create(accumulator, path_options_for(source.config));

    // `trajectory::create` consumes the path and the options, so each iteration needs its
    // own copies. Rebuilding them is setup, not the thing being measured, so it happens
    // with the clock stopped.
    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        state.PauseTiming();
        auto path_copy = p;
        auto options = trajectory_options_for(source.config);
        state.ResumeTiming();

        auto traj = trajectory::create(std::move(path_copy), std::move(options));
        benchmark::DoNotOptimize(traj);
    }
}

void bm_sample(benchmark::State& state, const std::string& filename) {
    const auto& source = loaded_record(filename);
    const auto waypoints = prefix_of(source.waypoints, static_cast<std::size_t>(state.range(0)));
    const waypoint_accumulator accumulator{waypoints};
    auto p = path::create(accumulator, path_options_for(source.config));
    const auto traj = trajectory::create(std::move(p), trajectory_options_for(source.config));

    // Read every joint of position, velocity, and acceleration out of each sample, the way
    // the universal-robots module does when it converts samples to spline points. Sampling
    // into a discarded value would measure something production never does.
    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        auto sampler = uniform_sampler::quantized_for_trajectory(traj, viam::trajex::types::hertz{k_sampling_freq_hz});
        double sink = 0.0;
        for (const auto& sample : traj.samples(sampler)) {
            for (std::size_t joint = 0; joint != sample.configuration.size(); ++joint) {
                sink += sample.configuration(joint) + sample.velocity(joint) + sample.acceleration(joint);
            }
        }
        benchmark::DoNotOptimize(sink);
    }
}

void bm_sample_collect(benchmark::State& state, const std::string& filename) {
    const auto& source = loaded_record(filename);
    const auto waypoints = prefix_of(source.waypoints, static_cast<std::size_t>(state.range(0)));
    const waypoint_accumulator accumulator{waypoints};
    auto p = path::create(accumulator, path_options_for(source.config));
    const auto traj = trajectory::create(std::move(p), trajectory_options_for(source.config));

    const auto n_dof = traj.path().dof();
    const auto n_samples = uniform_sampler::calculate_quantized_samples(traj.duration().count(), k_sampling_freq_hz);

    // Marshal each sample into the flat per-quantity arrays a caller receives when it wants a
    // trajectory whole rather than a sample at a time. `bm_sample` above reads the same values
    // into a scalar; the difference between the two is what the row writes cost.
    //
    // The destinations are built once. Allocating three arrays of this size per iteration
    // would measure the allocation rather than the row writes, and the row writes are what
    // scales with the length of the trajectory.
    xvector<> times(xvector<>::shape_type{n_samples});
    xmatrix<> configurations(xmatrix<>::shape_type{n_samples, n_dof});
    xmatrix<> velocities(xmatrix<>::shape_type{n_samples, n_dof});
    xmatrix<> accelerations(xmatrix<>::shape_type{n_samples, n_dof});

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        auto sampler = uniform_sampler::quantized_for_trajectory(traj, viam::trajex::types::hertz{k_sampling_freq_hz});

        std::size_t idx = 0;
        for (const auto& sample : traj.samples(sampler)) {
            times(idx) = sample.time.count();
            xt::view(configurations, idx, xt::all()) = sample.configuration;
            xt::view(velocities, idx, xt::all()) = sample.velocity;
            xt::view(accelerations, idx, xt::all()) = sample.acceleration;
            ++idx;
        }

        // Naming the destinations individually, not just clobbering memory: the copy
        // benchmarks in xtensor_idioms.cpp measured a deleted loop at one cycle for 12 KB
        // because ClobberMemory alone left the optimiser free to drop the stores.
        benchmark::DoNotOptimize(times.data());
        benchmark::DoNotOptimize(configurations.data());
        benchmark::DoNotOptimize(velocities.data());
        benchmark::DoNotOptimize(accelerations.data());
        benchmark::ClobberMemory();
    }
}

// Benchmarks are registered at runtime rather than with the BENCHMARK macro so each one can
// be named for the record it runs against: `bm_path_create/VIK-182-stall/4096`. That makes
// --benchmark_filter select by stage or by record, whichever is wanted.
void register_workloads() {
    for (const auto& item : k_workloads) {
        const std::string filename = item.filename;
        const auto total = waypoint_count(filename);
        const auto prefixes = sizes_for(k_prefix_sizes, total);

        const auto register_stage = [&](const char* stage, auto function, benchmark::TimeUnit unit) {
            auto* registered = benchmark::RegisterBenchmark(std::string{stage} + "/" + item.label,
                                                            [filename, function](benchmark::State& state) { function(state, filename); });
            for (const auto size : prefixes) {
                registered->Arg(size);
            }
            registered->Unit(unit)->MinWarmUpTime(k_warmup_seconds);
        };

        register_stage("bm_accumulate", bm_accumulate, benchmark::kMicrosecond);
        register_stage("bm_path_create", bm_path_create, benchmark::kMillisecond);
        register_stage("bm_trajectory_create", bm_trajectory_create, benchmark::kMillisecond);
        register_stage("bm_sample", bm_sample, benchmark::kMillisecond);
        register_stage("bm_sample_collect", bm_sample_collect, benchmark::kMillisecond);

        const auto store_sizes = sizes_for(k_store_sizes, total);
        const auto register_store = [&](const char* stage, auto function) {
            auto* registered = benchmark::RegisterBenchmark(std::string{stage} + "/" + item.label,
                                                            [filename, function](benchmark::State& state) { function(state, filename); });
            for (const auto size : store_sizes) {
                for (const auto batch : k_store_batches) {
                    registered->Args({size, batch});
                }
            }
            registered->Unit(benchmark::kMillisecond)->MinWarmUpTime(k_warmup_seconds);
        };

        register_store("bm_waypoint_store", bm_waypoint_store);
    }
}

}  // namespace

namespace viam::trajex::totg::bench {

void register_pipeline_benchmarks() {
    register_workloads();
}

}  // namespace viam::trajex::totg::bench
