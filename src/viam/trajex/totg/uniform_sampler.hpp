#pragma once

#include <optional>

#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/hertz.hpp>

namespace viam::trajex::totg {

///
/// Uniform time-step sampler.
///
/// Samples a trajectory at regular time intervals over the closed range `[start, duration]`,
/// where `start` defaults to zero and `duration` is the trajectory's duration. Non-zero
/// `start` values support chaining samplers across trajectory boundaries without
/// duplicating the seam sample.
///
class uniform_sampler {
   public:
    ///
    /// Constructs uniform sampler with sample count and a starting time.
    ///
    /// @param num_samples Total number of samples to generate
    /// @param start Time at which the first sample lands. Defaults to zero.
    ///
    explicit uniform_sampler(std::size_t num_samples, trajectory::seconds start = trajectory::seconds{0.0});

    ///
    /// Creates uniform sampler with adjusted timestep to hit duration endpoint exactly.
    ///
    /// Calculates number of samples needed for the given frequency, rounds up to ensure
    /// endpoint coverage, then adjusts dt to land exactly on duration.
    /// Slightly oversamples to guarantee endpoint hit.
    ///
    /// @param duration Duration to quantize over
    /// @param frequency Desired sampling frequency
    /// @return uniform_sampler with dt adjusted to align with duration
    ///
    static uniform_sampler quantized_for_duration(trajectory::seconds duration, types::hertz frequency);

    ///
    /// Creates uniform sampler whose first sample lands at `start` and whose last sample
    /// lands exactly at the trajectory's duration.
    ///
    /// Quantization adjusts the internal sample spacing so an integer number of samples
    /// covers `[start, traj.duration()]` with both endpoints hit. The effective dt is
    /// approximately `1 / frequency` but is adjusted per-call to fit the requested span.
    ///
    /// With the default `start = 0`, this is the standard "sample the whole trajectory"
    /// behavior. Non-zero `start` lets a caller chain samplers across trajectory
    /// transitions by skipping past the seam (which the previous sampler already emitted
    /// as its terminal sample).
    ///
    /// @param traj Trajectory to sample
    /// @param frequency Target sample rate, subject to per-trajectory quantization
    /// @param start Time at which the first sample lands. Defaults to zero.
    /// @return uniform_sampler covering [start, traj.duration()]
    /// @throws std::invalid_argument if `start` is negative or is not strictly less than the trajectory's duration
    ///
    static uniform_sampler quantized_for_trajectory(const trajectory& traj,
                                                    types::hertz frequency,
                                                    trajectory::seconds start = trajectory::seconds{0.0});

    ///
    /// Calculates adjusted timestep for quantized sampling.
    ///
    /// Exposes the core quantization calculation for testing and custom use.
    ///
    /// @param duration_sec Duration in seconds
    /// @param frequency_hz Frequency in Hz
    /// @return Adjusted timestep in seconds
    /// @throws std::invalid_argument if values are non-positive or exceed limits
    ///
    static double calculate_quantized_dt(double duration_sec, double frequency_hz);

    ///
    /// Calculates number of samples for quantized sampling.
    ///
    /// Exposes the sample count calculation for testing and custom use.
    ///
    /// @param duration_sec Duration in seconds
    /// @param frequency_hz Frequency in Hz
    /// @return Number of samples that will be generated
    /// @throws std::invalid_argument if values are non-positive or exceed limits
    ///
    static std::size_t calculate_quantized_samples(double duration_sec, double frequency_hz);

    ///
    /// Gets next sample, advancing cursor by dt.
    ///
    /// @param cursor Cursor to sample and advance
    /// @return Sample at current cursor time, or nullopt if past trajectory end
    ///
    std::optional<struct trajectory::sample> next(trajectory::cursor& cursor);

   private:
    std::size_t num_samples_;
    std::size_t next_sample_ = 0;
    trajectory::seconds start_;
};

// Verify that uniform_sampler satisfies the sampler concept
static_assert(trajectory_details::sampler<uniform_sampler, trajectory::cursor, struct trajectory::sample>);

}  // namespace viam::trajex::totg
