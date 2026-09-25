// Uniform sampler tests
// Extracted from test.cpp lines 2062-2141

#include <cstddef>

#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/uniform_sampler.hpp>
#include <viam/trajex/types/arc_length.hpp>
#include <viam/trajex/types/arc_velocity.hpp>
#include <viam/trajex/types/xt.hpp>

#include <boost/test/unit_test.hpp>

using viam::trajex::xmatrix;

BOOST_AUTO_TEST_SUITE(uniform_sampler_tests)

BOOST_AUTO_TEST_CASE(calculate_quantized_dt_basic) {
    using namespace viam::trajex::totg;

    // 1.0 second @ 100 Hz should give 101 samples
    // dt = 1.0 / (101 - 1) = 1.0 / 100 = 0.01
    const double dt = uniform_sampler::calculate_quantized_dt(1.0, 100.0);
    BOOST_CHECK_CLOSE(dt, 0.01, 0.001);  // Within 0.001% tolerance
}

BOOST_AUTO_TEST_CASE(calculate_quantized_dt_ensures_endpoint_hit) {
    using namespace viam::trajex::totg;

    // 1.01 seconds @ 100 Hz
    // putative_samples = 1.01 * 100 = 101.0
    // num_samples = ceil(101.0) + 1 = 102
    // dt = 1.01 / (102 - 1) = 1.01 / 101 ≈ 0.01
    const double dt = uniform_sampler::calculate_quantized_dt(1.01, 100.0);

    // Verify we can reach exactly 1.01 with 101 steps
    const double endpoint = 101 * dt;
    BOOST_CHECK_CLOSE(endpoint, 1.01, 0.001);
}

BOOST_AUTO_TEST_CASE(calculate_quantized_dt_oversamples) {
    using namespace viam::trajex::totg;

    // 0.99 seconds @ 100 Hz
    // putative_samples = 0.99 * 100 = 99.0
    // num_samples = ceil(99.0) + 1 = 100
    // dt = 0.99 / (100 - 1) = 0.99 / 99 = 0.01
    const double dt = uniform_sampler::calculate_quantized_dt(0.99, 100.0);

    // Should slightly oversample (100 samples instead of 99)
    const int num_samples = static_cast<int>(std::ceil(0.99 / dt)) + 1;
    BOOST_CHECK_EQUAL(num_samples, 100);
}

BOOST_AUTO_TEST_CASE(calculate_quantized_dt_invalid_duration) {
    using namespace viam::trajex::totg;

    // Zero duration
    BOOST_CHECK_THROW(uniform_sampler::calculate_quantized_dt(0.0, 100.0), std::invalid_argument);

    // Negative duration
    BOOST_CHECK_THROW(uniform_sampler::calculate_quantized_dt(-1.0, 100.0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(calculate_quantized_dt_invalid_frequency) {
    using namespace viam::trajex::totg;

    // Zero frequency
    BOOST_CHECK_THROW(uniform_sampler::calculate_quantized_dt(1.0, 0.0), std::invalid_argument);

    // Negative frequency
    BOOST_CHECK_THROW(uniform_sampler::calculate_quantized_dt(1.0, -100.0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(calculate_quantized_dt_at_least_two_samples) {
    using namespace viam::trajex::totg;

    // Very small duration and frequency
    const double dt = uniform_sampler::calculate_quantized_dt(0.001, 1.0);

    // Should still get at least 2 samples (start and end)
    // putative = 0.001 * 1 = 0.001
    // num_samples = ceil(0.001) + 1 = 2
    // dt = 0.001 / (2 - 1) = 0.001
    BOOST_CHECK_CLOSE(dt, 0.001, 0.001);
}

BOOST_AUTO_TEST_CASE(no_duplicate_timestamps_at_end) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    using viam::trajex::arc_acceleration;
    using viam::trajex::arc_length;
    using viam::trajex::arc_velocity;

    // Create a simple path
    const xmatrix<> waypoints = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    path p = path::create(waypoints);

    // Create trajectory with explicit integration points to have precise control over duration.
    // We choose duration = 1.0 seconds sampled at 10Hz to trigger floating-point accumulation error.
    // With 10Hz over 1.0s, we get 11 samples with dt = 1.0/10 = 0.1
    // The value 0.1 cannot be represented exactly in binary floating point (infinite binary expansion).
    // After accumulating 10 steps of 0.1 (binary approximation), we fall SLIGHTLY SHORT of 1.0,
    // causing the sampler to snap to exactly 1.0, then attempt to sample again at 1.0 with a
    // tiny delta (duplicate timestamp bug).
    const trajectory::seconds target_duration{1.0};

    std::vector<trajectory::integration_point> points = {
        {.time = trajectory::seconds{0.0}, .s = arc_length{0.0}, .s_dot = arc_velocity{0.0}, .s_ddot = arc_acceleration{0.5}},
        {.time = target_duration, .s = p.length(), .s_dot = arc_velocity{0.0}, .s_ddot = arc_acceleration{0.0}}};

    const trajectory::options opts{.max_velocity = xt::ones<double>({3}), .max_acceleration = xt::ones<double>({3})};

    const trajectory traj = trajectory::create(std::move(p), opts, std::move(points));

    // Verify we got the expected duration
    BOOST_CHECK_CLOSE(traj.duration().count(), target_duration.count(), 0.001);

    // Create quantized sampler at exactly 10Hz (produces 11 samples, dt = 0.1)
    // This is the classic case where 0.1 * 10 != 1.0 in floating point
    const auto sample_freq = hertz{10.0};
    auto sampler = uniform_sampler::quantized_for_trajectory(traj, sample_freq);
    auto cursor = traj.create_cursor();

    // Collect all samples and their time deltas
    std::vector<trajectory::seconds> timestamps;
    std::vector<double> deltas;

    std::optional<trajectory::seconds> prev_time;
    while (auto sample = sampler.next(cursor)) {
        timestamps.push_back(sample->time);
        if (prev_time) {
            deltas.push_back((sample->time - *prev_time).count());
        }
        prev_time = sample->time;
    }

    // 1. Basic sanity: got at least 2 samples
    BOOST_REQUIRE_GE(timestamps.size(), 2);

    // 2. Check we got EXACTLY the expected number of samples (no duplicates)
    const size_t expected_samples = static_cast<size_t>(std::ceil(traj.duration().count() * sample_freq.value)) + 1;
    BOOST_CHECK_MESSAGE(
        timestamps.size() == expected_samples,
        "Got " << timestamps.size() << " samples but expected " << expected_samples << " (extra samples indicate duplicate timestamps)");

    // 3. Verify first sample is EXACTLY at t=0
    BOOST_CHECK_EQUAL(timestamps.front().count(), 0.0);

    // 4. Verify last sample is EXACTLY at trajectory duration
    BOOST_CHECK_EQUAL(timestamps.back().count(), traj.duration().count());

    // 5. Verify all timestamps are strictly monotonically increasing (no duplicates)
    for (size_t i = 1; i < timestamps.size(); ++i) {
        BOOST_CHECK_MESSAGE(timestamps[i] > timestamps[i - 1],
                            "Duplicate or non-monotonic timestamp at index " << i << ": t[" << i - 1 << "]=" << timestamps[i - 1].count()
                                                                             << "s, t[" << i << "]=" << timestamps[i].count() << "s");
    }

    // 6. Verify all deltas are close to the expected uniform delta
    // This will FAIL if we get duplicate timestamps (delta ~= 0) or uneven spacing
    const double expected_delta = uniform_sampler::calculate_quantized_dt(traj.duration().count(), sample_freq.value);
    for (size_t i = 0; i < deltas.size(); ++i) {
        BOOST_CHECK_CLOSE(deltas[i], expected_delta, 0.01);  // Within 0.01% tolerance
    }
}

namespace {

viam::trajex::totg::trajectory build_unit_duration_trajectory() {
    using namespace viam::trajex::totg;
    using viam::trajex::arc_acceleration;
    using viam::trajex::arc_length;
    using viam::trajex::arc_velocity;

    const xmatrix<> waypoints = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    path p = path::create(waypoints);

    std::vector<trajectory::integration_point> points = {
        {.time = trajectory::seconds{0.0}, .s = arc_length{0.0}, .s_dot = arc_velocity{0.0}, .s_ddot = arc_acceleration{0.5}},
        {.time = trajectory::seconds{1.0}, .s = p.length(), .s_dot = arc_velocity{0.0}, .s_ddot = arc_acceleration{0.0}}};

    const trajectory::options opts{.max_velocity = xt::ones<double>({3}), .max_acceleration = xt::ones<double>({3})};

    return trajectory::create(std::move(p), opts, std::move(points));
}

}  // namespace

BOOST_AUTO_TEST_CASE(quantized_for_trajectory_with_start_emits_first_sample_at_start) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    const trajectory traj = build_unit_duration_trajectory();
    const auto start = trajectory::seconds{0.1};

    auto sampler = uniform_sampler::quantized_for_trajectory(traj, hertz{10.0}, start);
    auto cursor = traj.create_cursor();

    const auto first = sampler.next(cursor);
    BOOST_REQUIRE(first.has_value());
    BOOST_CHECK_EQUAL(first->time.count(), start.count());
}

BOOST_AUTO_TEST_CASE(quantized_for_trajectory_with_start_emits_last_sample_at_duration) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    const trajectory traj = build_unit_duration_trajectory();
    const auto start = trajectory::seconds{0.1};

    auto sampler = uniform_sampler::quantized_for_trajectory(traj, hertz{10.0}, start);
    auto cursor = traj.create_cursor();

    std::optional<struct trajectory::sample> last;
    while (auto s = sampler.next(cursor)) {
        last = s;
    }
    BOOST_REQUIRE(last.has_value());
    BOOST_CHECK_EQUAL(last->time.count(), traj.duration().count());
}

BOOST_AUTO_TEST_CASE(quantized_for_trajectory_throws_on_negative_start) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    const trajectory traj = build_unit_duration_trajectory();
    BOOST_CHECK_THROW(uniform_sampler::quantized_for_trajectory(traj, hertz{10.0}, trajectory::seconds{-0.1}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(quantized_for_trajectory_throws_on_start_at_duration) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    const trajectory traj = build_unit_duration_trajectory();
    BOOST_CHECK_THROW(uniform_sampler::quantized_for_trajectory(traj, hertz{10.0}, traj.duration()), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(quantized_for_trajectory_throws_on_start_beyond_duration) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    const trajectory traj = build_unit_duration_trajectory();
    const auto past = traj.duration() + trajectory::seconds{0.5};
    BOOST_CHECK_THROW(uniform_sampler::quantized_for_trajectory(traj, hertz{10.0}, past), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(cursor_fill_sample_matches_value_sample) {
    using namespace viam::trajex::totg;

    const trajectory traj = build_unit_duration_trajectory();

    auto value_cursor = traj.create_cursor();
    auto fill_cursor = traj.create_cursor();

    // Reused across iterations deliberately: after the first fill its arrays are already
    // correctly shaped, which is the path the range takes for every sample but the first.
    struct trajectory::sample filled;

    for (int step = 0; step <= 10; ++step) {
        const auto t = trajectory::seconds{static_cast<double>(step) / 10.0};

        value_cursor.seek(t);
        fill_cursor.seek(t);

        const auto expected = value_cursor.sample();
        fill_cursor.sample(filled);

        BOOST_CHECK_EQUAL(filled.time.count(), expected.time.count());
        BOOST_REQUIRE_EQUAL(filled.configuration.size(), expected.configuration.size());

        for (std::size_t i = 0; i != expected.configuration.size(); ++i) {
            BOOST_CHECK_EQUAL(filled.configuration(i), expected.configuration(i));
            BOOST_CHECK_EQUAL(filled.velocity(i), expected.velocity(i));
            BOOST_CHECK_EQUAL(filled.acceleration(i), expected.acceleration(i));
        }
    }
}

BOOST_AUTO_TEST_CASE(range_refills_sample_storage_in_place) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    const trajectory traj = build_unit_duration_trajectory();
    auto range = traj.samples(uniform_sampler::quantized_for_trajectory(traj, hertz{10.0}));

    auto it = range.begin();
    BOOST_REQUIRE(it != range.end());

    // The iterator refills one sample rather than replacing it, so its storage must not move
    // as the range advances. Going back to assigning a freshly built sample per step would
    // relocate these and cost three allocations per sample, which is the whole point.
    const double* const configuration_storage = (*it).configuration.data();
    const double* const velocity_storage = (*it).velocity.data();
    const double* const acceleration_storage = (*it).acceleration.data();

    std::size_t seen = 1;
    for (++it; it != range.end(); ++it) {
        BOOST_CHECK_EQUAL((*it).configuration.data(), configuration_storage);
        BOOST_CHECK_EQUAL((*it).velocity.data(), velocity_storage);
        BOOST_CHECK_EQUAL((*it).acceleration.data(), acceleration_storage);
        ++seen;
    }

    BOOST_CHECK_GT(seen, std::size_t{1});
}

BOOST_AUTO_TEST_CASE(range_emits_expected_sample_count_then_ends) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    const trajectory traj = build_unit_duration_trajectory();
    const auto expected = uniform_sampler::calculate_quantized_samples(traj.duration().count(), 10.0);

    std::size_t seen = 0;
    for (const auto& sample : traj.samples(uniform_sampler::quantized_for_trajectory(traj, hertz{10.0}))) {
        static_cast<void>(sample);
        ++seen;
    }

    BOOST_CHECK_EQUAL(seen, expected);
}

BOOST_AUTO_TEST_CASE(advance_and_next_agree) {
    using namespace viam::trajex::totg;
    using namespace viam::trajex::types;

    const trajectory traj = build_unit_duration_trajectory();

    auto next_sampler = uniform_sampler::quantized_for_trajectory(traj, hertz{10.0});
    auto advance_sampler = uniform_sampler::quantized_for_trajectory(traj, hertz{10.0});

    auto next_cursor = traj.create_cursor();
    auto advance_cursor = traj.create_cursor();

    // `next` is documented as `advance` followed by `sample`, so the two must agree on both
    // the times visited and when they run out.
    while (const auto expected = next_sampler.next(next_cursor)) {
        BOOST_REQUIRE(advance_sampler.advance(advance_cursor));
        BOOST_CHECK_EQUAL(advance_cursor.sample().time.count(), expected->time.count());
    }

    BOOST_CHECK(!advance_sampler.advance(advance_cursor));
}

BOOST_AUTO_TEST_SUITE_END()
