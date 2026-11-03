#pragma once

#include <chrono>
#include <exception>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <viam/trajex/totg/path.hpp>

#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/totg/waypoint_utils.hpp>
#include <viam/trajex/types/xt.hpp>

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
#include <Eigen/Dense>

#include <third_party/trajectories/Path.h>
#include <third_party/trajectories/Trajectory.h>

#include <viam/trajex/totg/tools/legacy.hpp>
#endif

namespace viam::trajex::totg {

///
/// Non-template base holding config, state, and serialization for all
/// planner instantiations regardless of Receiver type.
///
class planner_base {
   public:
    ///
    /// The joint limits, path settings and optional TCP speed cap that a planner plans with.
    ///
    /// Constructing a planner validates the TCP settings. It throws std::invalid_argument if
    /// tcp_max_linear_speed is set without a kinematics_chain, or if the chain's degrees of
    /// freedom don't match the joint limits.
    ///
    struct config {
        // We initialize the optional members with {} even though clang-tidy calls that redundant,
        // because without it clang's -Wmissing-designated-field-initializers rejects any designated
        // initializer that leaves them out.
        // NOLINTBEGIN(readability-redundant-member-init)

        ///
        /// The velocity limit of each joint, in configuration-space units per second.
        ///
        xrank_checked<xvector<>> velocity_limits;

        ///
        /// The acceleration limit of each joint, in configuration-space units per second squared.
        ///
        xrank_checked<xvector<>> acceleration_limits;

        ///
        /// The furthest a blend may stray from the corner it rounds, in configuration-space units.
        /// Both generators honor it, and zero disables blending.
        ///
        double path_blend_tolerance = 0.0;

        ///
        /// When set, nearly collinear waypoints are merged before the path is built, within a
        /// tolerance of this ratio times path_blend_tolerance. Both generators honor it.
        ///
        std::optional<double> colinearization_ratio{};

        ///
        /// When set, the smallest curvature TOTG will give a blend, so that no blend radius exceeds
        /// its reciprocal. Otherwise path::options::k_default_min_blend_curvature applies.
        ///
        std::optional<double> min_blend_curvature{};

        ///
        /// When set, the largest curvature TOTG will give a blend. A corner that would need a
        /// tighter blend is left unblended. Otherwise path::options::k_default_max_blend_curvature
        /// applies.
        ///
        std::optional<double> max_blend_curvature{};

        ///
        /// Whether TOTG runs on the segmenter's segments. If false and a segmenter is registered,
        /// TOTG runs on the whole waypoint sequence instead, while the legacy generator still runs
        /// on each segment.
        ///
        bool segment_totg = true;

        ///
        /// An observer for TOTG's integration, which the planner does not own. The legacy generator
        /// ignores it.
        ///
        trajectory::integration_observer* observer = nullptr;

        ///
        /// The kinematics of the arm and anything it carries, for a TCP speed cap.
        ///
        /// A chain without a cap is valid and simply goes unused, since the arm belongs to the
        /// caller while the cap is chosen for each call. The planner keeps the chain rather than a
        /// limit built from it, because the chain is what serialize_for_replay can record.
        ///
        std::optional<kinematics::chain> kinematics_chain{};

        ///
        /// When set, the speed cap for the origin of the chain's last frame, in the length unit
        /// of the chain's translations per second. It requires kinematics_chain, and only TOTG
        /// enforces it.
        ///
        std::optional<double> tcp_max_linear_speed{};

        // NOLINTEND(readability-redundant-member-init)
    };

    ///
    /// Per-phase wall-clock timing from the most recent execute() call.
    ///
    /// Optional fields are nullopt when the corresponding phase did not run:
    /// preprocessing and validation only run when their handlers are registered,
    /// segmentation only runs when a segmenter is registered, colinearization
    /// only runs when colinearization_ratio is set, and the generation totals
    /// are nullopt when the corresponding algorithm is not enabled.
    ///
    struct timing_stats {
        std::chrono::microseconds waypoint_provisioning{};
        std::optional<std::chrono::microseconds> waypoint_preprocessing;
        std::optional<std::chrono::microseconds> move_validation;
        std::optional<std::chrono::microseconds> segmentation;
        std::optional<std::chrono::microseconds> colinearization;
        std::optional<std::chrono::microseconds> totg_generation_total;
        std::optional<std::chrono::microseconds> legacy_generation_total;
    };

    ///
    /// Returns the planner configuration.
    ///
    const struct config& get_config() const noexcept;

    ///
    /// Per-phase wall-clock timing from the most recent execute() call.
    ///
    /// Available after execute() returns.
    ///
    const timing_stats& timing() const noexcept;

    ///
    /// Number of waypoints that survived preprocessing.
    ///
    /// Available after execute() has been called. Allows the decider to
    /// distinguish "nothing to do" (< 2 waypoints) from "both generators failed."
    ///
    std::size_t processed_waypoint_count() const noexcept;

    ///
    /// Serializes the planner's configuration and the supplied waypoints to a
    /// canonical JSON replay record suitable for replay and regression testing.
    ///
    /// The record contains everything needed to reproduce a trajectory generation
    /// attempt: waypoints, velocity/acceleration limits, and path options, plus the
    /// kinematic chain and TCP cap when the config carries them.
    /// Pass e.what() on failure paths; omit on success paths.
    ///
    /// @param waypoints Waypoints to serialize
    /// @param error_message Optional error message from a failed generation
    /// @return JSON string in canonical replay record format
    ///
    std::string serialize_for_replay(const waypoint_accumulator& waypoints,
                                     std::optional<std::string_view> error_message = std::nullopt) const;

   protected:
    explicit planner_base(struct config cfg);

    struct config& mutable_config() noexcept;
    timing_stats& mutable_timing() noexcept;
    std::size_t& mutable_processed_waypoint_count() noexcept;

   private:
    struct config config_;
    timing_stats timing_;
    std::size_t processed_waypoint_count_ = 0;
};

///
/// Reusable trajectory planning workflow coordinator.
///
/// Captures the essential control flow for trajectory generation -- waypoint
/// preprocessing, segmentation, dual-algorithm execution, and result selection
/// -- while allowing each consumer (URArm, trajex service, Yaskawa) to inject
/// its own behaviors at well-defined customization points.
///
/// The template parameter `Receiver` is a user-defined type that accumulates
/// per-segment results during trajectory generation. Each enabled algorithm
/// gets its own receiver instance. Per-segment success callbacks fold results
/// into the receiver. The decider receives both receivers and picks the winner.
///
/// Single-use: construct, configure via builder methods, call execute(), discard.
///
/// @tparam Receiver Default-constructible type for accumulating per-segment results
///
template <typename Receiver>
class planner : public planner_base {
   public:
    using config = planner_base::config;

    ///
    /// Result of running a single algorithm across all segments.
    ///
    /// The receiver is engaged when all segments complete successfully. If any
    /// segment fails, the receiver is disengaged and the exception is captured.
    ///
    struct algorithm_outcome {
        std::optional<Receiver> receiver;
        std::exception_ptr error;
    };

    /// @name Handler type aliases
    /// @{
    using waypoint_provider_fn = std::function<waypoint_accumulator(planner&)>;

    using waypoint_preprocessor_fn = std::function<void(planner&, waypoint_accumulator&)>;

    using move_validator_fn = std::function<void(planner&, const waypoint_accumulator&)>;

    using segmenter_fn = std::function<std::vector<waypoint_accumulator>(planner&, waypoint_accumulator)>;

    using success_fn = std::function<void(const planner&, Receiver&, const waypoint_accumulator&, trajectory&&, std::chrono::microseconds)>;

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
    using legacy_success_fn =
        std::function<void(const planner&, Receiver&, const waypoint_accumulator&, Path&&, Trajectory&&, std::chrono::microseconds)>;
#endif

    using algorithm_failure_fn = std::function<void(const planner&, const Receiver&, const waypoint_accumulator&, const std::exception&)>;
    /// @}

    explicit planner(config cfg) : planner_base(std::move(cfg)) {}

    ///
    /// Extends the lifetime of data created inside callbacks.
    ///
    /// Essential because waypoint_accumulator operates on views -- it does not
    /// own its data. When a callback creates a waypoint matrix, that array
    /// must outlive the waypoint_accumulator that views it. Stash stores it
    /// here. The returned shared_ptr can be used immediately; the data it
    /// points to remains valid until the planner is destroyed.
    ///
    template <typename T>
    std::shared_ptr<T> stash(T&& value) {
        auto ptr = std::make_shared<T>(std::forward<T>(value));
        stashed_.push_back(ptr);
        return ptr;
    }

    /// @name Workflow phase registration
    /// @{

    ///
    /// Required. Returns the initial waypoint_accumulator.
    ///
    planner& with_waypoint_provider(waypoint_provider_fn fn) {
        waypoint_provider_ = std::move(fn);
        return *this;
    }

    ///
    /// Optional. Mutates the waypoint accumulator in place (e.g. deduplication).
    ///
    planner& with_waypoint_preprocessor(waypoint_preprocessor_fn fn) {
        preprocessor_ = std::move(fn);
        return *this;
    }

    ///
    /// Optional. Throws if waypoints violate some precondition.
    ///
    planner& with_move_validator(move_validator_fn fn) {
        validator_ = std::move(fn);
        return *this;
    }

    ///
    /// Optional. Splits waypoints into segments. If omitted, the entire
    /// waypoint sequence is treated as a single segment.
    ///
    planner& with_segmenter(segmenter_fn fn) {
        segmenter_ = std::move(fn);
        return *this;
    }

    /// @}

    /// @name Algorithm registration
    /// @{

    ///
    /// Enables the TOTG algorithm.
    ///
    planner& with_totg(success_fn on_success, algorithm_failure_fn on_failure = nullptr) {
        on_success_ = std::move(on_success);
        on_failure_ = std::move(on_failure);
        return *this;
    }

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
    ///
    /// Enables the legacy generator.
    ///
    /// @throws std::logic_error if the config carries a TCP speed limit. The legacy
    ///         generator cannot enforce a TCP limit, so registering it alongside one
    ///         would let a TOTG failure fall back to a trajectory that silently
    ///         ignores a caller-requested safety cap. A consumer that wants a
    ///         deliberate uncapped legacy run must clear config.tcp_max_linear_speed
    ///         first.
    ///
    planner& with_legacy(legacy_success_fn on_success, algorithm_failure_fn on_failure = nullptr) {
        if (get_config().tcp_max_linear_speed) {
            throw std::logic_error(
                "planner: cannot register the legacy generator with a TCP speed limit set; "
                "the legacy generator cannot enforce a TCP limit");
        }
        legacy_on_success_ = std::move(on_success);
        legacy_on_failure_ = std::move(on_failure);
        return *this;
    }
#endif

    /// @}

    ///
    /// Runs the full planning workflow and invokes the decider.
    ///
    /// @param decider Callable receiving (const planner&,
    ///        algorithm_outcome, algorithm_outcome). Return type is deduced.
    ///
    template <typename Decider>
    auto execute(Decider&& decider) -> std::invoke_result_t<Decider, const planner&, algorithm_outcome, algorithm_outcome> {
        if (!waypoint_provider_) {
            throw std::logic_error("waypoint provider not set");
        }

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
        if (!on_success_ && !legacy_on_success_) {
            throw std::logic_error("no algorithms registered");
        }
#else
        if (!on_success_) {
            throw std::logic_error("no algorithms registered");
        }
#endif

        auto& timing = mutable_timing();

        auto t = std::chrono::steady_clock::now();
        auto accumulator = waypoint_provider_(*this);
        timing.waypoint_provisioning = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t);

        if (preprocessor_) {
            t = std::chrono::steady_clock::now();
            preprocessor_(*this, accumulator);
            timing.waypoint_preprocessing = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t);
        }

        mutable_processed_waypoint_count() = accumulator.size();
        if (processed_waypoint_count() < 2) {
            return std::forward<Decider>(decider)(*this, algorithm_outcome{}, algorithm_outcome{});
        }

        if (validator_) {
            t = std::chrono::steady_clock::now();
            validator_(*this, accumulator);
            timing.move_validation = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t);
        }

        // When segment_totg is false, copy the accumulator before the segmenter
        // consumes it. The copy is used as a single unsegmented input for TOTG.
        std::optional<std::vector<waypoint_accumulator>> unsegmented_for_totg;
        if (!get_config().segment_totg && segmenter_) {
            unsegmented_for_totg.emplace();
            unsegmented_for_totg->push_back(accumulator);
        }

        std::vector<waypoint_accumulator> segments;
        if (segmenter_) {
            t = std::chrono::steady_clock::now();
            segments = segmenter_(*this, std::move(accumulator));
            timing.segmentation = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t);
        } else {
            segments.push_back(std::move(accumulator));
        }

        const auto& totg_segments = unsegmented_for_totg ? *unsegmented_for_totg : segments;

        auto totg_outcome = run_totg_(totg_segments);
        auto legacy_outcome = run_legacy_(segments);

        return std::forward<Decider>(decider)(*this, std::move(totg_outcome), std::move(legacy_outcome));
    }

   private:
    algorithm_outcome run_totg_(const std::vector<waypoint_accumulator>& segments) {
        if (!on_success_) {
            return {};
        }

        algorithm_outcome outcome;
        outcome.receiver.emplace();
        std::chrono::microseconds total_gen{};

        for (const auto& segment : segments) {
            if (!outcome.receiver) {
                break;
            }

            try {
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

                auto p = path::create(segment, path_opts);

                trajectory::options topts;
                topts.max_velocity = get_config().velocity_limits;
                topts.max_acceleration = get_config().acceleration_limits;
                topts.observer = get_config().observer;
                if (get_config().tcp_max_linear_speed) {
                    topts.tcp = trajectory::tcp_limits::from(get_config().kinematics_chain.value(), *get_config().tcp_max_linear_speed);
                }

                auto start = std::chrono::steady_clock::now();
                auto traj = trajectory::create(std::move(p), std::move(topts));
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
                total_gen += elapsed;

                on_success_(*this, *outcome.receiver, segment, std::move(traj), elapsed);

            } catch (const std::exception& e) {
                outcome.error = std::current_exception();
                if (on_failure_) {
                    on_failure_(*this, *outcome.receiver, segment, e);
                }
                outcome.receiver.reset();
            }
        }

        mutable_timing().totg_generation_total = total_gen;
        return outcome;
    }

    algorithm_outcome run_legacy_(const std::vector<waypoint_accumulator>& segments [[maybe_unused]]) {
#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
        if (!legacy_on_success_) {
            return {};
        }

        algorithm_outcome outcome;
        outcome.receiver.emplace();
        std::chrono::microseconds total_gen{};
        std::chrono::microseconds total_colinear{};

        for (const auto& segment : segments) {
            if (!outcome.receiver) {
                break;
            }

            try {
                auto start = std::chrono::steady_clock::now();

                // Zero-copy Eigen::Map transform of waypoint views, then copy
                // into the list the legacy Path constructor requires.
                constexpr auto to_eigen = [](const auto& view) {
                    const auto* const data = view.data() + view.data_offset();
                    const auto size = static_cast<Eigen::Index>(view.shape(0));
                    return Eigen::Map<const Eigen::VectorXd>(data, size);
                };
                auto transformed = segment | std::views::transform(to_eigen);
                std::list<Eigen::VectorXd> eigen_waypoints(std::begin(transformed), std::end(transformed));

                if (get_config().colinearization_ratio) {
                    auto colinear_start = std::chrono::steady_clock::now();
                    legacy::apply_colinearization(eigen_waypoints, get_config().path_blend_tolerance * *get_config().colinearization_ratio);
                    total_colinear +=
                        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - colinear_start);
                }

                Path legacy_path(eigen_waypoints, get_config().path_blend_tolerance);

                auto vel_eigen = Eigen::Map<const Eigen::VectorXd>(get_config().velocity_limits->data(),
                                                                   static_cast<Eigen::Index>(get_config().velocity_limits->size()));
                auto acc_eigen = Eigen::Map<const Eigen::VectorXd>(get_config().acceleration_limits->data(),
                                                                   static_cast<Eigen::Index>(get_config().acceleration_limits->size()));

                Trajectory traj(legacy_path, vel_eigen, acc_eigen);
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
                total_gen += elapsed;

                if (!traj.isValid()) {
                    throw std::runtime_error("legacy trajectory generator produced invalid result");
                }

                legacy_on_success_(*this, *outcome.receiver, segment, std::move(legacy_path), std::move(traj), elapsed);

            } catch (const std::exception& e) {
                outcome.error = std::current_exception();
                if (legacy_on_failure_) {
                    legacy_on_failure_(*this, *outcome.receiver, segment, e);
                }
                outcome.receiver.reset();
            }
        }

        auto& timing = mutable_timing();
        timing.legacy_generation_total = total_gen;
        if (total_colinear.count() > 0) {
            timing.colinearization = total_colinear;
        }
        return outcome;
#else
        return {};
#endif
    }

    std::vector<std::shared_ptr<void>> stashed_;

    waypoint_provider_fn waypoint_provider_;
    waypoint_preprocessor_fn preprocessor_;
    move_validator_fn validator_;
    segmenter_fn segmenter_;

    success_fn on_success_;
    algorithm_failure_fn on_failure_;

#if defined(VIAM_TRAJEX_LEGACY_ENABLED)
    legacy_success_fn legacy_on_success_;
    algorithm_failure_fn legacy_on_failure_;
#endif
};

}  // namespace viam::trajex::totg
