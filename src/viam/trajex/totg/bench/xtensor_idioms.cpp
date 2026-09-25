// Cost of the xtensor idioms trajex uses on small, fixed-width vectors.
//
// Every hot path in the integrator manipulates one configuration-space vector at a time,
// which for the arms we care about is six doubles. At that width the useful work is a
// handful of arithmetic operations, so whatever a container spends establishing shape,
// strides, or storage is not amortised over anything -- it is the cost. These benchmarks
// put numbers on that, so decisions about which container to hold geometry in, and whether
// to return it or fill it in place, rest on measurement rather than on reasoning about what
// the compiler ought to manage.
//
// Operations are batched over a run of rows because a six-element operation takes a couple
// of nanoseconds, which is the same order as google-benchmark's own loop overhead. Timing
// one operation per iteration would mostly measure the harness.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#endif

namespace {

// Six revolute joints, matching the arms trajex is used with.
constexpr std::size_t k_dof = 6;

// Rows touched per iteration. Large enough that the harness overhead is negligible against
// the work, small enough that everything stays in L1 and we measure the operation rather
// than the memory system.
constexpr std::size_t k_rows = 256;

using fixed_row = xt::xtensor_fixed<double, xt::xshape<k_dof>>;
using raw_row = std::array<double, k_dof>;

// Arbitrary but reproducible values, and not constant across rows, so nothing folds away.
double sample_value(std::size_t row, std::size_t joint) {
    return 1.0 + (static_cast<double>((row * k_dof) + joint) * 0.125);
}

template <typename Array2D>
Array2D make_2d() {
    auto result = Array2D::from_shape({k_rows, k_dof});
    for (std::size_t row = 0; row != k_rows; ++row) {
        for (std::size_t joint = 0; joint != k_dof; ++joint) {
            result(row, joint) = sample_value(row, joint);
        }
    }
    return result;
}

template <typename Row>
std::vector<Row> make_rows() {
    std::vector<Row> result(k_rows);
    for (std::size_t row = 0; row != k_rows; ++row) {
        for (std::size_t joint = 0; joint != k_dof; ++joint) {
            result[row][joint] = sample_value(row, joint);
        }
    }
    return result;
}

//
// Copying one row. This is what `waypoint_store::append` does per waypoint, and the
// comparison that sent us here: the scalar loop measured materially faster than the view
// assignment on `xarray`.
//

template <typename Array2D>
void copy_scalar_loop(benchmark::State& state) {
    const auto source = make_2d<Array2D>();
    auto destination = make_2d<Array2D>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            for (std::size_t joint = 0; joint != k_dof; ++joint) {
                destination(row, joint) = source(row, joint);
            }
        }
        benchmark::DoNotOptimize(destination.data());
        benchmark::ClobberMemory();
    }
}

template <typename Array2D>
void copy_view_assign(benchmark::State& state) {
    const auto source = make_2d<Array2D>();
    auto destination = make_2d<Array2D>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            xt::view(destination, row, xt::all()) = xt::view(source, row, xt::all());
        }
        benchmark::DoNotOptimize(destination.data());
        benchmark::ClobberMemory();
    }
}

template <typename Row>
void copy_whole_row(benchmark::State& state) {
    const auto source = make_rows<Row>();
    auto destination = make_rows<Row>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            destination[row] = source[row];
        }
        benchmark::DoNotOptimize(destination.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(copy_scalar_loop<xt::xarray<double>>)->Name("bm_copy/xarray_scalar_loop");
BENCHMARK(copy_view_assign<xt::xarray<double>>)->Name("bm_copy/xarray_view_assign");
BENCHMARK(copy_scalar_loop<xt::xtensor<double, 2>>)->Name("bm_copy/xtensor2_scalar_loop");
BENCHMARK(copy_view_assign<xt::xtensor<double, 2>>)->Name("bm_copy/xtensor2_view_assign");
BENCHMARK(copy_whole_row<fixed_row>)->Name("bm_copy/xtensor_fixed_assign");
BENCHMARK(copy_whole_row<raw_row>)->Name("bm_copy/std_array_assign");

//
// Handing a vector back to a caller. This is the shape of `path::cursor::tangent()` and its
// siblings, which return `xt::xarray<double>` by value on every geometry query.
//
// The producers are marked noinline deliberately: the real accessors are defined in path.cpp
// and called from trajectory.cpp with no link-time optimisation, so the caller cannot see
// through them and elide the return. A benchmark that let them inline would measure
// something the integrator never gets.
//

[[gnu::noinline]] xt::xarray<double> produce_xarray(const xt::xarray<double>& source, std::size_t row) {
    auto result = xt::xarray<double>::from_shape({k_dof});
    for (std::size_t joint = 0; joint != k_dof; ++joint) {
        result(joint) = source(row, joint);
    }
    return result;
}

[[gnu::noinline]] fixed_row produce_fixed(const xt::xarray<double>& source, std::size_t row) {
    fixed_row result;
    for (std::size_t joint = 0; joint != k_dof; ++joint) {
        result[joint] = source(row, joint);
    }
    return result;
}

[[gnu::noinline]] void fill_xarray(const xt::xarray<double>& source, std::size_t row, xt::xarray<double>& out) {
    for (std::size_t joint = 0; joint != k_dof; ++joint) {
        out(joint) = source(row, joint);
    }
}

[[gnu::noinline]] void fill_span(const xt::xarray<double>& source, std::size_t row, std::span<double> out) {
    for (std::size_t joint = 0; joint != k_dof; ++joint) {
        out[joint] = source(row, joint);
    }
}

void bm_produce_xarray(benchmark::State& state) {
    const auto source = make_2d<xt::xarray<double>>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            auto value = produce_xarray(source, row);
            benchmark::DoNotOptimize(value.data());
        }
    }
}

void bm_produce_fixed(benchmark::State& state) {
    const auto source = make_2d<xt::xarray<double>>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            auto value = produce_fixed(source, row);
            benchmark::DoNotOptimize(value.data());
        }
    }
}

void bm_fill_xarray(benchmark::State& state) {
    const auto source = make_2d<xt::xarray<double>>();
    auto out = xt::xarray<double>::from_shape({k_dof});

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            fill_xarray(source, row, out);
            benchmark::DoNotOptimize(out.data());
        }
    }
}

void bm_fill_span(benchmark::State& state) {
    const auto source = make_2d<xt::xarray<double>>();
    raw_row out{};

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            fill_span(source, row, out);
            benchmark::DoNotOptimize(out.data());
        }
    }
}

BENCHMARK(bm_produce_xarray)->Name("bm_produce/return_xarray");
BENCHMARK(bm_produce_fixed)->Name("bm_produce/return_xtensor_fixed");
BENCHMARK(bm_fill_xarray)->Name("bm_produce/fill_xarray_out_param");
BENCHMARK(bm_fill_span)->Name("bm_produce/fill_span_out_param");

//
// The joint velocity limit, which is the innermost arithmetic in the integrator: the
// smallest ratio of a joint's velocity limit to the magnitude of its path derivative.
//
// The epsilon guard the real `compute_joint_velocity_limit` carries is omitted, because an
// expression form cannot branch per element. A near-zero derivative yields a huge ratio
// that the minimum discards anyway, so the arithmetic compared here is representative even
// though the semantics are not identical.
//

template <typename Vector>
void limit_scalar_loop(benchmark::State& state) {
    const auto derivatives = make_rows<Vector>();
    const auto limits = make_rows<Vector>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        double smallest = std::numeric_limits<double>::infinity();
        for (std::size_t row = 0; row != k_rows; ++row) {
            for (std::size_t joint = 0; joint != k_dof; ++joint) {
                smallest = std::min(smallest, limits[row][joint] / std::abs(derivatives[row][joint]));
            }
        }
        benchmark::DoNotOptimize(smallest);
    }
}

void limit_expression(benchmark::State& state) {
    const auto derivatives = make_rows<xt::xarray<double>>();
    const auto limits = make_rows<xt::xarray<double>>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        double smallest = std::numeric_limits<double>::infinity();
        for (std::size_t row = 0; row != k_rows; ++row) {
            smallest = std::min(smallest, xt::amin(limits[row] / xt::abs(derivatives[row]))());
        }
        benchmark::DoNotOptimize(smallest);
    }
}

BENCHMARK(limit_scalar_loop<xt::xarray<double>>)->Name("bm_limit/xarray_scalar_loop");
BENCHMARK(limit_scalar_loop<fixed_row>)->Name("bm_limit/xtensor_fixed_scalar_loop");
BENCHMARK(limit_scalar_loop<raw_row>)->Name("bm_limit/std_array_scalar_loop");
BENCHMARK(limit_expression)->Name("bm_limit/xarray_expression");

//
// The waypoint coalescing test from `path::create`, which a profile put at roughly three
// quarters of that stage. For each triple of consecutive rows it asks whether the middle one
// lies close enough to the line between its neighbours to be dropped: a squared length, a
// dot product, a projection, and a norm, all over six doubles.
//
// The point of the grid is to separate two effects that the existing `bm_limit` numbers
// conflate. Rank varies across `xarray` (runtime rank, shape carried in an svector) and
// `xtensor<double, 2>` (rank fixed at compile time, size still dynamic). Reduction strategy
// varies across xtensor's default lazy stepper and its immediate path. The scalar loop is
// the floor: the same arithmetic with no container machinery at all.
//
// The lazy cells reproduce the real code faithfully, including binding `start_to_next` once
// and using it five times, because that repetition is part of what is being measured.
//

constexpr double k_coalesce_radius = 1.0;

template <typename Array2D>
void coalesce_lazy(benchmark::State& state) {
    const auto rows = make_2d<Array2D>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        std::size_t coalesced = 0;

        for (std::size_t i = 0; i + 2 < k_rows; ++i) {
            const auto start = xt::view(rows, i, xt::all());
            const auto locus = xt::view(rows, i + 1, xt::all());
            const auto next = xt::view(rows, i + 2, xt::all());

            const auto start_to_next = next - start;
            if (xt::all(xt::equal(start_to_next, 0.0))) {
                continue;
            }

            const auto start_to_locus = locus - start;
            const double start_to_next_sq = xt::sum(start_to_next * start_to_next)();
            const double start_to_locus_dot_direction = xt::sum(start_to_locus * start_to_next)();

            if (start_to_locus_dot_direction < 0.0 || start_to_locus_dot_direction > start_to_next_sq) {
                continue;
            }

            const double t = start_to_locus_dot_direction / start_to_next_sq;
            const auto projected_point = start + (t * start_to_next);
            const auto deviation_vector = locus - projected_point;

            if (xt::norm_l2(deviation_vector)() <= k_coalesce_radius) {
                ++coalesced;
            }
        }

        benchmark::DoNotOptimize(coalesced);
    }
}

// The row container matching a given 2-D container's rank discipline. This is what
// `xt::eval` would hand back for an expression over rows of that container, spelled out so
// the benchmark states which type it is testing rather than deriving it from a trait.
template <typename Array2D>
struct row_of;

template <>
struct row_of<xt::xarray<double>> {
    using type = xt::xarray<double>;
};

template <>
struct row_of<xt::xtensor<double, 2>> {
    using type = xt::xtensor<double, 1>;
};

template <typename Array2D>
using row_of_t = typename row_of<Array2D>::type;

// Identical to coalesce_lazy except that the one expression used more than once is evaluated
// into a container first. In the real code `next - start` is walked five times -- once for the
// identical-endpoints test, twice inside the squared length, once for the dot product and once
// for the projection -- and every walk goes through the view steppers again.
//
// This trades four of those walks for one allocation per triple. Note when reading the result
// that the allocation sits at full weight here, where in `path::create` the same allocation is
// diluted across everything else that stage does per waypoint.
template <typename Array2D>
void coalesce_materialized(benchmark::State& state) {
    const auto rows = make_2d<Array2D>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        std::size_t coalesced = 0;

        for (std::size_t i = 0; i + 2 < k_rows; ++i) {
            const auto start = xt::view(rows, i, xt::all());
            const auto locus = xt::view(rows, i + 1, xt::all());
            const auto next = xt::view(rows, i + 2, xt::all());

            const row_of_t<Array2D> start_to_next = next - start;
            if (xt::all(xt::equal(start_to_next, 0.0))) {
                continue;
            }

            const auto start_to_locus = locus - start;
            const double start_to_next_sq = xt::sum(start_to_next * start_to_next)();
            const double start_to_locus_dot_direction = xt::sum(start_to_locus * start_to_next)();

            if (start_to_locus_dot_direction < 0.0 || start_to_locus_dot_direction > start_to_next_sq) {
                continue;
            }

            const double t = start_to_locus_dot_direction / start_to_next_sq;
            const auto projected_point = start + (t * start_to_next);
            const auto deviation_vector = locus - projected_point;

            if (xt::norm_l2(deviation_vector)() <= k_coalesce_radius) {
                ++coalesced;
            }
        }

        benchmark::DoNotOptimize(coalesced);
    }
}

template <typename Array2D>
void coalesce_immediate(benchmark::State& state) {
    const auto rows = make_2d<Array2D>();
    constexpr auto immediate = xt::evaluation_strategy::immediate;

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        std::size_t coalesced = 0;

        for (std::size_t i = 0; i + 2 < k_rows; ++i) {
            const auto start = xt::view(rows, i, xt::all());
            const auto locus = xt::view(rows, i + 1, xt::all());
            const auto next = xt::view(rows, i + 2, xt::all());

            const auto start_to_next = next - start;
            if (xt::all(xt::equal(start_to_next, 0.0))) {
                continue;
            }

            const auto start_to_locus = locus - start;
            const double start_to_next_sq = xt::sum(start_to_next * start_to_next, immediate)();
            const double start_to_locus_dot_direction = xt::sum(start_to_locus * start_to_next, immediate)();

            if (start_to_locus_dot_direction < 0.0 || start_to_locus_dot_direction > start_to_next_sq) {
                continue;
            }

            const double t = start_to_locus_dot_direction / start_to_next_sq;
            const auto projected_point = start + (t * start_to_next);
            const auto deviation_vector = locus - projected_point;

            if (xt::norm_l2(deviation_vector, immediate)() <= k_coalesce_radius) {
                ++coalesced;
            }
        }

        benchmark::DoNotOptimize(coalesced);
    }
}

void coalesce_scalar_loop(benchmark::State& state) {
    const auto rows = make_2d<xt::xtensor<double, 2>>();
    const double* const data = rows.data();

    // Degrees of freedom is a runtime property everywhere in trajex, and the expression cells
    // above carry it as runtime data inside the container whatever happens. Taking it from the
    // shape and hiding it from the optimiser keeps this loop honest: against a constexpr bound
    // it unrolls completely and the comparison measures something no production call site gets.
    std::size_t dof = rows.shape(1);
    benchmark::DoNotOptimize(dof);

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        std::size_t coalesced = 0;

        for (std::size_t i = 0; i + 2 < k_rows; ++i) {
            const double* const start = data + (i * dof);
            const double* const locus = data + ((i + 1) * dof);
            const double* const next = data + ((i + 2) * dof);

            // One pass covers the identical-endpoints test, the squared length and the dot
            // product; the expression forms above need three traversals for the same three
            // answers because each is a separate reduction.
            bool identical = true;
            double start_to_next_sq = 0.0;
            double start_to_locus_dot_direction = 0.0;

            for (std::size_t joint = 0; joint != dof; ++joint) {
                const double direction = next[joint] - start[joint];
                identical = identical && (direction == 0.0);
                start_to_next_sq += direction * direction;
                start_to_locus_dot_direction += (locus[joint] - start[joint]) * direction;
            }

            if (identical) {
                continue;
            }
            if (start_to_locus_dot_direction < 0.0 || start_to_locus_dot_direction > start_to_next_sq) {
                continue;
            }

            const double t = start_to_locus_dot_direction / start_to_next_sq;
            double deviation_sq = 0.0;

            for (std::size_t joint = 0; joint != dof; ++joint) {
                const double deviation = locus[joint] - (start[joint] + (t * (next[joint] - start[joint])));
                deviation_sq += deviation * deviation;
            }

            if (std::sqrt(deviation_sq) <= k_coalesce_radius) {
                ++coalesced;
            }
        }

        benchmark::DoNotOptimize(coalesced);
    }
}

BENCHMARK(coalesce_lazy<xt::xarray<double>>)->Name("bm_coalesce/xarray_lazy");
BENCHMARK(coalesce_materialized<xt::xarray<double>>)->Name("bm_coalesce/xarray_materialized");
BENCHMARK(coalesce_immediate<xt::xarray<double>>)->Name("bm_coalesce/xarray_immediate");
BENCHMARK(coalesce_lazy<xt::xtensor<double, 2>>)->Name("bm_coalesce/xtensor2_lazy");
BENCHMARK(coalesce_materialized<xt::xtensor<double, 2>>)->Name("bm_coalesce/xtensor2_materialized");
BENCHMARK(coalesce_immediate<xt::xtensor<double, 2>>)->Name("bm_coalesce/xtensor2_immediate");
BENCHMARK(coalesce_scalar_loop)->Name("bm_coalesce/scalar_loop");

}  // namespace
