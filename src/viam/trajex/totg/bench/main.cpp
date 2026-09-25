// Entry point for the trajex benchmarks.
//
// Kept apart from the benchmark sources so that no one of them owns the program: the
// pipeline benchmarks and the xtensor idiom benchmarks are peers, and either could be run
// without the other.

#include <benchmark/benchmark.h>

#include <viam/trajex/totg/bench/benchmarks.hpp>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

int main(int argc, char** argv) {
#if defined(__APPLE__)
    // macOS offers no hard CPU affinity, but a user-interactive QoS class keeps this thread
    // on the performance cores. Without it, results wander as the scheduler moves work
    // between P and E cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    viam::trajex::totg::bench::register_pipeline_benchmarks();

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
