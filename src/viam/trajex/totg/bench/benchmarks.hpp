#pragma once

namespace viam::trajex::totg::bench {

// Registers the pipeline benchmarks, which are named for the replay record they run against
// and so cannot be declared with the BENCHMARK macro.
//
// Called from main rather than from a static initialiser because it reads the replay records
// to size each record's sweep, and a failure to do so should surface as a reported error
// rather than as a terminate before main is entered.
//
// Throws if a replay record cannot be read or parsed.
void register_pipeline_benchmarks();

}  // namespace viam::trajex::totg::bench
