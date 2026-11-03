// Times a unit straight move. At unit acceleration, with a velocity limit that never binds, the move accelerates
// for one second over the first half and decelerates for one second over the second, so it takes two seconds.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/xt.hpp>

int main() {
    using viam::trajex::xmatrix;
    using viam::trajex::xvector;
    using viam::trajex::totg::path;
    using viam::trajex::totg::trajectory;

    const xmatrix<> waypoints = {{0.0, 0.0}, {1.0, 0.0}};
    const trajectory::options options{.max_velocity = xvector<>{2.0, 2.0}, .max_acceleration = xvector<>{1.0, 1.0}};
    const trajectory planned = trajectory::create(path::create(waypoints), options);

    const double duration = planned.duration().count();
    std::cout << "totg: duration " << duration << " s\n";
    return std::abs(duration - 2.0) < 1e-2 ? EXIT_SUCCESS : EXIT_FAILURE;
}
