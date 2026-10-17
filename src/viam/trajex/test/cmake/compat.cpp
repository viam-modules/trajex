// Times the same unit straight move as totg.cpp, through the reference implementation's interface, included by the
// original header names that the compatibility layer's shim serves.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <list>

#include <Path.h>
#include <Trajectory.h>

int main() {
    const std::list<Eigen::VectorXd> waypoints{Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{1.0, 0.0}};
    const Trajectory trajectory(Path(waypoints, 0.1), Eigen::Vector2d{2.0, 2.0}, Eigen::Vector2d{1.0, 1.0});
    if (!trajectory.isValid()) {
        std::cout << "compat: trajectory is not valid\n";
        return EXIT_FAILURE;
    }

    const double duration = trajectory.getDuration();
    std::cout << "compat: duration " << duration << " s\n";
    return std::abs(duration - 2.0) < 1e-2 ? EXIT_SUCCESS : EXIT_FAILURE;
}
