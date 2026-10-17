// Constructs the MoveIt time parameterization with trajex's defaults. Timing a plan needs a robot model, which the
// ament package's tests provide; this only shows the installed library and its MoveIt dependencies resolve.

#include <cstdlib>
#include <iostream>

#include <viam/trajex/moveit/totg/time_parameterization.hpp>

int main() {
    const viam::trajex::moveit::totg::time_parameterization parameterization;
    std::cout << "moveit: time parameterization constructed\n";
    return EXIT_SUCCESS;
}
