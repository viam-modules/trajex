#pragma once

// This puts `Path` and `Trajectory` in the global namespace, so it can't be used in the same
// program as the original. Use the classes in `viam::trajex::totg::compat` directly to run the
// two side by side.

#include "Path.h"

#include <viam/trajex/totg/compat/trajectory.hpp>

using viam::trajex::totg::compat::Trajectory;
