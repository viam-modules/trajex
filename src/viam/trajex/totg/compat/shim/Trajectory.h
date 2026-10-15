#pragma once

// Claims the global names `Path` and `Trajectory`, so this cannot share a program with the
// original. Use the classes in `viam::trajex::totg::compat` directly to run the two side by side.

#include "Path.h"

#include <viam/trajex/totg/compat/trajectory.hpp>

using viam::trajex::totg::compat::Trajectory;
