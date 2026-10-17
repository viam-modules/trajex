#pragma once

// This puts `Path` in the global namespace, so it can't be used in the same program as the
// original. Use `viam::trajex::totg::compat::Path` directly to run the two side by side.

#include <viam/trajex/totg/compat/path.hpp>

using viam::trajex::totg::compat::Path;
