// Force the cgo build to link through the C++ toolchain driver so the platform
// C++ runtime (libstdc++ on Linux, libc++ on macOS) is pulled in to satisfy the
// prebuilt trajex archives, which are C++. cgo selects the C++ linker whenever a
// package contains any C++ source; this file exists purely for that effect. Its
// presence is what matters, not its contents.
