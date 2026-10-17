.PHONY: \
	module.tar.gz \
	format-check \
	test \
	verify-headers \
	verify-package \
	run-clang-tidy \
	run-clang-tidy-parallel \
	run-clang-check \

help:
	@echo "This Makefile is for CI orchestration only."
	@echo "Please use CMake for local development."
	@echo ""
	@echo "Available CI targets:"
	@echo "  make -f .github/scripts/Makefile [format-check,configure,build,verify-headers,verify-package,install,test,module.tar.gz,run-clang-tidy[-parallel],run-clang-check]"

SOURCE_FILES := $(shell find src/viam -type f \( -name '*.cpp' -o -name '*.c' -o -name '*.h' -o -name '*.hpp' \))

# The translation units the configured build compiles, from its compile database. Clang tools given a file
# the database lacks borrow a neighbour's flags and fail on its includes, so we only hand them what is there,
# filtered to our own sources. Sources of disabled targets go unchecked. Expanded in recipes, after configure.
DATABASE_TRANSLATION_UNITS = jq -r --arg root "$(CURDIR)/src/viam/" \
	'[.[].file | select(startswith($$root))] | unique[]' build/compile_commands.json

format-check:
	clang-format-19 --style=file --dry-run --Werror $(SOURCE_FILES)

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja

build: configure
	cmake --build build --target all

# Compiles each public header on its own, which `all` doesn't do.
verify-headers: configure
	cmake --build build --target all_verify_interface_header_sets

# Builds a find_package consumer against installs of this build and of a second build that fetches
# xtl and xtensor.
verify-package: build
	.github/scripts/verify-package.sh build

install: build
	DESTDIR=build/install cmake --install build --prefix /

test: build
	CTEST_OUTPUT_ON_FAILURE=1 cmake --build build --target test

module.tar.gz: format-check install
	cmake --build build --target package

run-clang-tidy: configure
	clang-tidy-19 \
        -p build \
        --config-file ./.clang-tidy \
        --header-filter=".*/viam/(ur/module|trajex)/.*" \
	$$($(DATABASE_TRANSLATION_UNITS))

run-clang-tidy-parallel: configure
	run-clang-tidy-19 \
        -p build \
        -config-file ./.clang-tidy \
        -header-filter=".*/viam/(ur/module|trajex)/.*" \
	$$($(DATABASE_TRANSLATION_UNITS))

run-clang-check: configure
	clang-check-19 -p build $$($(DATABASE_TRANSLATION_UNITS))
