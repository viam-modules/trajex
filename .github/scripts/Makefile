.PHONY: \
	module.tar.gz \
	format-check \
	test \
	run-clang-tidy \
	run-clang-tidy-parallel \
	run-clang-check \

help:
	@echo "This Makefile is for CI orchestration only."
	@echo "Please use CMake for local development."
	@echo ""
	@echo "Available CI targets:"
	@echo "  make -f .github/scripts/Makefile [format-check,configure,build,install,test,module.tar.gz,run-clang-tidy[-parallel],run-clang-check]"

SOURCE_FILES := $(shell find src/viam -type f \( -name '*.cpp' -o -name '*.hpp' \))
CPP_FILES := $(filter %.cpp,$(SOURCE_FILES))

format-check:
	clang-format-19 --style=file --dry-run --Werror $(SOURCE_FILES)

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja

build: configure
	cmake --build build --target all

install: build
	DESTDIR=build/install cmake --install build --prefix /

test: build
	CTEST_OUTPUT_ON_FAILURE=1 cmake --build build --target test

module.tar.gz: format-check install
	cmake --build build --target package

run-clang-tidy:
	clang-tidy-19 \
        -p build \
        --config-file ./.clang-tidy \
        --header-filter=".*/viam/(ur/module|trajex)/.*" \
	$(CPP_FILES)

run-clang-tidy-parallel:
	run-clang-tidy-19 \
        -p build \
        -config-file ./.clang-tidy \
        -header-filter=".*/viam/(ur/module|trajex)/.*" \
	$(CPP_FILES)

run-clang-check:
	clang-check-19 -p build $(CPP_FILES)
