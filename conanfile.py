from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
from conan.tools.scm import Git

class trajex_service(ConanFile):
    name = "viam-trajex-service"

    license = "Apache-2.0"
    url = "https://github.com/viam-modules/trajex"
    package_type = "application"
    settings = "os", "compiler", "build_type", "arch"

    options = {
        "shared": [True, False]
    }

    default_options = {
        "shared": True
    }

    def export_sources(self):
        for pat in ["CMakeLists.txt", "cmake/*", "src/*", "meta.json*"]:
            copy(self, pat, self.recipe_folder, self.export_sources_folder)

    def set_version(self):
        git = Git(self)
        try:
            self.version = git.run("describe --tags --always").strip().lstrip('v')
        except Exception:
            self.version = "0.0.0-unknown"

    def requirements(self):
        # Prebuilt binaries on viamconan only match the exact dependency versions
        # the SDK release was built with, which conan.lock pins. If you bump the
        # SDK here, regenerate conan.lock starting from the conan.lock attached to
        # that SDK release on GitHub (`conan lock create --lockfile-partial` per
        # platform, then `conan lock merge`), and update `bin/setup.sh` and the
        # viam-cpp-sdk GIT_TAG in CMakeLists.txt to match.
        self.requires("viam-cpp-sdk/0.44.0")
        self.requires("eigen/[>=3.4 <5.0]")
        self.requires("boost/[>=1.74.0]")
        self.requires("jsoncpp/[>=1.9.5]")
        self.requires("benchmark/[>=1.9.5]")
        # Note: grpc comes via viam-cpp-sdk
        # Note: trajex doesn't need URCL (Universal Robots Client Library)

    def validate(self):
        check_min_cppstd(self, 20)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def layout(self):
        cmake_layout(self, src_folder=".")

    def package(self):
        CMake(self).install()

        # Use CPack to build the module.tar.gz and manually copy it to the package folder
        CMake(self).build(target='package')
        copy(self, pattern="module.tar.gz", src=self.build_folder, dst=self.package_folder)

    def deploy(self):
        copy(self, pattern="module.tar.gz", src=self.package_folder, dst=self.deploy_folder)
