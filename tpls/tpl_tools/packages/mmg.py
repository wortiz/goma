from tpl_tools.packages import packages
from tpl_tools import utils


class Package(packages.CMakePackage):
    def __init__(self):
        self.name = "mmg"
        self.version = "5.8.0"
        self.sha256 = "686eaab84de79c072f3aedf26cd11ced44c84b435d51ce34e016ad203172922f"
        self.filename = "mmg-" + self.version + ".tar.gz"
        self.url = (
            "https://github.com/mmgtools/mmg/archive/refs/tags/v"
            + self.version
            + ".tar.gz"
        )
        self.libraries = ["mmg", "mmg2d", "mmg3d"]
        self.dependencies = ["cmake", "scotch", "lapack"]

    def set_environment(self, builder):
        builder.env = builder._registry.get_environment().copy()
        builder.env["CC"] = builder._registry.get_executable("mpicc")
        builder.env["CXX"] = builder._registry.get_executable("mpicxx")
        builder.env["FC"] = builder._registry.get_executable("mpifort")

    def configure_options(self, builder):
        if builder.build_shared:
            builder.add_option("-DBUILD_SHARED_LIBS:BOOL=ON")
        else:
            builder.add_option("-DBUILD_SHARED_LIBS:BOOL=OFF")
        builder.add_option("-DBLAS_LIBRARIES=" + builder.env["BLAS_LIBRARIES"])
        builder.add_option("-DLAPACK_LIBRARIES=" + builder.env["LAPACK_LIBRARIES"])

    def register(self, builder):
        registry = builder._registry
        registry.register_package(self.name, builder.install_dir())
        registry.set_environment_variable("MMG_DIR", builder.install_dir())
        registry.prepend_environment_variable(
            "CMAKE_PREFIX_PATH", builder.install_dir()
        )
