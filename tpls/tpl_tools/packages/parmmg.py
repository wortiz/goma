from tpl_tools.packages import packages
from tpl_tools import utils


class Package(packages.CMakePackage):
    def __init__(self):
        self.name = "parmmg"
        self.version = "1.5.0"
        self.sha256 = "0baec7914e49a26bdbb849ab64dcd92147eff79ac02ef3b2599cb05104901a7a"
        self.filename = "parmmg-" + self.version + ".tar.gz"
        self.url = (
            "https://github.com/mmgtools/parmmg/archive/refs/tags/v"
            + self.version
            + ".tar.gz"
        )
        self.libraries = ["parmmg"]
        self.dependencies = ["cmake", "openmpi", "metis", "scotch", "lapack", "mmg"]

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
        builder.add_option("-DMMG_DIR=" + builder.env["MMG_DIR"])

    def register(self, builder):
        registry = builder._registry
        registry.register_package(self.name, builder.install_dir())
        registry.set_environment_variable("PARMMG_DIR", builder.install_dir())
        registry.prepend_environment_variable(
            "CMAKE_PREFIX_PATH", builder.install_dir()
        )
