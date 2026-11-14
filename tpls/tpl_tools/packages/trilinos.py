from tpl_tools.packages import packages
from tpl_tools import utils
import os


class Package(packages.CMakePackage):
    def __init__(self):
        self.name = "trilinos"
        self.version = "17.3.0"
        self.sha256 = "8094740da60bf75b675ad556ed00e337c8547be0dc9603893c4894f4abf0396b"
        self.filename = "trilinos-" + self.version + ".tar.gz"
        self.url = (
            "https://github.com/trilinos/Trilinos/archive/refs/tags/trilinos-release-"
            + self.version.replace(".", "-")
            + ".tar.gz"
        )
        self.libraries = ["amesos2", "belos", "tpetra", "kokkoscore"]
        self.includes = [
            "Amesos2.hpp",
            "BelosSolverFactory.hpp",
            "Sacado.hpp",
            "Tpetra_CrsMatrix.hpp",
            "kokkos/Kokkos_Core.hpp",
            "NOX.H",
            "Piro_SolverFactory.hpp",
            "Tempus_Integrator.hpp",
            "Intrepid2_Basis.hpp",
            "Shards_CellTopology.hpp",
        ]
        self.dependencies = [
            "cmake",
            "openmpi",
            "lapack",
            "suitesparse",
            "superlu_dist",
            "mumps",
            "gtest",
        ]

    def setDependencies(self, builder):
        return

    def set_environment(self, builder):
        builder.env = builder._registry.get_environment().copy()
        builder.env["CC"] = builder._registry.get_executable("mpicc")
        builder.env["CXX"] = builder._registry.get_executable("mpicxx")
        builder.env["FC"] = builder._registry.get_executable("mpifort")
        if builder.enable_cuda:
            # Compile C++ through Kokkos' nvcc_wrapper, keeping mpicxx as the
            # CMake compiler so MPI flags are still handled by the wrapper.
            # nvcc_wrapper forwards host compilation to the original C++ compiler.
            host_cxx = builder._registry.get_environment().get("CXX", "g++")
            nvcc_wrapper = os.path.join(
                builder._extract_dir,
                builder._extracted_folder,
                "packages",
                "kokkos",
                "bin",
                "nvcc_wrapper",
            )
            builder.env["NVCC_WRAPPER_DEFAULT_COMPILER"] = host_cxx
            builder.env["OMPI_CXX"] = nvcc_wrapper
            builder.env["MPICH_CXX"] = nvcc_wrapper

    def configure_options(self, builder):
        if builder.build_shared:
            builder.add_option("-DBUILD_SHARED_LIBS:BOOL=ON")
            builder.add_option("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
        else:
            builder.add_option("-DBUILD_SHARED_LIBS:BOOL=OFF")
        builder.add_option("-DCMAKE_C_COMPILER=" + builder.env["CC"])
        builder.add_option("-DCMAKE_CXX_COMPILER=" + builder.env["CXX"])
        builder.add_option("-DCMAKE_Fortran_COMPILER=" + builder.env["FC"])
        builder.add_option("-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE")
        builder.add_option("-DTrilinos_SHOW_DEPRECATED_WARNINGS:BOOL=OFF")
        builder.add_option("-DTrilinos_ENABLE_SECONDARY_TESTED_CODE=ON")
        builder.add_option("-DTrilinos_ENABLE_TESTS:BOOL=OFF")
        builder.add_option("-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF")
        builder.add_option("-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON")
        builder.add_option("-DTPL_ENABLE_Boost:BOOL=OFF")

        # Legacy Epetra stack is removed, Tpetra/Kokkos only
        builder.add_option("-DTrilinos_ENABLE_Epetra:BOOL=OFF")
        builder.add_option("-DTrilinos_ENABLE_SEACAS:BOOL=OFF")

        builder.add_option("-DTrilinos_ENABLE_Kokkos:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_KokkosKernels:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Teuchos:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Tpetra:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Xpetra:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Ifpack2:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_MueLu:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Belos:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Amesos2:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Sacado:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Thyra:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Stratimikos:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Teko:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_NOX:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Piro:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Tempus:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Intrepid2:BOOL=ON")
        builder.add_option("-DTrilinos_ENABLE_Shards:BOOL=ON")
        builder.add_option("-DTpetra_INST_INT_INT:BOOL=ON")

        builder.add_option("-DTPL_ENABLE_MPI:BOOL=ON")
        builder.add_option("-DMPI_BASE_DIR:PATH=" + builder.env["MPI_HOME"])

        if builder.enable_cuda:
            builder.add_option("-DTPL_ENABLE_CUDA:BOOL=ON")
            builder.add_option("-DKokkos_ENABLE_CUDA:BOOL=ON")
            builder.add_option("-DKokkos_ENABLE_SERIAL:BOOL=ON")
            if builder.cuda_arch:
                builder.add_option("-DKokkos_ARCH_" + builder.cuda_arch + ":BOOL=ON")
            # The TPL OpenMPI is not built CUDA-aware
            builder.add_option("-DTpetra_ASSUME_GPU_AWARE_MPI:BOOL=OFF")
            builder.add_option("-DTPL_ENABLE_CUBLAS:BOOL=ON")
            builder.add_option("-DTPL_ENABLE_CUSPARSE:BOOL=ON")
            builder.add_option("-DTPL_ENABLE_CUSOLVER:BOOL=ON")

        # Required by ShyLU_NodeTacho (enabled through Amesos2)
        builder.add_option("-DTPL_ENABLE_gtest:BOOL=ON")
        builder.add_option("-DGTest_ROOT:PATH=" + builder.env["GTEST_DIR"])

        builder.add_option("-DTPL_ENABLE_LAPACK:BOOL=ON")
        builder.add_option("-DTPL_ENABLE_BLAS:BOOL=ON")
        if "OPENBLAS_DIR" in builder.env:
            builder.add_option("-DLAPACK_LIBRARY_DIRS=" + builder.env["OPENBLAS_DIR"])
            builder.add_option("-DLAPACK_LIBRARY_NAMES=openblas")
            builder.add_option("-DBLAS_LIBRARY_DIRS=" + builder.env["OPENBLAS_DIR"])
            builder.add_option("-DBLAS_LIBRARY_NAMES=openblas")
        else:
            builder.add_option(
                "-DLAPACK_LIBRARY_DIRS=" + builder.env["LAPACK_DIR"] + "/lib"
            )
            builder.add_option("-DLAPACK_LIBRARY_NAMES=lapack;blas")
            builder.add_option(
                "-DBLAS_LIBRARY_DIRS=" + builder.env["LAPACK_DIR"] + "/lib"
            )
            builder.add_option("-DBLAS_LIBRARY_NAMES=blas")

        builder.add_option("-DTPL_ENABLE_UMFPACK:BOOL=ON")
        builder.add_option(
            "-DUMFPACK_LIBRARY_NAMES:STRING=umfpack;amd;suitesparseconfig;cholmod;colamd;ccolamd;camd"
        )
        builder.add_option(
            "-DUMFPACK_LIBRARY_DIRS:PATH=" + builder.env["SUITESPARSE_DIR"] + "/lib"
        )
        builder.add_option(
            "-DUMFPACK_INCLUDE_DIRS:PATH="
            + builder.env["SUITESPARSE_DIR"]
            + "/include/suitesparse"
        )
        builder.add_option("-DTPL_ENABLE_AMD:BOOL=ON")
        builder.add_option("-DAMD_LIBRARY_NAMES:STRING=amd;suitesparseconfig")
        builder.add_option(
            "-DAMD_LIBRARY_DIRS:PATH=" + builder.env["SUITESPARSE_DIR"] + "/lib"
        )
        builder.add_option(
            "-DAMD_INCLUDE_DIRS:PATH="
            + builder.env["SUITESPARSE_DIR"]
            + "/include/suitesparse"
        )

        if "SUPERLU_DIST_DIR" in builder.env:
            builder.add_option("-DTPL_ENABLE_SuperLUDist:BOOL=ON")
            builder.add_option("-DSuperLUDist_LIBRARY_NAMES:STRING=superlu_dist")
            builder.add_option(
                "-DSuperLUDist_LIBRARY_DIRS:PATH="
                + builder.env["SUPERLU_DIST_DIR"]
                + "/lib"
            )
            builder.add_option(
                "-DSuperLUDist_INCLUDE_DIRS:PATH="
                + builder.env["SUPERLU_DIST_DIR"]
                + "/include"
            )

        ext = utils.get_library_extension(builder.build_shared)
        if "PARMETIS_DIR" in builder.env:
            builder.add_option("-DTPL_ENABLE_ParMETIS:BOOL=ON")
            builder.add_option(
                "-DParMETIS_LIBRARY_DIRS:PATH="
                + builder.env["PARMETIS_DIR"]
                + "/lib;"
                + builder.env["METIS_DIR"]
                + "/lib"
            )
            builder.add_option(
                "-DTPL_ParMETIS_INCLUDE_DIRS:PATH="
                + builder.env["PARMETIS_DIR"]
                + "/include;"
                + builder.env["METIS_DIR"]
                + "/include"
            )
            builder.add_option(
                "-DTPL_ParMETIS_LIBRARIES="
                + builder.env["PARMETIS_DIR"]
                + "/lib/libparmetis"
                + ext
                + ";"
                + builder.env["METIS_DIR"]
                + "/lib/libmetis"
                + ext
            )
        else:
            builder.add_option("-DTPL_ENABLE_ParMETIS:BOOL=OFF")

        builder.add_option("-DTPL_ENABLE_MUMPS:BOOL=ON")
        builder.add_option(
            "-DMUMPS_LIBRARY_NAMES:STRING=dmumps;mumps_common;pord;scalapack;ptesmumps;ptscotch;ptscotcherr;scotch;scotcherr"
        )
        builder.add_option(
            "-DMUMPS_LIBRARY_DIRS:PATH="
            + builder.env["MUMPS_DIR"]
            + "/lib;"
            + builder.env["SCALAPACK_DIR"]
            + "/lib;"
            + builder.env["SCOTCH_DIR"]
            + "/lib"
        )
        builder.add_option(
            "-DMUMPS_INCLUDE_DIRS:PATH=" + builder.env["MUMPS_DIR"] + "/include"
        )
        builder.add_option(
            "-DSCALAPACK_LIBRARY_DIRS:FILEPATH=" + builder.env["SCALAPACK_DIR"] + "/lib"
        )
        builder.add_option("-DSCALAPACK_LIBRARY_NAMES:STRING=scalapack")

    def register(self, builder):
        registry = builder._registry
        registry.register_package(self.name, builder.install_dir())
        registry.set_environment_variable("TRILINOS_DIR", builder.install_dir())
        registry.prepend_environment_variable(
            "CMAKE_PREFIX_PATH", builder.install_dir()
        )
