# Third party libraries

These are third party libraries with separate license files.

Please look at comments or license files for the separate license

# Script to install all tpls

Third party libraries required for Goma can be installed using the `install-tpls.py` script
available in this directory.

## Warning on licenses

By default the `install-tpls.py` script will install with `ParMETIS` but the license is quite
restrictive. If you are not a government or education employee consider using `--disable-parmetis`
or read the license restrictions. 

Currently our main interface to SuperLU_DIST is through Trilinos and Trilinos requires SuperLU_DIST 
be built with ParMETIS support so if ParMETIS is disabled so is SuperLU_DIST.

## Examples:

Build with a nonstandard compiler:

    ./install-tpls.py --cc=gcc-13 --cxx=g++-13 --fc=gfortran-13 /path/to/install

Build with a prebuilt openmpi with 4 CPUs:

    ./install-tpls.py -j 4 --cc=mpicc --cxx=mpicxx --fc==mpifort --openmpi-dir=$MPI_HOME /path/to/install

Build with static libraries, shared is set by default:

    ./install-tpls.py --build-static /path/to/install

Build Trilinos with CUDA (Kokkos) support, `nvcc` (CUDA >= 12.2) must be on `PATH`.
Currently only Trilinos is built with CUDA, set `--cuda-arch` to the Kokkos architecture
of the target GPU (e.g. `AMPERE80`, `AMPERE86`, `ADA89`, `HOPPER90`), otherwise Kokkos
autodetects the GPU on the build machine:

    ./install-tpls.py --enable-cuda --cuda-arch=AMPERE86 /path/to/install

A container image can be built with podman using `Dockerfile-cuda`:

    podman build -f Dockerfile-cuda --build-arg KOKKOS_CUDA_ARCH=AMPERE86 -t goma-libs-cuda .
    podman run --rm -it --device nvidia.com/gpu=all goma-libs-cuda

Use ccache to speed up rebuilds (`ccache` must be on `PATH`), `--ccache-dir` sets `CCACHE_DIR`
and implies `--enable-ccache`. ccache is only used while building, it is not written to `config.sh`:

    ./install-tpls.py --enable-ccache /path/to/install
    ./install-tpls.py --ccache-dir=/path/to/ccache /path/to/install

The Dockerfiles use ccache with a build cache mount (`id=goma-tpls-ccache`) so rebuilds of the
images reuse previous compilations, the size can be set with `--build-arg CCACHE_MAXSIZE=20G`.

The number of parallel build jobs in the Dockerfiles can be set with `--build-arg JOBS=16` (default 2).

Reuse downloads:

    ./install-tpls.py --download-dir=/path/to/downloads /path/to/install

Add LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on MacOS) in config files, this should hopefully not be needed with CMake RPATH handling:

    ./install-tpls.py --write-dynamic-library-path /path/to/install

Default locations:

Downloads are downloaded to `/path/to/install/downloads`, and sources/builds are found in `/path/to/install/sources`
feel free to remove after installation.

## Usage

Once installed a `config.sh` and `config.fish` file are generated in `/path/to/install` and can be sourced
in your shell of choice and used to compile and run Goma.

For current command line options do `./install-tpls.py -h`


Usage:

    usage: install-tpls.py [-h] [--cc CC] [--cxx CXX] [--fc FC] [--download-dir DOWNLOAD_DIR] [--extract-dir EXTRACT_DIR] [--build-shared]
                           [--build-static] [-j JOBS] [--netlib-blas] [--openblas] [--enable-parmetis] [--disable-parmetis] [--skip-ssl-verify]
                           [--write-dynamic-library-path] [--cmake-dir CMAKE_DIR] [--openmpi-dir OPENMPI_DIR] [--hdf5-dir HDF5_DIR]
                           [--pnetcdf-dir PNETCDF_DIR] [--netcdf-dir NETCDF_DIR] [--omega_h-dir OMEGA_H_DIR] [--fmt-dir FMT_DIR]
                           [--seacas-dir SEACAS_DIR] [--bison-dir BISON_DIR] [--flex-dir FLEX_DIR] [--openblas-dir OPENBLAS_DIR]
                           [--lapack-dir LAPACK_DIR] [--metis-dir METIS_DIR] [--parmetis-dir PARMETIS_DIR] [--scotch-dir SCOTCH_DIR]
                           [--arpack_ng-dir ARPACK_NG_DIR] [--scalapack-dir SCALAPACK_DIR] [--mumps-dir MUMPS_DIR]
                           [--superlu_dist-dir SUPERLU_DIST_DIR] [--suitesparse-dir SUITESPARSE_DIR] [--trilinos-dir TRILINOS_DIR]
                           [--petsc-dir PETSC_DIR] [--petsc_complex-dir PETSC_COMPLEX_DIR] [--sparse-dir SPARSE_DIR] [--catch2-dir CATCH2_DIR]
                           INSTALL_DIR
    
    Third party library installer for the finite element code Goma
    
    positional arguments:
      INSTALL_DIR           Install location of TPLs
    
    options:
      -h, --help            show this help message and exit
      --cc CC               C compiler to use, default is CC environment variable or gcc
      --cxx CXX             C++ compiler to use, default is CXX environment variable or g++
      --fc FC               Fortran compiler to use, defualt is FC environment variable or gfortran
      --download-dir DOWNLOAD_DIR
                            Download location of tarballs
      --extract-dir EXTRACT_DIR
                            Extract and Build location
      --build-shared        Build shared libraries (Default)
      --build-static        Build static libraries
      -j JOBS, --jobs JOBS  Number of parallel jobs
      --netlib-blas         Build using reference BLAS/LAPACK
      --openblas            Build using OpenBLAS (Default)
      --enable-parmetis     Build ParMETIS library, (Default, check license requirements)
      --disable-parmetis    Disable ParMETIS library
      --skip-ssl-verify     Disable SSL checks on download
      --write-dynamic-library-path
                            Writes (DY)LD_LIBRARY_PATH to config, default is off
      --cmake-dir CMAKE_DIR
                            System location of package cmake
      --openmpi-dir OPENMPI_DIR
                            System location of package openmpi
      --hdf5-dir HDF5_DIR   System location of package hdf5
      --pnetcdf-dir PNETCDF_DIR
                            System location of package pnetcdf
      --netcdf-dir NETCDF_DIR
                            System location of package netcdf
      --omega_h-dir OMEGA_H_DIR
                            System location of package omega-h
      --fmt-dir FMT_DIR     System location of package fmt
      --seacas-dir SEACAS_DIR
                            System location of package seacas
      --bison-dir BISON_DIR
                            System location of package bison
      --flex-dir FLEX_DIR   System location of package flex
      --openblas-dir OPENBLAS_DIR
                            System location of package openblas
      --lapack-dir LAPACK_DIR
                            System location of package lapack
      --metis-dir METIS_DIR
                            System location of package metis
      --parmetis-dir PARMETIS_DIR
                            System location of package parmetis
      --scotch-dir SCOTCH_DIR
                            System location of package scotch
      --arpack_ng-dir ARPACK_NG_DIR
                            System location of package arpack-ng
      --scalapack-dir SCALAPACK_DIR
                            System location of package scalapack
      --mumps-dir MUMPS_DIR
                            System location of package mumps
      --superlu_dist-dir SUPERLU_DIST_DIR
                            System location of package superlu_dist
      --suitesparse-dir SUITESPARSE_DIR
                            System location of package suitesparse
      --trilinos-dir TRILINOS_DIR
                            System location of package trilinos
      --petsc-dir PETSC_DIR
                            System location of package petsc
      --petsc_complex-dir PETSC_COMPLEX_DIR
                            System location of package petsc-complex
      --sparse-dir SPARSE_DIR
                            System location of package sparse
      --catch2-dir CATCH2_DIR
                            System location of package catch2
        