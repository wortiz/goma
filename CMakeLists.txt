# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
SET(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})

# Get Trilinos as one entity
FIND_PACKAGE(Trilinos REQUIRED)

set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG}")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")

SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

INCLUDE_DIRECTORIES(SYSTEM ${MPI_INCLUDE_PATH} ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
LINK_DIRECTORIES(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} ${MPI_LIBRARIES})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

FILE(GLOB SOURCE_FILES
  "*.cu"
  "*.c"
  "*.cpp"
  "*.cuh"
  "*.h")

cuda_add_executable(
  ns_cuda
  ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(ns_cuda
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  ${Trilinos_EXTRA_LD_FLAGS}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_cusolver_LIBRARY})
# Specify binary name and source file to build it from
