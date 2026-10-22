cmake_minimum_required (VERSION 3.11)
cmake_policy(VERSION 3.11)

find_package(Catch2 3 REQUIRED)

set(GOMA_TEST_SOURCES
    gds/gds_vector.cpp
    bc/rotate_util.cpp
)

add_executable(goma_unit_tests unit_tests_main.cpp ${GOMA_TEST_SOURCES})
target_link_libraries(goma_unit_tests Catch2::Catch2 goma_util gds ${MPI_C_LIBRARIES})
target_include_directories(goma_unit_tests PRIVATE ${MPI_C_INCLUDE_PATH})

include(CTest)
include(Catch)
catch_discover_tests(goma_unit_tests)

# Runs goma several times in one process through goma.h
add_executable(goma_lib_sequence library/goma_lib_sequence.c)
target_link_libraries(goma_lib_sequence goma goma_user goma_util ${MPI_C_LIBRARIES})
target_include_directories(goma_lib_sequence PRIVATE ${MPI_C_INCLUDE_PATH})

# Library mode regression tests: GUTS problems run several times in one process
# by goma_lib_sequence and compared with their blessed results
find_package(Python3 COMPONENTS Interpreter)
set(GOMA_GUTS_DIR ${PROJECT_SOURCE_DIR}/guts)
if(Python3_FOUND AND EXISTS ${GOMA_GUTS_DIR}/problems)
  set(GOMA_LIB_REGRESSION
      ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/goma_lib_regression.py
      --driver $<TARGET_FILE:goma_lib_sequence> --goma $<TARGET_FILE:goma_exe>
      --guts-dir ${GOMA_GUTS_DIR} -j 1)
  # same problem twice, after teardown of a context stopped after create and setup
  add_test(NAME goma_lib_repeat
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_repeat --partial
                                        -p 2_phase_diff -p ldc)
  # different physics, meshes and solvers one after another
  add_test(
    NAME goma_lib_chain
    COMMAND
      ${GOMA_LIB_REGRESSION} --output-dir goma_lib_chain --mode chain --chain-length 6 --partial
      -p 2_phase_diff -p ldc -p overset_and_levelset_tst -p gd_table -p heat_box_fcont -p tri)
  # two MPI ranks
  add_test(NAME goma_lib_mpi COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_mpi --partial
                                     -p p_lub_pore)
  set_tests_properties(goma_lib_repeat goma_lib_chain goma_lib_mpi
                       PROPERTIES SKIP_RETURN_CODE 77 LABELS library)
  set_tests_properties(goma_lib_mpi PROPERTIES PROCESSORS 2)
else()
  message(STATUS "GUTS problems not found, library mode tests disabled")
endif()
