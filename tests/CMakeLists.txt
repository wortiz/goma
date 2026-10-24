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

# Unit tests of functions in the goma library
set(GOMA_SOLVER_TEST_SOURCES
    rf_util/import_elem_size.cpp
)

add_executable(goma_solver_unit_tests unit_tests_main.cpp ${GOMA_SOLVER_TEST_SOURCES})
target_compile_definitions(goma_solver_unit_tests PRIVATE GOMA_UNIT_TESTS_LINK_GOMA)
target_link_libraries(goma_solver_unit_tests Catch2::Catch2 goma goma_user goma_util
                      ${MPI_C_LIBRARIES})
target_include_directories(goma_solver_unit_tests PRIVATE ${MPI_C_INCLUDE_PATH})

include(CTest)
include(Catch)
catch_discover_tests(goma_unit_tests)
catch_discover_tests(goma_solver_unit_tests)

# Exercise both build-time diagnostic paths without linking the solver.
foreach(backend_mode legacy kokkos)
  add_executable(goma_assembly_${backend_mode}_tests assembly/backend.cpp
                 ${PROJECT_SOURCE_DIR}/src/mm_assembly.c)
  target_include_directories(goma_assembly_${backend_mode}_tests PRIVATE
                             ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(goma_assembly_${backend_mode}_tests PRIVATE Catch2::Catch2WithMain)
  if(backend_mode STREQUAL "kokkos")
    target_compile_definitions(goma_assembly_${backend_mode}_tests PRIVATE
                               GOMA_ENABLE_KOKKOS_ASSEMBLY)
  endif()
  catch_discover_tests(goma_assembly_${backend_mode}_tests TEST_PREFIX "${backend_mode}:")
endforeach()


# Runs goma several times in one process through goma.h
add_executable(goma_lib_sequence library/goma_lib_sequence.c)
target_link_libraries(goma_lib_sequence goma_library)

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
  # each run must use its own petscrc, not the options of the run before
  add_test(NAME goma_lib_petsc COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_petsc --mode chain
                                       -p p_C_T_U_petsc_gamg -p p_C_T_U_petsc)
  # Goma on a communicator holding only some of the ranks
  add_test(NAME goma_lib_subcomm COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_subcomm --mode chain
                                         --extra-ranks 1 -p ldc -p p_lub_pore -p p_C_T_U_petsc)
  # runs on shifting sets of ranks, ranks 0-1 then 1-2: PETSc must be usable on ranks that
  # did not take part in the run that first used it
  add_test(NAME goma_lib_shift_ranks
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_shift_ranks --mode chain --no-shuffle
                   --extra-ranks 1 --shift-ranks -p p_C_T_U_petsc -p p_C_T_U_petsc_gamg)
  # solvers that give up, steady (ldc: one Newton iteration), transient (tbc) and segregated
  # (poise_drop_seg: time step below its minimum): goma_solve and goma_advance_steps must
  # return an error
  set(GOMA_LIB_FAILURE_ARGS --mode chain --no-shuffle --expect-failure
                            --set-card "Number of Newton Iterations=1"
                            --set-card "Minimum time step=1.0e6" -p ldc -p tbc -p poise_drop_seg)
  add_test(NAME goma_lib_failure COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_failure
                                         ${GOMA_LIB_FAILURE_ARGS})
  add_test(NAME goma_lib_failure_advance
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_failure_advance --advance-steps 1
                   ${GOMA_LIB_FAILURE_ARGS})
  # solved in steps by goma_advance_steps, the output must match goma_solve's and the
  # fields Goma returns the output's; steady (2_phase_diff) and transient problems, and a
  # continuation problem that cannot be advanced
  add_test(NAME goma_lib_advance
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_advance --mode chain --advance-steps 2
                   --check-fields -p tbc -p current -p 2_phase_diff -p heat_box_fcont)
  # solved by goma_advance_to every 1.0 of simulation time (two of poise_drop's fixed time
  # steps), the output must match goma_solve's
  add_test(NAME goma_lib_advance_to
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_advance_to --runs 1 --advance-to 1.0
                   -p poise_drop)
  # targets between time steps (every 0.75, steps of 0.5): steps are cut to land on each
  # target, two runs in one process must give the same results
  add_test(NAME goma_lib_advance_to_cut
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_advance_to_cut --runs 2
                   --advance-to 0.75 --compare-runs -p poise_drop)
  # advanced, then new values for an imported field and goma_solve for the rest: the output
  # must have the new values
  add_test(NAME goma_lib_solve_after
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_solve_after --runs 2
                   --advance-steps 2 --solve-after 2 --declare-value SOURCE:2.5:Q2 --import-value-step 1
                   --check-imports -p tbc)
  # the segregated solver in steps, twice in one process
  add_test(NAME goma_lib_advance_segregated
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_advance_segregated --runs 2
                   --advance-steps 3 -p poise_drop_seg)
  # external fields set by the host instead of read from a file, serial and two ranks
  add_test(NAME goma_lib_import
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_import --runs 1 --import-fields
                   --advance-steps 1 -p lub_film -p lub_CYmold)
  # the same with the field declared by goma_declare_external_field instead of a card
  add_test(NAME goma_lib_declare
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_declare --runs 2 --import-fields
                   --declare-fields --advance-steps 1 -p lub_film)
  # an imported field set once keeps its values through mesh adaptation (poise_drop adapts)
  add_test(NAME goma_lib_adapt_import
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_adapt_import --runs 2
                   --advance-steps 3 --declare-value SOURCE:2.5 -p poise_drop)
  # new values for an imported field before each call, after adaptation sized for the
  # adapted mesh
  add_test(NAME goma_lib_adapt_import_reset
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_adapt_import_reset --runs 1
                   --advance-steps 3 --declare-value SOURCE:2.5 --import-value-step 1 -p poise_drop)
  # a coupling loop: before each time step the host sets the imported field from Goma's
  # solution, two runs in one process must give the same results
  add_test(NAME goma_lib_coupling
           COMMAND ${GOMA_LIB_REGRESSION} --output-dir goma_lib_coupling --runs 2 --import-fields
                   --advance-steps 1 --feedback THETA:SH_FH:1000 --compare-runs -p lub_film)
  # a host program outside Goma's tree, built against Goma installed with cmake --install
  add_test(NAME goma_lib_install
           COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${PROJECT_BINARY_DIR}
                   -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
                   -DPROBLEM_DIR=${GOMA_GUTS_DIR}/problems/2_phase_diff
                   "-DGOMA_ARGS=-i simple_diff.gap -a" "-DGENERATOR=${CMAKE_GENERATOR}"
                   -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/goma_lib_install
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/library/install_check.cmake)
  set_tests_properties(goma_lib_install PROPERTIES LABELS library)
  set_tests_properties(goma_lib_repeat goma_lib_chain goma_lib_mpi goma_lib_petsc goma_lib_subcomm
                       goma_lib_shift_ranks goma_lib_failure goma_lib_failure_advance
                       goma_lib_advance goma_lib_advance_to goma_lib_advance_to_cut
                       goma_lib_solve_after goma_lib_advance_segregated goma_lib_import
                       goma_lib_declare goma_lib_adapt_import goma_lib_adapt_import_reset
                       goma_lib_coupling
                       PROPERTIES SKIP_RETURN_CODE 77 LABELS library)
  set_tests_properties(goma_lib_import PROPERTIES PROCESSORS 2)
  set_tests_properties(goma_lib_mpi goma_lib_petsc PROPERTIES PROCESSORS 2)
  set_tests_properties(goma_lib_subcomm goma_lib_shift_ranks PROPERTIES PROCESSORS 3)
  if(CMAKE_BUILD_TYPE STREQUAL "Asan")
    # the host program would need the sanitizer flags of Goma's libraries
    set_tests_properties(goma_lib_install PROPERTIES DISABLED TRUE)
    set_tests_properties(
      goma_lib_repeat goma_lib_chain goma_lib_mpi goma_lib_petsc goma_lib_subcomm goma_lib_shift_ranks
      goma_lib_failure goma_lib_failure_advance goma_lib_advance goma_lib_advance_to
      goma_lib_advance_to_cut goma_lib_solve_after goma_lib_advance_segregated goma_lib_import
      goma_lib_declare goma_lib_adapt_import goma_lib_adapt_import_reset goma_lib_coupling
      PROPERTIES
        ENVIRONMENT
        "ASAN_OPTIONS=fast_unwind_on_malloc=0;LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/library/lsan.supp"
    )
  endif()
else()
  message(STATUS "GUTS problems not found, library mode tests disabled")
endif()
