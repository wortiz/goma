# Installs Goma, builds tests/library/install against the install with
# find_package(Goma) and solves a problem with it
#
# cmake -DBUILD_DIR=<goma build> -DSOURCE_DIR=<goma source> -DPROBLEM_DIR=<problem>
#       "-DGOMA_ARGS=<goma args>" -DWORK_DIR=<dir> [-DGENERATOR=<generator>] -P install_check.cmake

function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "failed (${result}): ${ARGN}")
  endif()
endfunction()

file(REMOVE_RECURSE ${WORK_DIR})
set(prefix ${WORK_DIR}/prefix)
run(${CMAKE_COMMAND} --install ${BUILD_DIR} --prefix ${prefix})

# only what was installed, not Goma's build or source tree
set(host_build ${WORK_DIR}/host)
set(generator_args)
if(GENERATOR)
  set(generator_args -G ${GENERATOR})
endif()
run(${CMAKE_COMMAND} -S ${SOURCE_DIR}/tests/library/install -B ${host_build} ${generator_args}
    -DCMAKE_PREFIX_PATH=${prefix})
run(${CMAKE_COMMAND} --build ${host_build})

file(COPY ${PROBLEM_DIR}/ DESTINATION ${WORK_DIR}/problem)
separate_arguments(GOMA_ARGS)
execute_process(COMMAND ${host_build}/goma_install_host ${GOMA_ARGS}
                WORKING_DIRECTORY ${WORK_DIR}/problem RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "goma_install_host failed (${result})")
endif()
