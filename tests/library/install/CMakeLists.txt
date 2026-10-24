# A host program built outside Goma's tree against an installed Goma, see
# tests/library/install_check.cmake
cmake_minimum_required(VERSION 3.23)
project(goma_install_host C)

find_package(Goma REQUIRED)

add_executable(goma_install_host host.c)
target_link_libraries(goma_install_host PRIVATE Goma::goma)
