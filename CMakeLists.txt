cmake_minimum_required(VERSION 3.20)
project(fem_test_resolve LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Put runtime binaries in: <build>/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(ReSolve CONFIG 
  PATHS ${ReSolve_DIR} ${ReSolve_DIR}/share/resolve/cmake
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  REQUIRED)

# Build your executable 
add_executable(sysGmres
  main.cpp
)

target_compile_features(sysGmres PRIVATE cxx_std_17)

target_include_directories(sysGmres PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/helper
)

target_link_libraries(sysGmres PRIVATE ReSolve::ReSolve)