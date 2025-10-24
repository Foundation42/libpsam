cmake_minimum_required(VERSION 3.15)
project(libpsam VERSION 0.1.0 LANGUAGES C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_WASM "Build WebAssembly module" OFF)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -fPIC)
endif()

# Thread support
find_package(Threads REQUIRED)

# Core library
add_subdirectory(core)

# CLI
add_executable(psam_cli core/src/cli/psam_cli.c)
set_target_properties(psam_cli PROPERTIES OUTPUT_NAME psam)
target_include_directories(psam_cli PRIVATE core/include)
target_link_libraries(psam_cli PRIVATE psam Threads::Threads)
if(NOT WIN32)
    target_link_libraries(psam_cli PRIVATE m)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples/c)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS psam
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS psam_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES core/include/psam.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libpsamConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libpsamConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libpsam
)
