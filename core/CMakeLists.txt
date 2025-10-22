# Core PSAM library

# Collect source files
set(PSAM_SOURCES
    src/core/model.c
    src/core/csr.c
    src/core/train.c
    src/core/infer.c
    src/composition/layers.c
    src/io/serialize.c
)

# Create library
add_library(psam ${PSAM_SOURCES})

# Include directories
target_include_directories(psam
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(psam
    PRIVATE
        Threads::Threads
        m  # Math library
)

# Library properties
set_target_properties(psam PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER include/psam.h
)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
