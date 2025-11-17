# Core PSAM library

# Collect source files
set(PSAM_SOURCES
    src/core/model.c
    src/core/csr.c
    src/core/train.c
    src/core/infer.c
    src/core/generator.c
    src/core/inspect.c
    src/composition/composite_layered.c
    src/composition/vocab_alignment.c
    src/io/serialize.c
    src/io/composite.c
    src/io/composite_v1.c
    src/io/sha256.c
)

# Create library
add_library(psam ${PSAM_SOURCES})
add_library(psam::psam ALIAS psam)

if(WIN32 AND BUILD_SHARED_LIBS)
    set_target_properties(psam PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(BUILD_SHARED_LIBS)
    target_compile_definitions(psam
        PRIVATE PSAM_BUILDING_DLL
        PUBLIC PSAM_SHARED
    )
endif()

set_target_properties(psam PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set_target_properties(psam PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_CURRENT_BINARY_DIR}/${cfg}
        LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_CURRENT_BINARY_DIR}/${cfg}
        RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_CURRENT_BINARY_DIR}/${cfg}
    )
endforeach()

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
)

# Link math library on Unix-like systems (not needed on Windows)
if(NOT WIN32)
    target_link_libraries(psam PRIVATE m)
endif()

# Library properties
set_target_properties(psam PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
