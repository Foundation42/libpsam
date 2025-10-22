# C examples for libpsam

# Basic usage example
add_executable(basic_usage basic_usage.c)
target_link_libraries(basic_usage psam)

# Layer composition example
add_executable(layer_composition layer_composition.c)
target_link_libraries(layer_composition psam)

# Install examples
install(TARGETS basic_usage layer_composition
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
)
