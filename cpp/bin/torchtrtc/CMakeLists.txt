set(executable_name "torchtrtc")

add_executable(${executable_name}
    ${CMAKE_CURRENT_SOURCE_DIR}/accuracy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fileio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parser_util.cpp
)

if (MSVC)
    target_link_libraries(${executable_name} PRIVATE torch torchtrt)
else()
    target_link_libraries(${executable_name} PRIVATE torch "-Wl,--no-as-needed -ldl" torchtrt "-Wl,--as-needed")
    set_target_properties(
        ${executable_name}
        PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE #
                   LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

target_include_directories(${executable_name}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

install(TARGETS ${executable_name} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
