# set CMAKE_BUILD_TYPE default value
if(NOT CMAKE_CONFIGURATION_TYPES)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE
            "Release"
            CACHE STRING "Build configuration" FORCE)
    endif()
endif()

# validate CMAKE_BUILD_TYPE against default CMake build types
set(VALID_BUILD_TYPES "Release" "Debug" "RelWithDebInfo" "MinSizeRel")
if(NOT CMAKE_CONFIGURATION_TYPES)
    list(FIND VALID_BUILD_TYPES "${CMAKE_BUILD_TYPE}" INDEX)
    if(${INDEX} MATCHES -1)
        message(
            FATAL_ERROR
                "Invalid build type. Valid types are [${VALID_BUILD_TYPES}]")
    endif()
endif()

if(NOT CMAKE_CONFIGURATION_TYPES)
    if(DEFINED CMAKE_BUILD_TYPE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                                                     ${VALID_BUILD_TYPES})
    endif()
endif()
