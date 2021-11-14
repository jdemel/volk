
# Copyright 2021 Johannes Demel
#
# This file is part of VOLK
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Setup profiler
########################################################################

find_package(fmt REQUIRED)
# find_package(benchmark REQUIRED 1.5)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL
    "Use benchmark without tests"
    FORCE)
add_subdirectory(benchmark)


add_executable (profile volk_profile)

target_link_libraries(profile PRIVATE volk benchmark::benchmark fmt::fmt std::filesystem)

target_include_directories(profile
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/lib>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

install(
    TARGETS profile
    DESTINATION bin
    COMPONENT "volk"
)
