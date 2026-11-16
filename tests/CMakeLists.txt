#
# Copyright 2022, 2024 Johannes Demel
#
# This file is part of VOLK.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

find_package(fmt)
find_package(GTest)

if(NOT fmt_FOUND OR NOT GTest_FOUND)
    if(fmt_FOUND)
        set(fmt_status "${fmt_VERSION}")
        if(NOT fmt_status)
            set(fmt_status "found, version unknown")
        endif()
    else()
        set(fmt_status "missing")
    endif()
    if(GTest_FOUND)
        set(gtest_status "${GTest_VERSION}")
        if(NOT gtest_status)
            set(gtest_status "found, version unknown")
        endif()
    else()
        set(gtest_status "missing")
    endif()
    message(WARNING "Skipping tests: fmt ${fmt_status}; GoogleTest ${gtest_status}")
    return()
endif(NOT fmt_FOUND OR NOT GTest_FOUND)

file(GLOB volk_test_files CONFIGURE_DEPENDS "test_*.cc")

add_executable(
  volk_tests
  volk_test.cc
  ${volk_test_files}
)

target_compile_features(volk_tests PUBLIC cxx_std_20)

target_link_libraries(volk_tests
  PRIVATE
    GTest::gtest_main
    volk
    fmt::fmt
)

include(GoogleTest)
if(CMAKE_CROSSCOMPILING)
  gtest_discover_tests(volk_tests DISCOVERY_TIMEOUT 120)
else()
  gtest_discover_tests(volk_tests)
endif()


target_include_directories(volk_tests
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/lib>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
