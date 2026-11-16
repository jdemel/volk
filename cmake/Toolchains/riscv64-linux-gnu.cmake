#
# Copyright 2026 Johannes Demel
#
# This file is part of VOLK.
# SPDX-License-Identifier: LGPL-3.0-or-later
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc-14)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++-14)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})

foreach(language C CXX ASM)
    if(NOT CMAKE_${language}_FLAGS MATCHES "(^| )-march=")
        string(APPEND CMAKE_${language}_FLAGS " -march=rv64gcv")
    endif()
endforeach()

execute_process(
    COMMAND ${CMAKE_C_COMPILER} -print-file-name=libc.so
    OUTPUT_VARIABLE RISCV64_LIBC
    OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(RISCV64_LIBC "${RISCV64_LIBC}" REALPATH)
get_filename_component(RISCV64_SYSROOT "${RISCV64_LIBC}" DIRECTORY)
get_filename_component(RISCV64_SYSROOT "${RISCV64_SYSROOT}" DIRECTORY)
get_filename_component(RISCV64_SYSROOT_PARENT "${RISCV64_SYSROOT}" DIRECTORY)

set(CMAKE_FIND_ROOT_PATH "${RISCV64_SYSROOT}" "${RISCV64_SYSROOT_PARENT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(ENV{PKG_CONFIG_LIBDIR}
    "${RISCV64_SYSROOT}/lib/pkgconfig:${RISCV64_SYSROOT}/usr/lib/pkgconfig:${RISCV64_SYSROOT_PARENT}/lib/riscv64-linux-gnu/pkgconfig")

set(RISCV64_QEMU_VLEN 128 CACHE STRING "RISC-V vector length for QEMU")
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-riscv64 -L "${RISCV64_SYSROOT}"
    -cpu "max,v=true,vlen=${RISCV64_QEMU_VLEN}")
