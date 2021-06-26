/* -*- C++ -*- */
/*
 * Copyright 2021 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_VOLK_COMPLEX_HH
#define INCLUDED_VOLK_COMPLEX_HH

/*!
 * \brief Provide typedefs and operators for all complex types in C++.
 *
 * The typedefs encompass all signed integer and floating point types.
 * Each operator function is intended to work across all data types.
 * Under C++, these operators are defined as inline templates.
 * Under C, these operators are defined as preprocessor macros.
 * The use of macros makes the operators agnostic to the type.
 *
 * The following operator functions are defined:
 * - lv_cmake - make a complex type from components
 * - lv_creal - get the real part of the complex number
 * - lv_cimag - get the imaginary part of the complex number
 * - lv_conj - take the conjugate of the complex number
 */


#include <stdint.h>
#include <complex>

typedef std::complex<int8_t> lv_8sc_t;
typedef std::complex<int16_t> lv_16sc_t;
typedef std::complex<int32_t> lv_32sc_t;
typedef std::complex<int64_t> lv_64sc_t;
typedef std::complex<float> lv_32fc_t;
typedef std::complex<double> lv_64fc_t;

template <typename T>
inline std::complex<T> lv_cmake(const T& r, const T& i)
{
    return std::complex<T>(r, i);
}

template <typename T>
inline typename T::value_type lv_creal(const T& x)
{
    return x.real();
}

template <typename T>
inline typename T::value_type lv_cimag(const T& x)
{
    return x.imag();
}

template <typename T>
inline T lv_conj(const T& x)
{
    return std::conj(x);
}

#endif /* INCLUDE_VOLK_COMPLEX_HH */