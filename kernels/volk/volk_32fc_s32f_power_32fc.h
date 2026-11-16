/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_power_32fc
 *
 * \b Overview
 *
 * Raises each complex input value to the real-valued power and stores the
 * principal value in the complex output vector. For nonzero input, the phase
 * is determined by atan2(imaginary, real), including its signed-zero branch
 * cut along the negative real axis. Zero raised to a positive power is zero;
 * any input raised to zero is one.
 * Inputs and the power must be finite, and the squared magnitude must be
 * representable as a float without overflow or underflow.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_power_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const
 * float power, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: The complex input vector.
 * \li power: The power value to be applied to each data point.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li cVector: The complex floating-point output vector.
 *
 * \b Example
 * \code
 * lv_32fc_t input[2] = { lv_cmake(3.0f, 4.0f), lv_cmake(1.0f, 1.0f) };
 * lv_32fc_t output[2];
 * volk_32fc_s32f_power_32fc(output, input, 2.0f, 2);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_power_32fc_a_H
#define INCLUDED_volk_32fc_s32f_power_32fc_a_H

#include <math.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32f_power_32fc_generic(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const float power,
                                                     unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        const float real = lv_creal(aVector[number]);
        const float imag = lv_cimag(aVector[number]);
        const float magnitude_squared = real * real + imag * imag;
        const float magnitude = powf(magnitude_squared, power * 0.5f);
        const float angle = power * atan2f(imag, real);
        cVector[number] = lv_cmake(magnitude * cosf(angle), magnitude * sinf(angle));
    }
}

#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_s32f_power_32fc_a_H */
