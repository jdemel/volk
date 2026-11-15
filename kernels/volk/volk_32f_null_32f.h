/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <inttypes.h>

#ifndef INCLUDED_volk_32f_null_32f_a_H
#define INCLUDED_volk_32f_null_32f_a_H

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_null_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; number++) {
        *bVector++ = *aVector++;
    }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_null_32f_u_H */
