/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
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

/* For documentation see 'kernels/volk/volk_8u_unpack8_8u.h' */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_UNPACK8PUPPET_8U_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_UNPACK8PUPPET_8U_H_
#include<volk/volk_8u_unpack8_8u.h>

#ifdef LV_HAVE_GENERIC
static inline void
volk_8u_unpack8puppet_8u_generic(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_bytes){
  volk_8u_unpack8_8u_generic(out_buf, in_buf, num_bytes / 8);
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSSE3
static inline void
volk_8u_unpack8puppet_8u_u_ssse3(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_bytes){
  volk_8u_unpack8_8u_u_ssse3(out_buf, in_buf, num_bytes / 8);
}
#endif /* LV_HAVE_SSSE3 */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_UNPACK8PUPPET_8U_H_ */
