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

#ifndef KERNELS_VOLK_VOLK_8U_X2_EXOR_8U_U_H_
#define KERNELS_VOLK_VOLK_8U_X2_EXOR_8U_U_H_

#ifdef LV_HAVE_GENERIC
static inline void
volk_8u_x2_exor_8u_generic(unsigned char* outputVector, const unsigned char* inputVector0, const unsigned char* inputVector1,
    const unsigned int num_bytes){

  int i;
  for(i = 0; i < num_bytes; ++i){
    *outputVector++ = *inputVector0++ ^ *inputVector1++;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX
#include "immintrin.h"

static inline void
volk_8u_x2_exor_8u_u_avx(unsigned char* outputVector, const unsigned char* inputVector0, const unsigned char* inputVector1,
    const unsigned int num_bytes){

  const unsigned int num_32_bytes = num_bytes / 32;
  int i;

  __m256 dst, src0, src1;

  for(i = 0; i < num_32_bytes; i += 32){
    src0 = _mm256_loadu_ps((float*) inputVector0);
    src1 = _mm256_loadu_ps((float*) inputVector1);
    dst = _mm256_xor_ps(src0, src1);
    _mm256_storeu_ps((float*) outputVector, dst);

    inputVector0 += 32;
    inputVector1 += 32;
    outputVector += 32;
  }

  for(; i < num_bytes; ++i){
    *outputVector++ = *inputVector0++ ^ *inputVector1++;
  }
}
#endif /* LV_HAVE_AVX */

#endif /* KERNELS_VOLK_VOLK_8U_X2_EXOR_8U_U_H_ */


#ifndef KERNELS_VOLK_VOLK_8U_X2_EXOR_8U_A_H_
#define KERNELS_VOLK_VOLK_8U_X2_EXOR_8U_A_H_

#ifdef LV_HAVE_AVX
#include "immintrin.h"

static inline void
volk_8u_x2_exor_8u_a_avx(unsigned char* outputVector, const unsigned char* inputVector0, const unsigned char* inputVector1,
    const unsigned int num_bytes){

  const unsigned int num_32_bytes = num_bytes / 32;
  int i;

  __m256 dst, src0, src1;

  for(i = 0; i < num_32_bytes; i += 32){
    src0 = _mm256_load_ps((float*) inputVector0);
    src1 = _mm256_load_ps((float*) inputVector1);
    dst = _mm256_xor_ps(src0, src1);
    _mm256_store_ps((float*) outputVector, dst);

    inputVector0 += 32;
    inputVector1 += 32;
    outputVector += 32;
  }

  for(; i < num_bytes; ++i){
    *outputVector++ = *inputVector0++ ^ *inputVector1++;
  }
}
#endif /* LV_HAVE_AVX */

#endif /* KERNELS_VOLK_VOLK_8U_X2_EXOR_8U_A_H_ */
