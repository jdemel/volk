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

/*!
 * \page volk_8u_pack8_8u
 *
 * \b Overview
 *
 * Pack 8 bits into byte. First bit goes to MSB.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8u_pack8_8u(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li out_buf: target buffer for packed bits
 * \li in_buf: source buffer with unpacked bits
 * \li num_points: number of packed bytes!
 *
 * \b Outputs
 * \li out_buf: bytes with bits packed into bytes.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_8u_pack8_8u(out_buf, in_buf, num_points);
 *
 * volk_free(x);
 * \endcode
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_PACK8_8U_U_H
#define VOLK_KERNELS_VOLK_VOLK_8U_PACK8_8U_U_H
#include <volk/volk_common.h>
#include <stdio.h>

static inline void
pack_byte(unsigned char* out_buf, const unsigned char* in_buf)
{
  const unsigned char* inputPtr = in_buf;
  unsigned char* outputPtr = out_buf;
  *outputPtr = 0x00; // set to zero first

  *outputPtr |= ((*inputPtr++) & 0x01) << 7;
  *outputPtr |= ((*inputPtr++) & 0x01) << 6;
  *outputPtr |= ((*inputPtr++) & 0x01) << 5;
  *outputPtr |= ((*inputPtr++) & 0x01) << 4;
  *outputPtr |= ((*inputPtr++) & 0x01) << 3;
  *outputPtr |= ((*inputPtr++) & 0x01) << 2;
  *outputPtr |= ((*inputPtr++) & 0x01) << 1;
  *outputPtr |= ((*inputPtr) & 0x01);
}

static inline void
print_vector(unsigned char* out_buf, const unsigned char* in_buf){
  int i;
  printf("result: %x %x for: ", (int)out_buf[0], (int)out_buf[1]);
  for(i = 0; i < 16; i ++){
    unsigned char b = in_buf[i];
    printf("%x, ", (int)b);
  }
  printf("\n");
}


#ifdef LV_HAVE_GENERIC

static inline void volk_8u_pack8_8u_generic(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_bytes){
  unsigned int byte;
  const unsigned char* inputPtr = in_buf;
  unsigned char* outputPtr = out_buf;

  for(byte = 0; byte < num_bytes; byte += 1){
    pack_byte(outputPtr, inputPtr);
    outputPtr++;
    inputPtr += 8;
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

// shuffle_epi8 only available with SSSE3.
static inline void volk_8u_pack8_8u_u_ssse3(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_bytes){
  const unsigned int fast_num_bytes = num_bytes & 0xFFffFFfe; // make sure we can process 2 bytes at a time.
  unsigned int byte = 0;
  const unsigned char* inputPtr = in_buf;
  unsigned char* outputPtr = out_buf;

  __m128i loaded, compared;
  const __m128i reverse_mask = _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
  const __m128i bit_mask = _mm_set1_epi8(0x01);

  for(; byte < fast_num_bytes; byte += 2){
    // welcome to little endian hell.
    loaded = _mm_loadu_si128((__m128i*) inputPtr);
    loaded = _mm_shuffle_epi8(loaded, reverse_mask);
    loaded = _mm_and_si128(loaded, bit_mask);
    compared = _mm_cmpeq_epi8(loaded, bit_mask);

    *((uint16_t*) outputPtr) = _mm_movemask_epi8(compared);
    outputPtr += 2;
    inputPtr += 16;
  }

  for(; byte < num_bytes; byte += 1){
    pack_byte(outputPtr, inputPtr);
    outputPtr++;
    inputPtr += 8;
  }
}

#endif /* LV_HAVE_SSSE3 */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_PACK8_8U_U_H */



#ifndef VOLK_KERNELS_VOLK_VOLK_8U_PACK8_8U_A_H
#define VOLK_KERNELS_VOLK_VOLK_8U_PACK8_8U_A_H
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_8u_pack8_8u_a_ssse3(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_bytes){
  const unsigned int fast_num_bytes = num_bytes & 0xFFffFFfe; // make sure we can process 2 bytes at a time.
  unsigned int byte = 0;
  const unsigned char* inputPtr = in_buf;
  unsigned char* outputPtr = out_buf;

  __m128i loaded, reversed, masked, compared;
  __m128i reverse_mask, bit_mask;

  reverse_mask = _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
  bit_mask = _mm_set_epi8(0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01);

  for(; byte < fast_num_bytes; byte += 2){
    // welcome to little endian hell.
    loaded = _mm_load_si128((__m128i*) inputPtr);
    reversed = _mm_shuffle_epi8(loaded, reverse_mask);
    masked = _mm_and_si128(reversed, bit_mask);
    compared = _mm_cmpeq_epi8(masked, bit_mask);

    *((uint16_t*) outputPtr) = _mm_movemask_epi8(compared);
    outputPtr += 2;
    inputPtr += 16;
  }

  for(; byte < num_bytes; byte += 1){
    pack_byte(outputPtr, inputPtr);
    outputPtr++;
    inputPtr += 8;
  }
}
#endif /* LV_HAVE_SSSE3 */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_PACK8_8U_A_H */
