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
 * \page volk_8u_unpack8_8u
 *
 * \b Overview
 *
 * unpack 8 bits in a byte to 1 bit in a byte.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8u_unpack8_8u(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_bytes)
 * \endcode
 *
 * \b Inputs
 * \li out_buf: target buffer for unpacked bits
 * \li in_buf: source buffer with packed bits
 * \li num_points: number of packed bytes
 *
 * \b Outputs
 * \li out_but: bytes with bit set in LSB.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_8u_unpack8_8u(out_buf, in_buf, num_bytes);
 *
 * volk_free(x);
 * \endcode
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_UNPACK8_8U_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_UNPACK8_8U_H_
#include <stdio.h>

static inline void
unpack_byte(unsigned char* out_buf, const unsigned char* in_buf)
{
  const unsigned char* inputPtr = in_buf;
  unsigned char* outputPtr = out_buf;
  *outputPtr++ = (*inputPtr & 0x80) >> 7;
  *outputPtr++ = (*inputPtr & 0x40) >> 6;
  *outputPtr++ = (*inputPtr & 0x20) >> 5;
  *outputPtr++ = (*inputPtr & 0x10) >> 4;
  *outputPtr++ = (*inputPtr & 0x08) >> 3;
  *outputPtr++ = (*inputPtr & 0x04) >> 2;
  *outputPtr++ = (*inputPtr & 0x02) >> 1;
  *outputPtr = (*inputPtr & 0x01);

}

static inline void
print_vector_unpack(unsigned char* out_buf, const unsigned char* in_buf){
  int i;
  printf("source: %x %x result: ", (int)in_buf[0], (int)in_buf[1]);
  for(i = 0; i < 16; i ++){
    unsigned char b = out_buf[i];
    printf("%x, ", (int)b);
  }
  printf("\n");
}

#ifdef LV_HAVE_GENERIC

static inline void volk_8u_unpack8_8u_generic(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_bytes){
  unsigned int byte;
  const unsigned char* inputPtr = in_buf;
  unsigned char* outputPtr = out_buf;

  for(byte = 0; byte < num_bytes; ++byte){
    unpack_byte(outputPtr, inputPtr);
    outputPtr += 8;
    inputPtr++;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_8u_unpack8_8u_u_ssse3(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_bytes){
  const unsigned int fast_num_bytes = num_bytes & 0xFFffFFfe;
  unsigned int byte;
  const unsigned char* inputPtr = in_buf;
  unsigned char* outputPtr = out_buf;

  __m128i loaded;
  const __m128i reverse_mask = _mm_set_epi8(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14);
  const __m128i bit_mask = _mm_set_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
  const __m128i ones = _mm_set1_epi8(1);

  for(byte = 0; byte < fast_num_bytes; byte += 2){
    loaded = _mm_set1_epi16(*((short*) inputPtr));
    loaded = _mm_shuffle_epi8(loaded, reverse_mask);
    loaded = _mm_and_si128(loaded, bit_mask);
    loaded = _mm_cmpeq_epi8(loaded, bit_mask);
    loaded = _mm_and_si128(loaded, ones);
    _mm_storeu_si128((__m128i*) outputPtr, loaded);
    inputPtr += 2;
    outputPtr += 16;
  }

  for(; byte < num_bytes; ++byte){
    unpack_byte(outputPtr, inputPtr);
    outputPtr += 8;
    inputPtr++;
  }
}

#endif /* LV_HAVE_SSSE3 */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_UNPACK8_8U_H_ */
