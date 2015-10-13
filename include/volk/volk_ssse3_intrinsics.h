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

/*
 * This file is intended to hold SSSE3 intrinsics of intrinsics.
 * They should be used in VOLK kernels to avoid copy-pasta.
 */

#ifndef INCLUDE_VOLK_VOLK_SSSE3_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_SSSE3_INTRINSICS_H_
#include <tmmintrin.h>

static inline __m128i
_mm_encodepolarstage_si128(__m128i _inbuf)
{
  // This is a single stage polar encoder.
  __m128i shifted;
  const __m128i mask_stage1 = _mm_set_epi8(0x0, 0xFF, 0x0, 0xFF, 0x0, 0xFF, 0x0, 0xFF, 0x0, 0xFF, 0x0, 0xFF, 0x0, 0xFF, 0x0, 0xFF);
  shifted = _mm_srli_si128(_inbuf, 1);
  shifted = _mm_and_si128(shifted, mask_stage1);
  return _mm_xor_si128(shifted, _inbuf);
}

static inline __m128i
_mm_16bitencodepolar_si128(__m128i _inbuf)
{
  // This encodes a 16bit polar code. Result is a 16bit codeword.
  // It is a basic encoder for all higher order polar codes.
  __m128i temp, shifted;
  const __m128i shuffle_stage4 = _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
  const __m128i mask_stage4 = _mm_set_epi8(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
  const __m128i mask_stage3 = _mm_set_epi8(0x0, 0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF, 0xFF, 0x0, 0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF, 0xFF);
  const __m128i mask_stage2 = _mm_set_epi8(0x0, 0x0, 0xFF, 0xFF, 0x0, 0x0, 0xFF, 0xFF, 0x0, 0x0, 0xFF, 0xFF, 0x0, 0x0, 0xFF, 0xFF);
  // shuffle once for bit-reversal.
  // this one time bit reversal substitutes a shuffle operation after every stage.
  temp = _mm_shuffle_epi8(_inbuf, shuffle_stage4);

  shifted = _mm_srli_si128(temp, 8);
  shifted = _mm_and_si128(shifted, mask_stage4);
  temp = _mm_xor_si128(shifted, temp);

  shifted = _mm_srli_si128(temp, 4);
  shifted = _mm_and_si128(shifted, mask_stage3);
  temp = _mm_xor_si128(shifted, temp);

  shifted = _mm_srli_si128(temp, 2);
  shifted = _mm_and_si128(shifted, mask_stage2);
  temp = _mm_xor_si128(shifted, temp);

  return _mm_encodepolarstage_si128(temp);
}

#endif /* INCLUDE_VOLK_VOLK_SSSE3_INTRINSICS_H_ */
