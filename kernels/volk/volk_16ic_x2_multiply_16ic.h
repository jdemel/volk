/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_x2_multiply_16ic
 *
 * \b Overview
 *
 * Multiplies two input complex vectors, point-by-point, storing the result in the third
 * vector. WARNING: Saturation is not checked.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_x2_multiply_16ic(lv_16sc_t* result, const lv_16sc_t* in_a, const
 * lv_16sc_t* in_b, unsigned int num_points); \endcode
 *
 * \b Inputs
 * \li in_a: One of the vectors to be multiplied.
 * \li in_b: The other vector to be multiplied.
 * \li num_points: The number of complex data points to be multiplied from both input
 * vectors.
 *
 * \b Outputs
 * \li result: The vector where the results will be stored.
 *
 */

#ifndef INCLUDED_volk_16ic_x2_multiply_16ic_H
#define INCLUDED_volk_16ic_x2_multiply_16ic_H

#include <volk/volk_common.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_x2_multiply_16ic_generic(lv_16sc_t* result,
                                                      const lv_16sc_t* in_a,
                                                      const lv_16sc_t* in_b,
                                                      unsigned int num_points)
{
    for (unsigned int n = 0; n < num_points; ++n) {
        const size_t index = (size_t)n * 2;
        const uint32_t real0 = (uint16_t)((const int16_t*)in_a)[index];
        const uint32_t imag0 = (uint16_t)((const int16_t*)in_a)[index + 1];
        const uint32_t real1 = (uint16_t)((const int16_t*)in_b)[index];
        const uint32_t imag1 = (uint16_t)((const int16_t*)in_b)[index + 1];
        ((int16_t*)result)[index] = (int16_t)(uint16_t)(real0 * real1 - imag0 * imag1);
        ((int16_t*)result)[index + 1] =
            (int16_t)(uint16_t)(real0 * imag1 + imag0 * real1);
    }
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE2
#include <immintrin.h>
static inline void volk_16ic_x2_multiply_16ic_a_sse2(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    const unsigned int blocks = num_points / 4;
    const __m128i real_mask = _mm_set1_epi32(0xffff);
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m128i vec0 = _mm_load_si128((const __m128i*)in_a);
        const __m128i vec1 = _mm_load_si128((const __m128i*)in_b);
        const __m128i products = _mm_mullo_epi16(vec0, vec1);
        const __m128i real = _mm_and_si128(
            _mm_sub_epi16(products, _mm_srli_si128(products, 2)), real_mask);
        const __m128i cross0 = _mm_mullo_epi16(vec0, _mm_slli_si128(vec1, 2));
        const __m128i cross1 = _mm_mullo_epi16(vec1, _mm_slli_si128(vec0, 2));
        const __m128i imag = _mm_andnot_si128(real_mask, _mm_add_epi16(cross0, cross1));
        const __m128i result = _mm_or_si128(real, imag);
        _mm_store_si128((__m128i*)out, result);
        in_a += 4;
        in_b += 4;
        out += 4;
    }
    volk_16ic_x2_multiply_16ic_generic(out, in_a, in_b, num_points % 4);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSE2
static inline void volk_16ic_x2_multiply_16ic_u_sse2(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    const unsigned int blocks = num_points / 4;
    const __m128i real_mask = _mm_set1_epi32(0xffff);
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m128i vec0 = _mm_loadu_si128((const __m128i*)in_a);
        const __m128i vec1 = _mm_loadu_si128((const __m128i*)in_b);
        const __m128i products = _mm_mullo_epi16(vec0, vec1);
        const __m128i real = _mm_and_si128(
            _mm_sub_epi16(products, _mm_srli_si128(products, 2)), real_mask);
        const __m128i cross0 = _mm_mullo_epi16(vec0, _mm_slli_si128(vec1, 2));
        const __m128i cross1 = _mm_mullo_epi16(vec1, _mm_slli_si128(vec0, 2));
        const __m128i imag = _mm_andnot_si128(real_mask, _mm_add_epi16(cross0, cross1));
        const __m128i result = _mm_or_si128(real, imag);
        _mm_storeu_si128((__m128i*)out, result);
        in_a += 4;
        in_b += 4;
        out += 4;
    }
    volk_16ic_x2_multiply_16ic_generic(out, in_a, in_b, num_points % 4);
}
#endif

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
static inline void volk_16ic_x2_multiply_16ic_a_avx2(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    const unsigned int blocks = num_points / 8;
    const __m256i real_mask = _mm256_set1_epi32(0xffff);
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256i vec0 = _mm256_load_si256((const __m256i*)in_a);
        const __m256i vec1 = _mm256_load_si256((const __m256i*)in_b);
        const __m256i products = _mm256_mullo_epi16(vec0, vec1);
        const __m256i real = _mm256_and_si256(
            _mm256_sub_epi16(products, _mm256_srli_si256(products, 2)), real_mask);
        const __m256i cross0 = _mm256_mullo_epi16(vec0, _mm256_slli_si256(vec1, 2));
        const __m256i cross1 = _mm256_mullo_epi16(vec1, _mm256_slli_si256(vec0, 2));
        const __m256i imag =
            _mm256_andnot_si256(real_mask, _mm256_add_epi16(cross0, cross1));
        const __m256i result = _mm256_or_si256(real, imag);
        _mm256_store_si256((__m256i*)out, result);
        in_a += 8;
        in_b += 8;
        out += 8;
    }
    volk_16ic_x2_multiply_16ic_generic(out, in_a, in_b, num_points % 8);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX2
static inline void volk_16ic_x2_multiply_16ic_u_avx2(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    const unsigned int blocks = num_points / 8;
    const __m256i real_mask = _mm256_set1_epi32(0xffff);
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256i vec0 = _mm256_loadu_si256((const __m256i*)in_a);
        const __m256i vec1 = _mm256_loadu_si256((const __m256i*)in_b);
        const __m256i products = _mm256_mullo_epi16(vec0, vec1);
        const __m256i real = _mm256_and_si256(
            _mm256_sub_epi16(products, _mm256_srli_si256(products, 2)), real_mask);
        const __m256i cross0 = _mm256_mullo_epi16(vec0, _mm256_slli_si256(vec1, 2));
        const __m256i cross1 = _mm256_mullo_epi16(vec1, _mm256_slli_si256(vec0, 2));
        const __m256i imag =
            _mm256_andnot_si256(real_mask, _mm256_add_epi16(cross0, cross1));
        const __m256i result = _mm256_or_si256(real, imag);
        _mm256_storeu_si256((__m256i*)out, result);
        in_a += 8;
        in_b += 8;
        out += 8;
    }
    volk_16ic_x2_multiply_16ic_generic(out, in_a, in_b, num_points % 8);
}
#endif

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
static inline void volk_16ic_x2_multiply_16ic_a_avx512f(lv_16sc_t* out,
                                                        const lv_16sc_t* in_a,
                                                        const lv_16sc_t* in_b,
                                                        unsigned int num_points)
{
    const unsigned int blocks = num_points / 16;
    const __m512i mask = _mm512_set1_epi32(0xffff);
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m512i vec0 = _mm512_load_si512((const void*)in_a);
        const __m512i vec1 = _mm512_load_si512((const void*)in_b);
        const __m512i imag0 = _mm512_srli_epi32(vec0, 16);
        const __m512i imag1 = _mm512_srli_epi32(vec1, 16);
        const __m512i real =
            _mm512_and_si512(_mm512_sub_epi32(_mm512_mullo_epi32(vec0, vec1),
                                              _mm512_mullo_epi32(imag0, imag1)),
                             mask);
        const __m512i imag =
            _mm512_slli_epi32(_mm512_add_epi32(_mm512_mullo_epi32(vec0, imag1),
                                               _mm512_mullo_epi32(imag0, vec1)),
                              16);
        const __m512i result = _mm512_or_si512(real, imag);
        _mm512_store_si512((void*)out, result);
        in_a += 16;
        in_b += 16;
        out += 16;
    }
    volk_16ic_x2_multiply_16ic_generic(out, in_a, in_b, num_points % 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512F
static inline void volk_16ic_x2_multiply_16ic_u_avx512f(lv_16sc_t* out,
                                                        const lv_16sc_t* in_a,
                                                        const lv_16sc_t* in_b,
                                                        unsigned int num_points)
{
    const unsigned int blocks = num_points / 16;
    const __m512i mask = _mm512_set1_epi32(0xffff);
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m512i vec0 = _mm512_loadu_si512((const void*)in_a);
        const __m512i vec1 = _mm512_loadu_si512((const void*)in_b);
        const __m512i imag0 = _mm512_srli_epi32(vec0, 16);
        const __m512i imag1 = _mm512_srli_epi32(vec1, 16);
        const __m512i real =
            _mm512_and_si512(_mm512_sub_epi32(_mm512_mullo_epi32(vec0, vec1),
                                              _mm512_mullo_epi32(imag0, imag1)),
                             mask);
        const __m512i imag =
            _mm512_slli_epi32(_mm512_add_epi32(_mm512_mullo_epi32(vec0, imag1),
                                               _mm512_mullo_epi32(imag0, vec1)),
                              16);
        const __m512i result = _mm512_or_si512(real, imag);
        _mm512_storeu_si512((void*)out, result);
        in_a += 16;
        in_b += 16;
        out += 16;
    }
    volk_16ic_x2_multiply_16ic_generic(out, in_a, in_b, num_points % 16);
}
#endif

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_16ic_x2_multiply_16ic_neon(lv_16sc_t* out,
                                                   const lv_16sc_t* in_a,
                                                   const lv_16sc_t* in_b,
                                                   unsigned int num_points)
{
    const unsigned int blocks = num_points / 4;
    for (unsigned int number = 0; number < blocks; ++number) {
        const int16x4x2_t vec0 = vld2_s16((const int16_t*)in_a);
        const int16x4x2_t vec1 = vld2_s16((const int16_t*)in_b);
        __VOLK_PREFETCH(in_a + 4);
        __VOLK_PREFETCH(in_b + 4);
        const int16x4x2_t result = {
            { vmls_s16(vmul_s16(vec0.val[0], vec1.val[0]), vec0.val[1], vec1.val[1]),
              vmla_s16(vmul_s16(vec0.val[0], vec1.val[1]), vec0.val[1], vec1.val[0]) }
        };
        vst2_s16((int16_t*)out, result);
        in_a += 4;
        in_b += 4;
        out += 4;
    }
    volk_16ic_x2_multiply_16ic_generic(out, in_a, in_b, num_points % 4);
}
#endif

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
static inline void volk_16ic_x2_multiply_16ic_neonv8(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        const int16x8x2_t vec0 = vld2q_s16((const int16_t*)in_a);
        const int16x8x2_t vec1 = vld2q_s16((const int16_t*)in_b);
        __VOLK_PREFETCH(in_a + 16);
        __VOLK_PREFETCH(in_b + 16);
        const int16x8x2_t result = {
            { vmlsq_s16(vmulq_s16(vec0.val[0], vec1.val[0]), vec0.val[1], vec1.val[1]),
              vmlaq_s16(vmulq_s16(vec0.val[0], vec1.val[1]), vec0.val[1], vec1.val[0]) }
        };
        vst2q_s16((int16_t*)out, result);
        in_a += 8;
        in_b += 8;
        out += 8;
    }
    volk_16ic_x2_multiply_16ic_generic(out, in_a, in_b, num_points % 8);
}
#endif

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_x2_multiply_16ic_rvv(lv_16sc_t* result,
                                                  const lv_16sc_t* in_a,
                                                  const lv_16sc_t* in_b,
                                                  unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in_a += vl, in_b += vl, result += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint32m8_t va = __riscv_vle32_v_i32m8((const int32_t*)in_a, vl);
        vint32m8_t vb = __riscv_vle32_v_i32m8((const int32_t*)in_b, vl);
        vint16m4_t var = __riscv_vnsra(va, 0, vl), vai = __riscv_vnsra(va, 16, vl);
        vint16m4_t vbr = __riscv_vnsra(vb, 0, vl), vbi = __riscv_vnsra(vb, 16, vl);
        vint16m4_t vr = __riscv_vnmsac(__riscv_vmul(var, vbr, vl), vai, vbi, vl);
        vint16m4_t vi = __riscv_vmacc(__riscv_vmul(var, vbi, vl), vai, vbr, vl);
        vuint16m4_t vru = __riscv_vreinterpret_u16m4(vr);
        vuint16m4_t viu = __riscv_vreinterpret_u16m4(vi);
        vuint32m8_t v = __riscv_vwmaccu(__riscv_vwaddu_vv(vru, viu, vl), 0xFFFF, viu, vl);
        __riscv_vse32((uint32_t*)result, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_16ic_x2_multiply_16ic_rvvseg(lv_16sc_t* result,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in_a += vl, in_b += vl, result += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4x2_t va = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)in_a, vl);
        vint16m4x2_t vb = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)in_b, vl);
        vint16m4_t var = __riscv_vget_i16m4(va, 0), vai = __riscv_vget_i16m4(va, 1);
        vint16m4_t vbr = __riscv_vget_i16m4(vb, 0), vbi = __riscv_vget_i16m4(vb, 1);
        vint16m4_t vr = __riscv_vnmsac(__riscv_vmul(var, vbr, vl), vai, vbi, vl);
        vint16m4_t vi = __riscv_vmacc(__riscv_vmul(var, vbi, vl), vai, vbr, vl);
        __riscv_vsseg2e16_v_i16m4x2(
            (int16_t*)result, __riscv_vcreate_v_i16m4x2(vr, vi), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /*INCLUDED_volk_16ic_x2_multiply_16ic_H*/
