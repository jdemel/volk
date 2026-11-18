/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_tanh_32f
 *
 * \b Overview
 *
 * Computes the hyperbolic tangent of each element of the aVector:
 *
 * c[i] = tanh(a[i])
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_tanh_32f(float* cVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The buffer of points.
 * \li num_points: The number of values in input buffer.
 *
 * \b Outputs
 * \li cVector: The output buffer.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       // the approximate artanh(x) for x<1
 *       float x = (float)ii / (float)N;
 *       in[ii] = 0.5 * std::log((1.f+x)/(1.f-x));
 *   }
 *
 *   volk_32f_tanh_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_tanh_32f_a_H
#define INCLUDED_volk_32f_tanh_32f_a_H

#include <math.h>
#include <stddef.h>


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_tanh_32f_generic(float* cVector, const float* aVector, unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        *cVector++ = tanhf(*aVector++);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_tanh_32f_series(float* cVector, const float* aVector, unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        const float x = *aVector++;
        if (x >= 6.0f) {
            *cVector++ = 1.0f;
        } else if (x <= -6.0f) {
            *cVector++ = -1.0f;
        } else {
            const float x2 = x * x;
            // Cubic rational approximation in x², fitted over [-6, 6].
            const float a =
                x * (219739.7471f + x2 * (27329.53076f + x2 * (527.4970102f + x2)));
            const float b = 219739.7471f +
                            x2 * (100575.549f + x2 * (4754.881259f + x2 * 34.30905948f));
            *cVector++ = a / b;
        }
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_tanh_32f_a_sse(float* cVector, const float* aVector, unsigned int num_points)
{
    const __m128 n0 = _mm_set1_ps(219739.7471f);
    const __m128 n1 = _mm_set1_ps(27329.53076f);
    const __m128 n2 = _mm_set1_ps(527.4970102f);
    const __m128 d1 = _mm_set1_ps(100575.549f);
    const __m128 d2 = _mm_set1_ps(4754.881259f);
    const __m128 d3 = _mm_set1_ps(34.30905948f);
    const unsigned int blocks = num_points / 4;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m128 x = _mm_load_ps(aVector);
        const __m128 x2 = _mm_mul_ps(x, x);
        const __m128 a = _mm_mul_ps(
            x,
            _mm_add_ps(_mm_mul_ps(x2, _mm_add_ps(_mm_mul_ps(x2, _mm_add_ps(x2, n2)), n1)),
                       n0));
        const __m128 b = _mm_add_ps(
            _mm_mul_ps(
                x2, _mm_add_ps(_mm_mul_ps(x2, _mm_add_ps(_mm_mul_ps(x2, d3), d2)), d1)),
            n0);
        _mm_store_ps(cVector, _mm_div_ps(a, b));
        aVector += 4;
        cVector += 4;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 4);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_tanh_32f_a_avx(float* cVector, const float* aVector, unsigned int num_points)
{
    const __m256 n0 = _mm256_set1_ps(219739.7471f);
    const __m256 n1 = _mm256_set1_ps(27329.53076f);
    const __m256 n2 = _mm256_set1_ps(527.4970102f);
    const __m256 d1 = _mm256_set1_ps(100575.549f);
    const __m256 d2 = _mm256_set1_ps(4754.881259f);
    const __m256 d3 = _mm256_set1_ps(34.30905948f);
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256 x = _mm256_load_ps(aVector);
        const __m256 x2 = _mm256_mul_ps(x, x);
        const __m256 a = _mm256_mul_ps(
            x,
            _mm256_add_ps(
                _mm256_mul_ps(
                    x2, _mm256_add_ps(_mm256_mul_ps(x2, _mm256_add_ps(x2, n2)), n1)),
                n0));
        const __m256 b = _mm256_add_ps(
            _mm256_mul_ps(
                x2,
                _mm256_add_ps(_mm256_mul_ps(x2, _mm256_add_ps(_mm256_mul_ps(x2, d3), d2)),
                              d1)),
            n0);
        _mm256_store_ps(cVector, _mm256_div_ps(a, b));
        aVector += 8;
        cVector += 8;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 8);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tanh_32f_a_avx_fma(float* cVector, const float* aVector, unsigned int num_points)
{
    const __m256 n0 = _mm256_set1_ps(219739.7471f);
    const __m256 n1 = _mm256_set1_ps(27329.53076f);
    const __m256 n2 = _mm256_set1_ps(527.4970102f);
    const __m256 d1 = _mm256_set1_ps(100575.549f);
    const __m256 d2 = _mm256_set1_ps(4754.881259f);
    const __m256 d3 = _mm256_set1_ps(34.30905948f);
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256 x = _mm256_load_ps(aVector);
        const __m256 x2 = _mm256_mul_ps(x, x);
        const __m256 a = _mm256_mul_ps(
            x, _mm256_fmadd_ps(x2, _mm256_fmadd_ps(x2, _mm256_add_ps(x2, n2), n1), n0));
        const __m256 b =
            _mm256_fmadd_ps(x2, _mm256_fmadd_ps(x2, _mm256_fmadd_ps(x2, d3, d2), d1), n0);
        _mm256_store_ps(cVector, _mm256_div_ps(a, b));
        aVector += 8;
        cVector += 8;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 8);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_tanh_32f_neon(float* cVector, const float* aVector, unsigned int num_points)
{
    const float32x4_t n0 = vdupq_n_f32(219739.7471f);
    const float32x4_t n1 = vdupq_n_f32(27329.53076f);
    const float32x4_t n2 = vdupq_n_f32(527.4970102f);
    const float32x4_t d1 = vdupq_n_f32(100575.549f);
    const float32x4_t d2 = vdupq_n_f32(4754.881259f);
    const float32x4_t d3 = vdupq_n_f32(34.30905948f);
    const unsigned int blocks = num_points / 4;
    for (unsigned int number = 0; number < blocks; ++number) {
        __VOLK_PREFETCH(aVector + 16);
        const float32x4_t x0 = vld1q_f32(aVector);
        const float32x4_t x20 = vmulq_f32(x0, x0);
        const float32x4_t a0 =
            vmulq_f32(x0, vmlaq_f32(n0, x20, vmlaq_f32(n1, x20, vaddq_f32(n2, x20))));
        const float32x4_t b0 =
            vmlaq_f32(n0, x20, vmlaq_f32(d1, x20, vmlaq_f32(d2, x20, d3)));
        const float32x4_t r0 = vrecpeq_f32(b0);
        const float32x4_t r1 = vmulq_f32(r0, vrecpsq_f32(b0, r0));
        const float32x4_t r2 = vmulq_f32(r1, vrecpsq_f32(b0, r1));
        const float32x4_t y0 = vmulq_f32(a0, r2);
        vst1q_f32(cVector, y0);
        aVector += 4;
        cVector += 4;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 4);
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_tanh_32f_neonv8(float* cVector, const float* aVector, unsigned int num_points)
{
    const float32x4_t n0 = vdupq_n_f32(219739.7471f);
    const float32x4_t n1 = vdupq_n_f32(27329.53076f);
    const float32x4_t n2 = vdupq_n_f32(527.4970102f);
    const float32x4_t d1 = vdupq_n_f32(100575.549f);
    const float32x4_t d2 = vdupq_n_f32(4754.881259f);
    const float32x4_t d3 = vdupq_n_f32(34.30905948f);
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        __VOLK_PREFETCH(aVector + 16);
        const float32x4_t x0 = vld1q_f32(aVector);
        const float32x4_t x1 = vld1q_f32(aVector + 4);
        const float32x4_t x20 = vmulq_f32(x0, x0);
        const float32x4_t x21 = vmulq_f32(x1, x1);
        const float32x4_t a0 =
            vmulq_f32(x0, vfmaq_f32(n0, x20, vfmaq_f32(n1, x20, vaddq_f32(n2, x20))));
        const float32x4_t a1 =
            vmulq_f32(x1, vfmaq_f32(n0, x21, vfmaq_f32(n1, x21, vaddq_f32(n2, x21))));
        const float32x4_t b0 =
            vfmaq_f32(n0, x20, vfmaq_f32(d1, x20, vfmaq_f32(d2, x20, d3)));
        const float32x4_t b1 =
            vfmaq_f32(n0, x21, vfmaq_f32(d1, x21, vfmaq_f32(d2, x21, d3)));
        const float32x4_t y0 = vdivq_f32(a0, b0);
        const float32x4_t y1 = vdivq_f32(a1, b1);
        vst1q_f32(cVector, y0);
        vst1q_f32(cVector + 4, y1);
        aVector += 8;
        cVector += 8;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 8);
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
static inline void
volk_32f_tanh_32f_a_avx512f(float* cVector, const float* aVector, unsigned int num_points)
{
    const __m512 n0 = _mm512_set1_ps(219739.7471f);
    const __m512 n1 = _mm512_set1_ps(27329.53076f);
    const __m512 n2 = _mm512_set1_ps(527.4970102f);
    const __m512 d1 = _mm512_set1_ps(100575.549f);
    const __m512 d2 = _mm512_set1_ps(4754.881259f);
    const __m512 d3 = _mm512_set1_ps(34.30905948f);
    const unsigned int blocks = num_points / 16;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m512 x = _mm512_load_ps(aVector);
        const __m512 x2 = _mm512_mul_ps(x, x);
        const __m512 a = _mm512_mul_ps(
            x,
            _mm512_add_ps(
                _mm512_mul_ps(
                    x2, _mm512_add_ps(_mm512_mul_ps(x2, _mm512_add_ps(x2, n2)), n1)),
                n0));
        const __m512 b = _mm512_add_ps(
            _mm512_mul_ps(
                x2,
                _mm512_add_ps(_mm512_mul_ps(x2, _mm512_add_ps(_mm512_mul_ps(x2, d3), d2)),
                              d1)),
            n0);
        _mm512_store_ps(cVector, _mm512_div_ps(a, b));
        aVector += 16;
        cVector += 16;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 16);
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32f_tanh_32f_a_H */


#ifndef INCLUDED_volk_32f_tanh_32f_u_H
#define INCLUDED_volk_32f_tanh_32f_u_H


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_tanh_32f_u_sse(float* cVector, const float* aVector, unsigned int num_points)
{
    const __m128 n0 = _mm_set1_ps(219739.7471f);
    const __m128 n1 = _mm_set1_ps(27329.53076f);
    const __m128 n2 = _mm_set1_ps(527.4970102f);
    const __m128 d1 = _mm_set1_ps(100575.549f);
    const __m128 d2 = _mm_set1_ps(4754.881259f);
    const __m128 d3 = _mm_set1_ps(34.30905948f);
    const unsigned int blocks = num_points / 4;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m128 x = _mm_loadu_ps(aVector);
        const __m128 x2 = _mm_mul_ps(x, x);
        const __m128 a = _mm_mul_ps(
            x,
            _mm_add_ps(_mm_mul_ps(x2, _mm_add_ps(_mm_mul_ps(x2, _mm_add_ps(x2, n2)), n1)),
                       n0));
        const __m128 b = _mm_add_ps(
            _mm_mul_ps(
                x2, _mm_add_ps(_mm_mul_ps(x2, _mm_add_ps(_mm_mul_ps(x2, d3), d2)), d1)),
            n0);
        _mm_storeu_ps(cVector, _mm_div_ps(a, b));
        aVector += 4;
        cVector += 4;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 4);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_tanh_32f_u_avx(float* cVector, const float* aVector, unsigned int num_points)
{
    const __m256 n0 = _mm256_set1_ps(219739.7471f);
    const __m256 n1 = _mm256_set1_ps(27329.53076f);
    const __m256 n2 = _mm256_set1_ps(527.4970102f);
    const __m256 d1 = _mm256_set1_ps(100575.549f);
    const __m256 d2 = _mm256_set1_ps(4754.881259f);
    const __m256 d3 = _mm256_set1_ps(34.30905948f);
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256 x = _mm256_loadu_ps(aVector);
        const __m256 x2 = _mm256_mul_ps(x, x);
        const __m256 a = _mm256_mul_ps(
            x,
            _mm256_add_ps(
                _mm256_mul_ps(
                    x2, _mm256_add_ps(_mm256_mul_ps(x2, _mm256_add_ps(x2, n2)), n1)),
                n0));
        const __m256 b = _mm256_add_ps(
            _mm256_mul_ps(
                x2,
                _mm256_add_ps(_mm256_mul_ps(x2, _mm256_add_ps(_mm256_mul_ps(x2, d3), d2)),
                              d1)),
            n0);
        _mm256_storeu_ps(cVector, _mm256_div_ps(a, b));
        aVector += 8;
        cVector += 8;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 8);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tanh_32f_u_avx_fma(float* cVector, const float* aVector, unsigned int num_points)
{
    const __m256 n0 = _mm256_set1_ps(219739.7471f);
    const __m256 n1 = _mm256_set1_ps(27329.53076f);
    const __m256 n2 = _mm256_set1_ps(527.4970102f);
    const __m256 d1 = _mm256_set1_ps(100575.549f);
    const __m256 d2 = _mm256_set1_ps(4754.881259f);
    const __m256 d3 = _mm256_set1_ps(34.30905948f);
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256 x = _mm256_loadu_ps(aVector);
        const __m256 x2 = _mm256_mul_ps(x, x);
        const __m256 a = _mm256_mul_ps(
            x, _mm256_fmadd_ps(x2, _mm256_fmadd_ps(x2, _mm256_add_ps(x2, n2), n1), n0));
        const __m256 b =
            _mm256_fmadd_ps(x2, _mm256_fmadd_ps(x2, _mm256_fmadd_ps(x2, d3, d2), d1), n0);
        _mm256_storeu_ps(cVector, _mm256_div_ps(a, b));
        aVector += 8;
        cVector += 8;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 8);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
static inline void
volk_32f_tanh_32f_u_avx512f(float* cVector, const float* aVector, unsigned int num_points)
{
    const __m512 n0 = _mm512_set1_ps(219739.7471f);
    const __m512 n1 = _mm512_set1_ps(27329.53076f);
    const __m512 n2 = _mm512_set1_ps(527.4970102f);
    const __m512 d1 = _mm512_set1_ps(100575.549f);
    const __m512 d2 = _mm512_set1_ps(4754.881259f);
    const __m512 d3 = _mm512_set1_ps(34.30905948f);
    const unsigned int blocks = num_points / 16;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m512 x = _mm512_loadu_ps(aVector);
        const __m512 x2 = _mm512_mul_ps(x, x);
        const __m512 a = _mm512_mul_ps(
            x,
            _mm512_add_ps(
                _mm512_mul_ps(
                    x2, _mm512_add_ps(_mm512_mul_ps(x2, _mm512_add_ps(x2, n2)), n1)),
                n0));
        const __m512 b = _mm512_add_ps(
            _mm512_mul_ps(
                x2,
                _mm512_add_ps(_mm512_mul_ps(x2, _mm512_add_ps(_mm512_mul_ps(x2, d3), d2)),
                              d1)),
            n0);
        _mm512_storeu_ps(cVector, _mm512_div_ps(a, b));
        aVector += 16;
        cVector += 16;
    }
    volk_32f_tanh_32f_generic(cVector, aVector, num_points % 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_tanh_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(219739.7471f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(27329.53076f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(527.4970102f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(100575.549f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(4754.881259f, vlmax);
    const vfloat32m2_t c6 = __riscv_vfmv_v_f_f32m2(34.30905948f, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t x = __riscv_vle32_v_f32m2(aVector, vl);
        vfloat32m2_t xx = __riscv_vfmul(x, x, vl);
        vfloat32m2_t a, b;
        a = __riscv_vfadd(xx, c3, vl);
        a = __riscv_vfmadd(a, xx, c2, vl);
        a = __riscv_vfmadd(a, xx, c1, vl);
        a = __riscv_vfmul(a, x, vl);
        b = c6;
        b = __riscv_vfmadd(b, xx, c5, vl);
        b = __riscv_vfmadd(b, xx, c4, vl);
        b = __riscv_vfmadd(b, xx, c1, vl);
        __riscv_vse32(bVector, __riscv_vfdiv(a, b, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_tanh_32f_u_H */
