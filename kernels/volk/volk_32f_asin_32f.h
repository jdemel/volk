/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_asin_32f
 *
 * \b Overview
 *
 * Computes arcsine of input vector and stores results in output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_asin_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The vector where results will be stored.
 *
 * \b Example
 * \code
 * Calculate common angles around the top half of the unit circle.
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0;
 *   in[1] = 0.5;
 *   in[2] = std::sqrt(2.f)/2.f;
 *   in[3] = std::sqrt(3.f)/2.f;
 *   in[4] = in[5] = 1;
 *   for(unsigned int ii = 6; ii < N; ++ii){
 *       in[ii] = - in[N-ii-1];
 *   }
 *
 *   volk_32f_asin_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("asin(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

/* This is the number of terms of Taylor series to evaluate, increase this for more
 * accuracy*/
#define ASIN_TERMS 2

#ifndef INCLUDED_volk_32f_asin_32f_a_H
#define INCLUDED_volk_32f_asin_32f_a_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_asin_32f_a_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, pio2, x, y, z, arcsine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        aVal = _mm256_div_ps(aVal,
                             _mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(
            z, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm256_add_ps(x, _mm256_sqrt_ps(_mm256_fmadd_ps(x, x, fones)));
        }
        x = _mm256_div_ps(fones, x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm256_fmadd_ps(
                y, _mm256_mul_ps(x, x), _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(y, _mm256_and_ps(_mm256_fnmadd_ps(y, ftwos, pio2), condition));
        arcsine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm256_sub_ps(arcsine,
                                _mm256_and_ps(_mm256_mul_ps(arcsine, ftwos), condition));

        _mm256_store_ps(bPtr, arcsine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_asin_32f_a_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, pio2, x, y, z, arcsine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        aVal = _mm256_div_ps(aVal,
                             _mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(
            z, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm256_add_ps(x,
                              _mm256_sqrt_ps(_mm256_add_ps(fones, _mm256_mul_ps(x, x))));
        }
        x = _mm256_div_ps(fones, x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm256_add_ps(_mm256_mul_ps(y, _mm256_mul_ps(x, x)),
                              _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(
            y, _mm256_and_ps(_mm256_sub_ps(pio2, _mm256_mul_ps(y, ftwos)), condition));
        arcsine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm256_sub_ps(arcsine,
                                _mm256_and_ps(_mm256_mul_ps(arcsine, ftwos), condition));

        _mm256_store_ps(bPtr, arcsine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX for aligned */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_asin_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    int i, j;

    __m128 aVal, pio2, x, y, z, arcsine;
    __m128 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm_setzero_ps();
    fones = _mm_set1_ps(1.0);
    ftwos = _mm_set1_ps(2.0);
    ffours = _mm_set1_ps(4.0);

    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        aVal = _mm_div_ps(
            aVal,
            _mm_sqrt_ps(_mm_mul_ps(_mm_add_ps(fones, aVal), _mm_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm_cmplt_ps(z, fzeroes);
        z = _mm_sub_ps(z, _mm_and_ps(_mm_mul_ps(z, ftwos), condition));
        condition = _mm_cmplt_ps(z, fones);
        x = _mm_add_ps(z, _mm_and_ps(_mm_sub_ps(_mm_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm_add_ps(x, _mm_sqrt_ps(_mm_add_ps(fones, _mm_mul_ps(x, x))));
        }
        x = _mm_div_ps(fones, x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm_add_ps(_mm_mul_ps(y, _mm_mul_ps(x, x)),
                           _mm_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm_mul_ps(y, _mm_mul_ps(x, ffours));
        condition = _mm_cmpgt_ps(z, fones);

        y = _mm_add_ps(y, _mm_and_ps(_mm_sub_ps(pio2, _mm_mul_ps(y, ftwos)), condition));
        arcsine = y;
        condition = _mm_cmplt_ps(aVal, fzeroes);
        arcsine = _mm_sub_ps(arcsine, _mm_and_ps(_mm_mul_ps(arcsine, ftwos), condition));

        _mm_store_ps(bPtr, arcsine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = asinf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#endif /* INCLUDED_volk_32f_asin_32f_a_H */

#ifndef INCLUDED_volk_32f_asin_32f_u_H
#define INCLUDED_volk_32f_asin_32f_u_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_asin_32f_u_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, pio2, x, y, z, arcsine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        aVal = _mm256_div_ps(aVal,
                             _mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(
            z, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm256_add_ps(x, _mm256_sqrt_ps(_mm256_fmadd_ps(x, x, fones)));
        }
        x = _mm256_div_ps(fones, x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm256_fmadd_ps(
                y, _mm256_mul_ps(x, x), _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(y, _mm256_and_ps(_mm256_fnmadd_ps(y, ftwos, pio2), condition));
        arcsine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm256_sub_ps(arcsine,
                                _mm256_and_ps(_mm256_mul_ps(arcsine, ftwos), condition));

        _mm256_storeu_ps(bPtr, arcsine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_asin_32f_u_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    int i, j;

    __m256 aVal, pio2, x, y, z, arcsine;
    __m256 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm256_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm256_setzero_ps();
    fones = _mm256_set1_ps(1.0);
    ftwos = _mm256_set1_ps(2.0);
    ffours = _mm256_set1_ps(4.0);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        aVal = _mm256_div_ps(aVal,
                             _mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(fones, aVal),
                                                          _mm256_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm256_cmp_ps(z, fzeroes, _CMP_LT_OS);
        z = _mm256_sub_ps(z, _mm256_and_ps(_mm256_mul_ps(z, ftwos), condition));
        condition = _mm256_cmp_ps(z, fones, _CMP_LT_OS);
        x = _mm256_add_ps(
            z, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm256_add_ps(x,
                              _mm256_sqrt_ps(_mm256_add_ps(fones, _mm256_mul_ps(x, x))));
        }
        x = _mm256_div_ps(fones, x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm256_add_ps(_mm256_mul_ps(y, _mm256_mul_ps(x, x)),
                              _mm256_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm256_mul_ps(y, _mm256_mul_ps(x, ffours));
        condition = _mm256_cmp_ps(z, fones, _CMP_GT_OS);

        y = _mm256_add_ps(
            y, _mm256_and_ps(_mm256_sub_ps(pio2, _mm256_mul_ps(y, ftwos)), condition));
        arcsine = y;
        condition = _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS);
        arcsine = _mm256_sub_ps(arcsine,
                                _mm256_and_ps(_mm256_mul_ps(arcsine, ftwos), condition));

        _mm256_storeu_ps(bPtr, arcsine);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = asin(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX for unaligned */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_asin_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    int i, j;

    __m128 aVal, pio2, x, y, z, arcsine;
    __m128 fzeroes, fones, ftwos, ffours, condition;

    pio2 = _mm_set1_ps(3.14159265358979323846 / 2);
    fzeroes = _mm_setzero_ps();
    fones = _mm_set1_ps(1.0);
    ftwos = _mm_set1_ps(2.0);
    ffours = _mm_set1_ps(4.0);

    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        aVal = _mm_div_ps(
            aVal,
            _mm_sqrt_ps(_mm_mul_ps(_mm_add_ps(fones, aVal), _mm_sub_ps(fones, aVal))));
        z = aVal;
        condition = _mm_cmplt_ps(z, fzeroes);
        z = _mm_sub_ps(z, _mm_and_ps(_mm_mul_ps(z, ftwos), condition));
        condition = _mm_cmplt_ps(z, fones);
        x = _mm_add_ps(z, _mm_and_ps(_mm_sub_ps(_mm_div_ps(fones, z), z), condition));

        for (i = 0; i < 2; i++) {
            x = _mm_add_ps(x, _mm_sqrt_ps(_mm_add_ps(fones, _mm_mul_ps(x, x))));
        }
        x = _mm_div_ps(fones, x);
        y = fzeroes;
        for (j = ASIN_TERMS - 1; j >= 0; j--) {
            y = _mm_add_ps(_mm_mul_ps(y, _mm_mul_ps(x, x)),
                           _mm_set1_ps(pow(-1, j) / (2 * j + 1)));
        }

        y = _mm_mul_ps(y, _mm_mul_ps(x, ffours));
        condition = _mm_cmpgt_ps(z, fones);

        y = _mm_add_ps(y, _mm_and_ps(_mm_sub_ps(pio2, _mm_mul_ps(y, ftwos)), condition));
        arcsine = y;
        condition = _mm_cmplt_ps(aVal, fzeroes);
        arcsine = _mm_sub_ps(arcsine, _mm_and_ps(_mm_mul_ps(arcsine, ftwos), condition));

        _mm_storeu_ps(bPtr, arcsine);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = asinf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_asin_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = asinf(*aPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void
volk_32f_asin_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t cpio2 = __riscv_vfmv_v_f_f32m2(1.5707964f, vlmax);
    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t cf2 = __riscv_vfmv_v_f_f32m2(2.0f, vlmax);
    const vfloat32m2_t cf4 = __riscv_vfmv_v_f_f32m2(4.0f, vlmax);

#if ASIN_TERMS == 2
    const vfloat32m2_t cfm1o3 = __riscv_vfmv_v_f_f32m2(-1 / 3.0f, vlmax);
#elif ASIN_TERMS == 3
    const vfloat32m2_t cf1o5 = __riscv_vfmv_v_f_f32m2(1 / 5.0f, vlmax);
#elif ASIN_TERMS == 4
    const vfloat32m2_t cfm1o7 = __riscv_vfmv_v_f_f32m2(-1 / 7.0f, vlmax);
#endif

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v = __riscv_vle32_v_f32m2(aVector, vl);
        vfloat32m2_t a =
            __riscv_vfdiv(__riscv_vfsqrt(__riscv_vfmsac(cf1, v, v, vl), vl), v, vl);
        vfloat32m2_t z = __riscv_vfabs(a, vl);
        vfloat32m2_t x = __riscv_vfdiv_mu(__riscv_vmflt(z, cf1, vl), z, cf1, z, vl);
        x = __riscv_vfadd(x, __riscv_vfsqrt(__riscv_vfmadd(x, x, cf1, vl), vl), vl);
        x = __riscv_vfadd(x, __riscv_vfsqrt(__riscv_vfmadd(x, x, cf1, vl), vl), vl);
        x = __riscv_vfdiv(cf1, x, vl);
        vfloat32m2_t xx = __riscv_vfmul(x, x, vl);

#if ASIN_TERMS < 1
        vfloat32m2_t y = __riscv_vfmv_v_f_f32m2(0, vl);
#elif ASIN_TERMS == 1
        y = __riscv_vfmadd(y, xx, cf1, vl);
#elif ASIN_TERMS == 2
        vfloat32m2_t y = cfm1o3;
        y = __riscv_vfmadd(y, xx, cf1, vl);
#elif ASIN_TERMS == 3
        vfloat32m2_t y = cf1o5;
        y = __riscv_vfmadd(y, xx, cfm1o3, vl);
        y = __riscv_vfmadd(y, xx, cf1, vl);
#elif ASIN_TERMS == 4
        vfloat32m2_t y = cfm1o7;
        y = __riscv_vfmadd(y, xx, cf1o5, vl);
        y = __riscv_vfmadd(y, xx, cfm1o3, vl);
        y = __riscv_vfmadd(y, xx, cf1, vl);
#else
#error "ASIN_TERMS > 4 not supported by volk_32f_asin_32f_rvv"
#endif
        y = __riscv_vfmul(y, __riscv_vfmul(x, cf4, vl), vl);
        y = __riscv_vfadd_mu(
            __riscv_vmfgt(z, cf1, vl), y, y, __riscv_vfnmsub(y, cf2, cpio2, vl), vl);

        vfloat32m2_t asine;
        asine = __riscv_vfneg_mu(RISCV_VMFLTZ(32m2, a, vl), y, y, vl);

        __riscv_vse32(bVector, asine, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_asin_32f_u_H */
