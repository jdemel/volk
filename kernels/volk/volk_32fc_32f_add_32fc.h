/* -*- c++ -*- */
/*
 * Copyright 2018 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_32f_add_32fc
 *
 * \b Overview
 *
 * Adds two vectors together element by element:
 *
 * c[i] = a[i] + b[i]
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_32f_add_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const float*
 * bVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: First vector of input points.
 * \li bVector: Second vector of input points.
 * \li num_points: The number of values in both input vector.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 *
 * The follow example adds the increasing and decreasing vectors such that the result of
 * every summation pair is 10
 *
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* increasing = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* decreasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (lv_32fc_t)ii;
 *       decreasing[ii] = 10.f - (float)ii;
 *   }
 *
 *   volk_32fc_32f_add_32fc(out, increasing, decreasing, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %1.2f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(decreasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_32f_add_32fc_u_H
#define INCLUDED_volk_32fc_32f_add_32fc_u_H

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_32f_add_32fc_generic(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  const float* bVector,
                                                  unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const float* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_32f_add_32fc_u_avx(lv_32fc_t* cVector,
                                                const lv_32fc_t* aVector,
                                                const float* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const float* bPtr = bVector;

    __m256 aVal1, aVal2, bVal, cVal1, cVal2;
    __m256 cpx_b1, cpx_b2;
    __m256 zero;
    zero = _mm256_setzero_ps();
    __m256 tmp1, tmp2;
    for (; number < eighthPoints; number++) {

        aVal1 = _mm256_loadu_ps((float*)aPtr);
        aVal2 = _mm256_loadu_ps((float*)(aPtr + 4));
        bVal = _mm256_loadu_ps(bPtr);
        cpx_b1 = _mm256_unpacklo_ps(bVal, zero); // b0, 0, b1, 0, b4, 0, b5, 0
        cpx_b2 = _mm256_unpackhi_ps(bVal, zero); // b2, 0, b3, 0, b6, 0, b7, 0

        tmp1 = _mm256_permute2f128_ps(cpx_b1, cpx_b2, 0x0 + (0x2 << 4));
        tmp2 = _mm256_permute2f128_ps(cpx_b1, cpx_b2, 0x1 + (0x3 << 4));

        cVal1 = _mm256_add_ps(aVal1, tmp1);
        cVal2 = _mm256_add_ps(aVal2, tmp2);

        _mm256_storeu_ps((float*)cPtr,
                         cVal1); // Store the results back into the C container
        _mm256_storeu_ps((float*)(cPtr + 4),
                         cVal2); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_32f_add_32fc_a_avx(lv_32fc_t* cVector,
                                                const lv_32fc_t* aVector,
                                                const float* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const float* bPtr = bVector;

    __m256 aVal1, aVal2, bVal, cVal1, cVal2;
    __m256 cpx_b1, cpx_b2;
    __m256 zero;
    zero = _mm256_setzero_ps();
    __m256 tmp1, tmp2;
    for (; number < eighthPoints; number++) {

        aVal1 = _mm256_load_ps((float*)aPtr);
        aVal2 = _mm256_load_ps((float*)(aPtr + 4));
        bVal = _mm256_load_ps(bPtr);
        cpx_b1 = _mm256_unpacklo_ps(bVal, zero); // b0, 0, b1, 0, b4, 0, b5, 0
        cpx_b2 = _mm256_unpackhi_ps(bVal, zero); // b2, 0, b3, 0, b6, 0, b7, 0

        tmp1 = _mm256_permute2f128_ps(cpx_b1, cpx_b2, 0x0 + (0x2 << 4));
        tmp2 = _mm256_permute2f128_ps(cpx_b1, cpx_b2, 0x1 + (0x3 << 4));

        cVal1 = _mm256_add_ps(aVal1, tmp1);
        cVal2 = _mm256_add_ps(aVal2, tmp2);

        _mm256_store_ps((float*)cPtr,
                        cVal1); // Store the results back into the C container
        _mm256_store_ps((float*)(cPtr + 4),
                        cVal2); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_32f_add_32fc_neon(lv_32fc_t* cVector,
                                               const lv_32fc_t* aVector,
                                               const float* bVector,
                                               unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const float* bPtr = bVector;

    float32x4x4_t aVal0, aVal1;
    float32x4x2_t bVal0, bVal1;

    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;
    for (; number < sixteenthPoints; number++) {
        aVal0 = vld4q_f32((const float*)aPtr);
        aPtr += 8;
        aVal1 = vld4q_f32((const float*)aPtr);
        aPtr += 8;
        __VOLK_PREFETCH(aPtr + 16);

        bVal0 = vld2q_f32((const float*)bPtr);
        bPtr += 8;
        bVal1 = vld2q_f32((const float*)bPtr);
        bPtr += 8;
        __VOLK_PREFETCH(bPtr + 16);

        aVal0.val[0] = vaddq_f32(aVal0.val[0], bVal0.val[0]);
        aVal0.val[2] = vaddq_f32(aVal0.val[2], bVal0.val[1]);

        aVal1.val[2] = vaddq_f32(aVal1.val[2], bVal1.val[1]);
        aVal1.val[0] = vaddq_f32(aVal1.val[0], bVal1.val[0]);

        vst4q_f32((float*)(cPtr), aVal0);
        cPtr += 8;
        vst4q_f32((float*)(cPtr), aVal1);
        cPtr += 8;
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_32f_add_32fc_rvv(lv_32fc_t* cVector,
                                              const lv_32fc_t* aVector,
                                              const float* bVector,
                                              unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, cVector += vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m8_t vc = __riscv_vle32_v_f32m8((const float*)aVector, vl * 2);
        vuint32m4_t v = __riscv_vle32_v_u32m4((const uint32_t*)bVector, vl);
        vfloat32m8_t vf = __riscv_vreinterpret_f32m8(
            __riscv_vreinterpret_u32m8(__riscv_vzext_vf2_u64m8(v, vl)));
        __riscv_vse32((float*)cVector, __riscv_vfadd(vc, vf, vl * 2), vl * 2);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_32f_add_32fc_a_H */
