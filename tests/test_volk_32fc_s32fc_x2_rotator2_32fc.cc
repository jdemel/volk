/* -*- c++ -*- */
/*
 * Copyright 2025 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk/volk_complex.h"
#include "volk_test.h"
#include <gtest/gtest.h>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <algorithm>
#include <complex>
#include <span>

class volk_32fc_s32fc_x2_rotator2_32fc_test : public VolkTest
{
protected:
    void initialize_data() override
    {
        const size_t length = vector_length;
        // Tolerance scales with sample count because all complex-multiplication-based
        // rotators accumulate floating-point error. The error grows roughly as O(sqrt(N))
        // due to random walk behavior of rounding errors.
        //
        // For implementations using inc^K (K=2 for SSE, K=4 for AVX, K=8 for AVX512),
        // the error per step is larger, leading to faster divergence.
        if (length < 16) {
            absolute_error = 1.e-6f;
        } else if (length < 128) {
            absolute_error = 1.e-5f;
        } else if (length < 65536) {
            absolute_error = 1.e-4f;
        } else if (length < 1000000) {
            absolute_error = 2.e-3f;
        } else if (length < 10000000) {
            // 1M-10M samples: relax tolerance for accumulated error
            absolute_error = 2.e-2f;
        } else {
            // 10M+ samples: At this scale, error exceeds useful tolerance
            // Applications requiring accuracy at this scale should use
            // implementations with periodic angle-based resync
            absolute_error = 5.e-2f;
        }

        input_storage.resize(length + offset);
        result_storage.resize(length + offset);
        input = std::span(input_storage).subspan(offset);
        result = std::span(result_storage).subspan(offset);
        result_magnitude = volk::vector<float>(length);

        const float initial_phase = initial_phase_steps * increment;
        phase_increment = std::polar(1.0f, increment);
        phase = std::polar(1.0f, initial_phase);

        for (size_t i = 0; i < length; ++i) {
            input[i] = std::complex<float>(
                static_cast<float>(2.0f * std::cos(2.0f * M_PI * static_cast<double>(i) /
                                                   static_cast<double>(length))),
                static_cast<float>(2.0f *
                                   std::sin(0.3f + 2.0f * M_PI * static_cast<double>(i) /
                                                       static_cast<double>(length))));
        }

        // Calculate expected results using double precision for angle computation
        // This is critical for numerical stability at high sample counts.
        // Using float would lose precision for i > ~10M samples.
        expected = volk::vector<lv_32fc_t>(length);
        const double initial_phase_d = static_cast<double>(initial_phase);
        const double increment_d = static_cast<double>(increment);
        for (size_t i = 0; i < length; ++i) {
            double angle = initial_phase_d + static_cast<double>(i) * increment_d;
            // Reduce angle to [-π, π] for sincos accuracy at large angles
            angle = std::fmod(angle, 2.0 * M_PI);
            if (angle > M_PI)
                angle -= 2.0 * M_PI;
            else if (angle < -M_PI)
                angle += 2.0 * M_PI;
            expected[i] = input[i] * std::polar(1.0f, static_cast<float>(angle));
        }

        expected_magnitude = volk::vector<float>(length);
        for (size_t i = 0; i < length; ++i) {
            expected_magnitude[i] = std::abs(input[i]);
        }
    }

    void run()
    {
        const double final_angle = std::arg(std::complex<double>(phase)) +
                                   static_cast<double>(vector_length) *
                                       std::arg(std::complex<double>(phase_increment));
        const auto expected_phase = std::polar(1.0, final_angle);
        volk_32fc_s32fc_x2_rotator2_32fc_manual(result.data(),
                                                input.data(),
                                                &phase_increment,
                                                &phase,
                                                static_cast<unsigned int>(vector_length),
                                                implementation_name.c_str());
        EXPECT_NEAR(phase.real(), expected_phase.real(), absolute_error);
        EXPECT_NEAR(phase.imag(), expected_phase.imag(), absolute_error);
        std::transform(result.begin(),
                       result.end(),
                       result_magnitude.begin(),
                       [](const auto value) { return std::abs(value); });
        EXPECT_TRUE(AreFloatingPointArraysEqualWithAbsoluteError<float>(
            expected_magnitude, result_magnitude, absolute_magnitue_error));
        EXPECT_TRUE(
            AreComplexFloatingPointArraysEqualWithAbsoluteError<std::complex<float>>(
                expected, result, absolute_error));
    }

    static constexpr float increment = 0.07f;
    static constexpr float initial_phase_steps = 3.0f;
    static constexpr float absolute_magnitue_error = 1.0e-4f;
    float absolute_error{};
    volk::vector<lv_32fc_t> input_storage;
    volk::vector<lv_32fc_t> result_storage;
    std::span<lv_32fc_t> input;
    std::span<lv_32fc_t> result;
    lv_32fc_t phase_increment;
    lv_32fc_t phase;
    volk::vector<lv_32fc_t> expected;
    volk::vector<float> expected_magnitude;
    volk::vector<float> result_magnitude;
};

TEST_P(volk_32fc_s32fc_x2_rotator2_32fc_test, run)
{
    report_runtime([this] { run(); });
}

INSTANTIATE_TEST_SUITE_P(
    volk_32fc_s32fc_x2_rotator2_32fc,
    volk_32fc_s32fc_x2_rotator2_32fc_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32fc_s32fc_x2_rotator2_32fc_get_func_desc())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

// Large-scale tests for numerical stability verification
// Note: rotator2 implementations using complex multiplication (phase *= inc^N)
// accumulate floating-point error that grows with sample count.
// At 2M samples, error is typically ~1-2% for AVX (inc^4) implementations.
// For >10M samples, consider using implementations with periodic angle resync.
static constexpr std::array<size_t, 1> large_vector_sizes{ 2000000 };

INSTANTIATE_TEST_SUITE_P(
    volk_32fc_s32fc_x2_rotator2_32fc_large,
    volk_32fc_s32fc_x2_rotator2_32fc_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32fc_s32fc_x2_rotator2_32fc_get_func_desc())),
                     testing::ValuesIn(large_vector_sizes)),
    generate_volk_test_name());
