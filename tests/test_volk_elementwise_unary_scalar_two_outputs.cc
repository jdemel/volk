/* -*- c++ -*- */
/*
 * Copyright 2026 Johannes Sterz Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_test_elementwise.h"
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <algorithm>
#include <random>
#include <span>

namespace {
template <typename T>
struct ScaledPairCase {
    std::string name;
    float scalar;
    volk_func_desc_t description;
    void (*kernel)(float*, float*, const T*, float, unsigned int, const char*);
    void (*generate)(std::span<T>, std::mt19937&);
};

template <typename T>
std::vector<ScaledPairCase<T>> scaled_pair_cases(
    std::string_view name,
    volk_func_desc_t descriptor,
    void (*kernel)(float*, float*, const T*, float, unsigned int, const char*),
    void (*generate)(std::span<T>, std::mt19937&))
{
    return { { std::string(name) + "_negative", -2.0f, descriptor, kernel, generate },
             { std::string(name) + "_fraction", 0.5f, descriptor, kernel, generate },
             { std::string(name) + "_positive", 3.0f, descriptor, kernel, generate } };
}
template <typename T>
class ScaledPairTestBase
    : public VolkTestImpl<std::tuple<std::string, size_t, ScaledPairCase<T>>>
{
protected:
    void initialize_data() override
    {
        input_storage.resize(this->vector_length + this->offset);
        output0_storage.resize(this->vector_length + this->offset);
        output1_storage.resize(this->vector_length + this->offset);
        expected0.resize(this->vector_length);
        expected1.resize(this->vector_length);
        input = std::span(input_storage).subspan(this->offset, this->vector_length);
        output0 = std::span(output0_storage).subspan(this->offset, this->vector_length);
        output1 = std::span(output1_storage).subspan(this->offset, this->vector_length);
        std::mt19937 engine(0xdeadbeef);
        std::get<2>(this->GetParam()).generate(input, engine);
        const float scalar = std::get<2>(this->GetParam()).scalar;
        std::transform(
            input.begin(), input.end(), expected0.begin(), [scalar](const T value) {
                return static_cast<float>(value.real()) / scalar;
            });
        std::transform(
            input.begin(), input.end(), expected1.begin(), [scalar](const T value) {
                return static_cast<float>(value.imag()) / scalar;
            });
    }
    void run()
    {
        std::get<2>(this->GetParam())
            .kernel(output0.data(),
                    output1.data(),
                    input.data(),
                    std::get<2>(this->GetParam()).scalar,
                    static_cast<unsigned int>(this->vector_length),
                    this->implementation_name.c_str());
        const auto equal = [](const float lhs, const float rhs) {
            return std::abs(lhs - rhs) <= 2e-6f * (1.0f + std::abs(lhs));
        };
        EXPECT_TRUE(compare_volk_arrays<float>(
            expected0, output0, equal, describe_volk_difference<float>));
        EXPECT_TRUE(compare_volk_arrays<float>(
            expected1, output1, equal, describe_volk_difference<float>));
    }
    volk::vector<T> input_storage;
    volk::vector<float> output0_storage, output1_storage, expected0, expected1;
    std::span<T> input;
    std::span<float> output0, output1;
};
class ScaledPair16Test : public ScaledPairTestBase<lv_16sc_t>
{
};
class ScaledPair8Test : public ScaledPairTestBase<lv_8sc_t>
{
};
TEST_P(ScaledPair16Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         ScaledPair16Test,
                         testing::ValuesIn(kernel_cases(scaled_pair_cases<lv_16sc_t>(
                             "volk_16ic_s32f_deinterleave_32f_x2",
                             volk_16ic_s32f_deinterleave_32f_x2_get_func_desc(),
                             volk_16ic_s32f_deinterleave_32f_x2_manual,
                             generate_complex16_input))),
                         ElementwiseName());
TEST_P(ScaledPair8Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         ScaledPair8Test,
                         testing::ValuesIn(kernel_cases(scaled_pair_cases<lv_8sc_t>(
                             "volk_8ic_s32f_deinterleave_32f_x2",
                             volk_8ic_s32f_deinterleave_32f_x2_get_func_desc(),
                             volk_8ic_s32f_deinterleave_32f_x2_manual,
                             generate_complex8_input))),
                         ElementwiseName());
} // namespace
