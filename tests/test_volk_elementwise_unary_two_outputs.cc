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
class SincosTest : public VolkTest
{
protected:
    void initialize_data() override
    {
        input_storage.resize(vector_length + offset);
        sin_storage.resize(vector_length + offset);
        cos_storage.resize(vector_length + offset);
        expected_sin.resize(vector_length);
        expected_cos.resize(vector_length);
        input = std::span(input_storage).subspan(offset, vector_length);
        result_sin = std::span(sin_storage).subspan(offset, vector_length);
        result_cos = std::span(cos_storage).subspan(offset, vector_length);
        std::mt19937 engine(0xdeadbeef);
        std::normal_distribution<float> distribution(0.0f, 100.0f);
        generate_normal_input(input, engine, distribution);
        std::transform(input.begin(),
                       input.end(),
                       expected_sin.begin(),
                       [](const float value) { return std::sin(value); });
        std::transform(input.begin(),
                       input.end(),
                       expected_cos.begin(),
                       [](const float value) { return std::cos(value); });
    }
    void run()
    {
        volk_32f_sincos_32f_x2_manual(result_sin.data(),
                                      result_cos.data(),
                                      input.data(),
                                      static_cast<unsigned int>(vector_length),
                                      implementation_name.c_str());
        EXPECT_TRUE(AreFloatingPointArraysEqualWithTolerance<float>(
            expected_sin, result_sin, 2e-5f));
        EXPECT_TRUE(AreFloatingPointArraysEqualWithTolerance<float>(
            expected_cos, result_cos, 2e-5f));
    }
    volk::vector<float> input_storage, sin_storage, cos_storage, expected_sin,
        expected_cos;
    std::span<float> input, result_sin, result_cos;
};
TEST_P(SincosTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    volk_32f_sincos_32f_x2,
    SincosTest,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32f_sincos_32f_x2_get_func_desc())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

template <typename In, typename Out>
struct DeinterleaveCase {
    std::string name;
    void (*kernel)(Out*, Out*, const In*, unsigned int, const char*);
    Out (*component)(In, bool);
    void (*generate)(std::span<In>, std::mt19937&);
    volk_func_desc_t description;
};
template <typename In, typename Out>
class DeinterleaveTestBase
    : public VolkTestImpl<std::tuple<std::string, size_t, DeinterleaveCase<In, Out>>>
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
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.generate(input, engine);
        std::transform(
            input.begin(), input.end(), expected0.begin(), [&kernel](const In value) {
                return kernel.component(value, false);
            });
        std::transform(
            input.begin(), input.end(), expected1.begin(), [&kernel](const In value) {
                return kernel.component(value, true);
            });
    }
    void run()
    {
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.kernel(output0.data(),
                      output1.data(),
                      input.data(),
                      static_cast<unsigned int>(this->vector_length),
                      this->implementation_name.c_str());
        if constexpr (std::is_integral_v<Out>) {
            EXPECT_TRUE(AreIntegerArraysEqual<Out>(expected0, output0));
            EXPECT_TRUE(AreIntegerArraysEqual<Out>(expected1, output1));
        } else {
            EXPECT_TRUE(AreFloatingPointArraysEqualWithTolerance<Out>(
                expected0, output0, 0.0f));
            EXPECT_TRUE(AreFloatingPointArraysEqualWithTolerance<Out>(
                expected1, output1, 0.0f));
        }
    }
    volk::vector<In> input_storage;
    volk::vector<Out> output0_storage, output1_storage, expected0, expected1;
    std::span<In> input;
    std::span<Out> output0, output1;
};
class Deinterleave16Test : public DeinterleaveTestBase<lv_16sc_t, int16_t>
{
};
class Deinterleave8Test : public DeinterleaveTestBase<lv_8sc_t, int16_t>
{
};
class DeinterleaveFloatTest : public DeinterleaveTestBase<lv_32fc_t, float>
{
};
class DeinterleaveDoubleTest : public DeinterleaveTestBase<lv_32fc_t, double>
{
};

TEST_P(Deinterleave16Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Deinterleave16Test,
    testing::ValuesIn(kernel_cases(std::vector<DeinterleaveCase<lv_16sc_t, int16_t>>{
        { "volk_16ic_deinterleave_16i_x2",
          volk_16ic_deinterleave_16i_x2_manual,
          [](const lv_16sc_t value, const bool imaginary) -> int16_t {
              return static_cast<int16_t>(imaginary ? value.imag() : value.real());
          },
          generate_complex16_input,
          volk_16ic_deinterleave_16i_x2_get_func_desc() } })),
    ElementwiseName());

TEST_P(Deinterleave8Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Deinterleave8Test,
    testing::ValuesIn(kernel_cases(std::vector<DeinterleaveCase<lv_8sc_t, int16_t>>{
        { "volk_8ic_deinterleave_16i_x2",
          volk_8ic_deinterleave_16i_x2_manual,
          [](const lv_8sc_t value, const bool imaginary) -> int16_t {
              return static_cast<int16_t>(
                  static_cast<int>(imaginary ? value.imag() : value.real()) * 256);
          },
          generate_complex8_input,
          volk_8ic_deinterleave_16i_x2_get_func_desc() } })),
    ElementwiseName());

TEST_P(DeinterleaveFloatTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    DeinterleaveFloatTest,
    testing::ValuesIn(kernel_cases(std::vector<DeinterleaveCase<lv_32fc_t, float>>{
        { "volk_32fc_deinterleave_32f_x2",
          volk_32fc_deinterleave_32f_x2_manual,
          [](const lv_32fc_t value, const bool imaginary) -> float {
              return imaginary ? value.imag() : value.real();
          },
          generate_complex_float_input,
          volk_32fc_deinterleave_32f_x2_get_func_desc() } })),
    ElementwiseName());

TEST_P(DeinterleaveDoubleTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    DeinterleaveDoubleTest,
    testing::ValuesIn(kernel_cases(std::vector<DeinterleaveCase<lv_32fc_t, double>>{
        { "volk_32fc_deinterleave_64f_x2",
          volk_32fc_deinterleave_64f_x2_manual,
          [](const lv_32fc_t value, const bool imaginary) -> double {
              return static_cast<double>(imaginary ? value.imag() : value.real());
          },
          generate_complex_float_input,
          volk_32fc_deinterleave_64f_x2_get_func_desc() } })),
    ElementwiseName());

} // namespace
