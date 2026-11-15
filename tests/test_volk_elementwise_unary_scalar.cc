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
#include <functional>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {
struct ScalarCase {
    std::string name;
    void (*kernel)(float*, const float*, float, unsigned int, const char*);
    std::function<float(float, float)> reference;
    float scalar;
    volk_func_desc_t description;
    void (*generate)(std::span<float>, std::mt19937&);
};

std::vector<ScalarCase> scalar_kernels()
{
    std::vector<ScalarCase> cases;
    for (const auto& [label, scalar] : { std::pair{ "negative", -2.0f },
                                         std::pair{ "zero", 0.0f },
                                         std::pair{ "fraction", 0.5f },
                                         std::pair{ "positive", 3.0f } }) {
        cases.push_back({ std::string("volk_32f_s32f_add_32f_") + label,
                          volk_32f_s32f_add_32f_manual,
                          [](const float x, const float factor) { return x + factor; },
                          scalar,
                          volk_32f_s32f_add_32f_get_func_desc(),
                          generate_scalar_input });
        cases.push_back({ std::string("volk_32f_s32f_multiply_32f_") + label,
                          volk_32f_s32f_multiply_32f_manual,
                          [](const float x, const float factor) { return x * factor; },
                          scalar,
                          volk_32f_s32f_multiply_32f_get_func_desc(),
                          generate_scalar_input });
        cases.push_back(
            { std::string("volk_32f_s32f_power_32f_") + label,
              volk_32f_s32f_power_32f_manual,
              [](const float value, const float power) { return std::pow(value, power); },
              scalar,
              volk_32f_s32f_power_32f_get_func_desc(),
              generate_power_input });
    }
    return cases;
}

class ScalarFloatTest : public VolkTestImpl<std::tuple<std::string, size_t, ScalarCase>>
{
protected:
    void initialize_data() override
    {
        input_storage.resize(vector_length + offset);
        output_storage.resize(vector_length + offset);
        expected.resize(vector_length);
        input = std::span(input_storage).subspan(offset, vector_length);
        output = std::span(output_storage).subspan(offset, vector_length);
        std::mt19937 engine(0xdeadbeef);
        const auto& kernel = std::get<2>(GetParam());
        kernel.generate(input, engine);
        std::transform(
            input.begin(), input.end(), expected.begin(), [&kernel](const float x) {
                return kernel.reference(x, kernel.scalar);
            });
    }

    void run()
    {
        const auto& kernel = std::get<2>(GetParam());
        kernel.kernel(output.data(),
                      input.data(),
                      kernel.scalar,
                      static_cast<unsigned int>(vector_length),
                      implementation_name.c_str());
        const auto equal = [](const float lhs, const float rhs) {
            return std::abs(lhs - rhs) <= 2e-6f * (1.0f + std::abs(lhs));
        };
        EXPECT_TRUE(compare_volk_arrays<float>(
            expected, output, equal, describe_volk_difference<float>));
    }

    volk::vector<float> input_storage;
    volk::vector<float> output_storage;
    volk::vector<float> expected;
    std::span<float> input;
    std::span<float> output;
};

TEST_P(ScalarFloatTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         ScalarFloatTest,
                         testing::ValuesIn(kernel_cases(scalar_kernels())),
                         ElementwiseName());

template <typename In, typename Out>
struct ScaledComplexCase {
    std::string name;
    void (*kernel)(Out*, const In*, float, unsigned int, const char*);
    Out (*reference)(In, float);
    void (*generate)(std::span<In>, std::mt19937&);
    float scalar;
    volk_func_desc_t description;
};
template <typename In, typename Out>
class ScaledComplexTestBase
    : public VolkTestImpl<std::tuple<std::string, size_t, ScaledComplexCase<In, Out>>>
{
protected:
    void initialize_data() override
    {
        input_storage.resize(this->vector_length + this->offset);
        output_storage.resize(this->vector_length + this->offset);
        expected.resize(this->vector_length);
        input = std::span(input_storage).subspan(this->offset, this->vector_length);
        output = std::span(output_storage).subspan(this->offset, this->vector_length);
        const auto& kernel = std::get<2>(this->GetParam());
        std::mt19937 engine(0xdeadbeef);
        kernel.generate(input, engine);
        std::transform(
            input.begin(), input.end(), expected.begin(), [&kernel](const In value) {
                return kernel.reference(value, kernel.scalar);
            });
    }
    void run()
    {
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.kernel(output.data(),
                      input.data(),
                      kernel.scalar,
                      static_cast<unsigned int>(this->vector_length),
                      this->implementation_name.c_str());
        if constexpr (std::is_integral_v<Out>) {
            EXPECT_TRUE(AreIntegerArraysEqual<Out>(expected, output));
        } else if constexpr (is_complex<Out>::value) {
            const auto equal = [](const Out lhs, const Out rhs) {
                return std::abs(lhs - rhs) <= 2e-5f * (1.0f + std::abs(lhs));
            };
            EXPECT_TRUE(compare_volk_arrays<Out>(
                expected, output, equal, describe_volk_difference<Out>));
        } else {
            const auto equal = [](const Out lhs, const Out rhs) {
                return std::abs(lhs - rhs) <= Out(2e-5) * (Out(1) + std::abs(lhs));
            };
            EXPECT_TRUE(compare_volk_arrays<Out>(
                expected, output, equal, describe_volk_difference<Out>));
        }
    }
    volk::vector<In> input_storage;
    volk::vector<Out> output_storage, expected;
    std::span<In> input;
    std::span<Out> output;
};
class ScaledComplex16FloatTest : public ScaledComplexTestBase<lv_16sc_t, float>
{
};
TEST_P(ScaledComplex16FloatTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ScaledComplex16FloatTest,
    testing::ValuesIn(kernel_cases(std::vector<ScaledComplexCase<lv_16sc_t, float>>{
        { "volk_16ic_s32f_deinterleave_real_32f_positive",
          volk_16ic_s32f_deinterleave_real_32f_manual,
          [](const lv_16sc_t value, const float scalar) -> float {
              return static_cast<float>(value.real()) * (1.0f / scalar);
          },
          generate_complex16_input,
          2.0f,
          volk_16ic_s32f_deinterleave_real_32f_get_func_desc() },
        { "volk_16ic_s32f_deinterleave_real_32f_negative",
          volk_16ic_s32f_deinterleave_real_32f_manual,
          [](const lv_16sc_t value, const float scalar) -> float {
              return static_cast<float>(value.real()) * (1.0f / scalar);
          },
          generate_complex16_input,
          -2.0f,
          volk_16ic_s32f_deinterleave_real_32f_get_func_desc() },
        { "volk_16ic_s32f_magnitude_32f_positive",
          volk_16ic_s32f_magnitude_32f_manual,
          [](const lv_16sc_t value, const float scalar) -> float {
              return std::hypot(static_cast<float>(value.real()) / scalar,
                                static_cast<float>(value.imag()) / scalar);
          },
          generate_complex16_input,
          2.0f,
          volk_16ic_s32f_magnitude_32f_get_func_desc() },
        { "volk_16ic_s32f_magnitude_32f_negative",
          volk_16ic_s32f_magnitude_32f_manual,
          [](const lv_16sc_t value, const float scalar) -> float {
              return std::hypot(static_cast<float>(value.real()) / scalar,
                                static_cast<float>(value.imag()) / scalar);
          },
          generate_complex16_input,
          -2.0f,
          volk_16ic_s32f_magnitude_32f_get_func_desc() } })),
    ElementwiseName());
class ScaledComplex8FloatTest : public ScaledComplexTestBase<lv_8sc_t, float>
{
};
TEST_P(ScaledComplex8FloatTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ScaledComplex8FloatTest,
    testing::ValuesIn(kernel_cases(std::vector<ScaledComplexCase<lv_8sc_t, float>>{
        { "volk_8ic_s32f_deinterleave_real_32f_positive",
          volk_8ic_s32f_deinterleave_real_32f_manual,
          [](const lv_8sc_t value, const float scalar) -> float {
              return static_cast<float>(value.real()) * (1.0f / scalar);
          },
          generate_complex8_input,
          2.0f,
          volk_8ic_s32f_deinterleave_real_32f_get_func_desc() },
        { "volk_8ic_s32f_deinterleave_real_32f_negative",
          volk_8ic_s32f_deinterleave_real_32f_manual,
          [](const lv_8sc_t value, const float scalar) -> float {
              return static_cast<float>(value.real()) * (1.0f / scalar);
          },
          generate_complex8_input,
          -2.0f,
          volk_8ic_s32f_deinterleave_real_32f_get_func_desc() } })),
    ElementwiseName());
class ScaledComplex32FloatTest : public ScaledComplexTestBase<lv_32fc_t, float>
{
};
TEST_P(ScaledComplex32FloatTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ScaledComplex32FloatTest,
    testing::ValuesIn(kernel_cases(std::vector<ScaledComplexCase<lv_32fc_t, float>>{
        { "volk_32fc_s32f_atan2_32f_positive",
          volk_32fc_s32f_atan2_32f_manual,
          [](const lv_32fc_t value, const float scalar) -> float {
              return std::atan2(value.imag(), value.real()) / scalar;
          },
          generate_complex_float_input,
          2.0f,
          volk_32fc_s32f_atan2_32f_get_func_desc() },
        { "volk_32fc_s32f_atan2_32f_negative",
          volk_32fc_s32f_atan2_32f_manual,
          [](const lv_32fc_t value, const float scalar) -> float {
              return std::atan2(value.imag(), value.real()) / scalar;
          },
          generate_complex_float_input,
          -2.0f,
          volk_32fc_s32f_atan2_32f_get_func_desc() } })),
    ElementwiseName());
class ScaledComplex32IntegerTest : public ScaledComplexTestBase<lv_32fc_t, int16_t>
{
};
TEST_P(ScaledComplex32IntegerTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ScaledComplex32IntegerTest,
    testing::ValuesIn(kernel_cases(std::vector<ScaledComplexCase<lv_32fc_t, int16_t>>{
        { "volk_32fc_s32f_deinterleave_real_16i_positive",
          volk_32fc_s32f_deinterleave_real_16i_manual,
          [](const lv_32fc_t value, const float scalar) -> int16_t {
              return static_cast<int16_t>(std::rint(value.real() * scalar));
          },
          generate_complex_float_input,
          2.0f,
          volk_32fc_s32f_deinterleave_real_16i_get_func_desc() },
        { "volk_32fc_s32f_deinterleave_real_16i_negative",
          volk_32fc_s32f_deinterleave_real_16i_manual,
          [](const lv_32fc_t value, const float scalar) -> int16_t {
              return static_cast<int16_t>(std::rint(value.real() * scalar));
          },
          generate_complex_float_input,
          -2.0f,
          volk_32fc_s32f_deinterleave_real_16i_get_func_desc() },
        { "volk_32fc_s32f_magnitude_16i_positive",
          volk_32fc_s32f_magnitude_16i_manual,
          [](const lv_32fc_t value, const float scalar) -> int16_t {
              return static_cast<int16_t>(
                  std::rint(scalar * std::sqrt(value.real() * value.real() +
                                               value.imag() * value.imag())));
          },
          generate_complex_float_input,
          2.0f,
          volk_32fc_s32f_magnitude_16i_get_func_desc() },
        { "volk_32fc_s32f_magnitude_16i_negative",
          volk_32fc_s32f_magnitude_16i_manual,
          [](const lv_32fc_t value, const float scalar) -> int16_t {
              return static_cast<int16_t>(
                  std::rint(scalar * std::sqrt(value.real() * value.real() +
                                               value.imag() * value.imag())));
          },
          generate_complex_float_input,
          -2.0f,
          volk_32fc_s32f_magnitude_16i_get_func_desc() } })),
    ElementwiseName());

class ComplexPowerTest : public ScaledComplexTestBase<lv_32fc_t, lv_32fc_t>
{
};
TEST_P(ComplexPowerTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexPowerTest,
    testing::ValuesIn(kernel_cases(std::vector<ScaledComplexCase<lv_32fc_t, lv_32fc_t>>{
        { "volk_32fc_s32f_power_32fc_half",
          volk_32fc_s32f_power_32fc_manual,
          [](const lv_32fc_t value, const float scalar) {
              return std::pow(value, scalar);
          },
          generate_complex_float_input,
          0.5f,
          volk_32fc_s32f_power_32fc_get_func_desc() },
        { "volk_32fc_s32f_power_32fc_one",
          volk_32fc_s32f_power_32fc_manual,
          [](const lv_32fc_t value, const float scalar) {
              return std::pow(value, scalar);
          },
          generate_complex_float_input,
          1.0f,
          volk_32fc_s32f_power_32fc_get_func_desc() },
        { "volk_32fc_s32f_power_32fc_two",
          volk_32fc_s32f_power_32fc_manual,
          [](const lv_32fc_t value, const float scalar) {
              return std::pow(value, scalar);
          },
          generate_complex_float_input,
          2.0f,
          volk_32fc_s32f_power_32fc_get_func_desc() } })),
    ElementwiseName());

class ComplexSpectrumTest : public ScaledComplexTestBase<lv_32fc_t, float>
{
};
TEST_P(ComplexSpectrumTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexSpectrumTest,
    testing::ValuesIn(kernel_cases(std::vector<ScaledComplexCase<lv_32fc_t, float>>{
        { "volk_32fc_s32f_power_spectrum_32f",
          volk_32fc_s32f_power_spectrum_32f_manual,
          [](const lv_32fc_t value, const float scalar) {
              return 10.0f * std::log10(std::norm(value) / (scalar * scalar));
          },
          generate_spectrum_input,
          2.0f,
          volk_32fc_s32f_power_spectrum_32f_get_func_desc() } })),
    ElementwiseName());


struct ComplexPointerScalarCase {
    std::string name;
    lv_32fc_t scalar;
    volk_func_desc_t description;
};
std::vector<ComplexPointerScalarCase> complex_pointer_scalar_cases()
{
    const auto descriptor = volk_32fc_s32fc_multiply2_32fc_get_func_desc();
    return { { "volk_32fc_s32fc_multiply2_32fc_zero", { 0.0f, 0.0f }, descriptor },
             { "volk_32fc_s32fc_multiply2_32fc_complex", { 0.5f, -2.0f }, descriptor },
             { "volk_32fc_s32fc_multiply2_32fc_unit", { 1.0f, 0.0f }, descriptor } };
}
class ComplexPointerScalarTest
    : public VolkTestImpl<std::tuple<std::string, size_t, ComplexPointerScalarCase>>
{
protected:
    void initialize_data() override
    {
        input_storage.resize(vector_length + offset);
        output_storage.resize(vector_length + offset);
        expected.resize(vector_length);
        input = std::span(input_storage).subspan(offset, vector_length);
        output = std::span(output_storage).subspan(offset, vector_length);
        std::mt19937 engine(0xdeadbeef);
        generate_complex_float_input(input, engine);
        const lv_32fc_t scalar = std::get<2>(GetParam()).scalar;
        std::transform(input.begin(),
                       input.end(),
                       expected.begin(),
                       [scalar](const lv_32fc_t value) { return value * scalar; });
    }
    void run()
    {
        const auto& kernel = std::get<2>(GetParam());
        volk_32fc_s32fc_multiply2_32fc_manual(output.data(),
                                              input.data(),
                                              &kernel.scalar,
                                              static_cast<unsigned int>(vector_length),
                                              implementation_name.c_str());
        const auto equal = [](const lv_32fc_t lhs, const lv_32fc_t rhs) {
            return std::abs(lhs - rhs) <= 2e-5f * (1.0f + std::abs(lhs));
        };
        EXPECT_TRUE(compare_volk_arrays<lv_32fc_t>(
            expected, output, equal, describe_volk_difference<lv_32fc_t>));
    }
    volk::vector<lv_32fc_t> input_storage, output_storage, expected;
    std::span<lv_32fc_t> input, output;
};
TEST_P(ComplexPointerScalarTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         ComplexPointerScalarTest,
                         testing::ValuesIn(kernel_cases(complex_pointer_scalar_cases())),
                         ElementwiseName());

} // namespace
