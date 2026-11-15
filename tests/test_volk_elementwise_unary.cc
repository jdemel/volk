/* -*- c++ -*- */
/*
 * Copyright 2026 Johannes Sterz Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_test_elementwise.h"
#include <type_traits>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {
struct UnaryCase {
    std::string name;
    void (*kernel)(float*, const float*, unsigned int, const char*);
    std::function<float(float)> prepare;
    std::function<float(float)> reference;
    float standard_deviation;
    float tolerance;
    float scale_floor;
    volk_func_desc_t description;
};

std::vector<UnaryCase> unary_kernels()
{
    const auto identity = [](const float value) { return value; };
    const auto positive = [](const float value) {
        return std::max(std::abs(value), 1.0f);
    };
    return {
        { "volk_32f_null_32f",
          volk_32f_null_32f_manual,
          identity,
          identity,
          1000000.0f,
          0.0f,
          1.0f,
          volk_32f_null_32f_get_func_desc() },
        { "volk_32f_sqrt_32f",
          volk_32f_sqrt_32f_manual,
          positive,
          [](const float x) { return std::sqrt(x); },
          1000000.0f,
          2e-6f,
          1.0f,
          volk_32f_sqrt_32f_get_func_desc() },
        { "volk_32f_invsqrt_32f",
          volk_32f_invsqrt_32f_manual,
          positive,
          [](const float x) { return 1.0f / std::sqrt(x); },
          1000000.0f,
          1e-5f,
          1e-6f,
          volk_32f_invsqrt_32f_get_func_desc() },
        { "volk_32f_reciprocal_32f",
          volk_32f_reciprocal_32f_manual,
          [](const float x) { return std::abs(x) < 1.0f ? std::copysign(1.0f, x) : x; },
          [](const float x) { return 1.0f / x; },
          1000000.0f,
          6.2e-5f,
          1e-6f,
          volk_32f_reciprocal_32f_get_func_desc() },
        { "volk_32f_sin_32f",
          volk_32f_sin_32f_manual,
          identity,
          [](const float x) { return std::sin(x); },
          100.0f,
          2e-5f,
          1.0f,
          volk_32f_sin_32f_get_func_desc() },
        { "volk_32f_cos_32f",
          volk_32f_cos_32f_manual,
          identity,
          [](const float x) { return std::cos(x); },
          100.0f,
          2e-5f,
          1.0f,
          volk_32f_cos_32f_get_func_desc() },
        { "volk_32f_atan_32f",
          volk_32f_atan_32f_manual,
          identity,
          [](const float x) { return std::atan(x); },
          1000000.0f,
          2e-5f,
          1.0f,
          volk_32f_atan_32f_get_func_desc() },
        { "volk_32f_acos_32f",
          volk_32f_acos_32f_manual,
          [](const float value) { return std::clamp(value, -1.0f, 1.0f); },
          [](const float value) { return std::acos(value); },
          0.5f,
          2e-5f,
          1.0f,
          volk_32f_acos_32f_get_func_desc() },
        { "volk_32f_asin_32f",
          volk_32f_asin_32f_manual,
          [](const float value) { return std::clamp(value, -1.0f, 1.0f); },
          [](const float value) { return std::asin(value); },
          0.5f,
          2e-5f,
          1.0f,
          volk_32f_asin_32f_get_func_desc() },
        { "volk_32f_exp_32f",
          volk_32f_exp_32f_manual,
          [](const float value) { return std::clamp(value, -50.0f, 50.0f); },
          [](const float value) { return std::exp(value); },
          8.0f,
          2e-5f,
          1e-20f,
          volk_32f_exp_32f_get_func_desc() },
        { "volk_32f_expfast_32f",
          volk_32f_expfast_32f_manual,
          [](const float value) { return std::clamp(value, -50.0f, 50.0f); },
          [](const float value) { return std::exp(value); },
          8.0f,
          0.07f,
          1e-20f,
          volk_32f_expfast_32f_get_func_desc() },
        { "volk_32f_tan_32f",
          volk_32f_tan_32f_manual,
          [](const float value) { return std::fmod(value, 1.4f); },
          [](const float value) { return std::tan(value); },
          100.0f,
          2e-5f,
          1.0f,
          volk_32f_tan_32f_get_func_desc() },
        { "volk_32f_tanh_32f",
          volk_32f_tanh_32f_manual,
          identity,
          [](const float x) { return std::tanh(x); },
          4.0f,
          2e-5f,
          1.0f,
          volk_32f_tanh_32f_get_func_desc() },
    };
}

class UnaryFloatTest : public VolkTestImpl<std::tuple<std::string, size_t, UnaryCase>>
{
protected:
    void initialize_data() override
    {
        input_storage.resize(vector_length + offset);
        output_storage.resize(vector_length + offset);
        expected.resize(vector_length);
        input = std::span(input_storage).subspan(offset, vector_length);
        output = std::span(output_storage).subspan(offset, vector_length);
        const auto& kernel = std::get<2>(GetParam());
        std::mt19937 engine(0xdeadbeef);
        std::normal_distribution<float> distribution(0.0f, kernel.standard_deviation);
        generate_normal_input(input, engine, distribution);
        std::transform(input.begin(), input.end(), input.begin(), kernel.prepare);
        std::transform(input.begin(), input.end(), expected.begin(), kernel.reference);
    }
    void run()
    {
        const auto& kernel = std::get<2>(GetParam());
        kernel.kernel(output.data(),
                      input.data(),
                      static_cast<unsigned int>(vector_length),
                      implementation_name.c_str());
        const auto equal = [&kernel](const float lhs, const float rhs) {
            return std::abs(lhs - rhs) <=
                   kernel.tolerance * std::max(kernel.scale_floor, std::abs(lhs));
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

TEST_P(UnaryFloatTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         UnaryFloatTest,
                         testing::ValuesIn(kernel_cases(unary_kernels())),
                         ElementwiseName());


template <typename In, typename Out>
struct UnaryIntegerCase {
    std::string name;
    void (*kernel)(Out*, const In*, unsigned int, const char*);
    Out (*reference)(In);
    void (*generate)(std::span<In>, std::mt19937&);
    volk_func_desc_t description;
};

template <typename In, typename Out>
class UnaryIntegerTestBase
    : public VolkTestImpl<std::tuple<std::string, size_t, UnaryIntegerCase<In, Out>>>
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
        std::transform(input.begin(), input.end(), expected.begin(), kernel.reference);
    }
    void run()
    {
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.kernel(output.data(),
                      input.data(),
                      static_cast<unsigned int>(this->vector_length),
                      this->implementation_name.c_str());
        EXPECT_TRUE(AreIntegerArraysEqual<Out>(expected, output));
    }
    volk::vector<In> input_storage;
    volk::vector<Out> output_storage, expected;
    std::span<In> input;
    std::span<Out> output;
};
class Slicer32Test : public UnaryIntegerTestBase<float, int32_t>
{
};
class Slicer8Test : public UnaryIntegerTestBase<float, int8_t>
{
};
class Reverse32Test : public UnaryIntegerTestBase<uint32_t, uint32_t>
{
};
TEST_P(Slicer32Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Slicer32Test,
    testing::ValuesIn(kernel_cases(std::vector<UnaryIntegerCase<float, int32_t>>{
        { "volk_32f_binary_slicer_32i",
          volk_32f_binary_slicer_32i_manual,
          [](const float value) -> int32_t { return value >= 0.0f ? 1 : 0; },
          generate_slicer_input,
          volk_32f_binary_slicer_32i_get_func_desc() } })),
    ElementwiseName());
TEST_P(Slicer8Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Slicer8Test,
    testing::ValuesIn(kernel_cases(std::vector<UnaryIntegerCase<float, int8_t>>{
        { "volk_32f_binary_slicer_8i",
          volk_32f_binary_slicer_8i_manual,
          [](const float value) -> int8_t { return value >= 0.0f ? 1 : 0; },
          generate_slicer_input,
          volk_32f_binary_slicer_8i_get_func_desc() } })),
    ElementwiseName());
TEST_P(Reverse32Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Reverse32Test,
    testing::ValuesIn(kernel_cases(std::vector<UnaryIntegerCase<uint32_t, uint32_t>>{
        { "volk_32u_reverse_32u",
          volk_32u_reverse_32u_manual,
          [](uint32_t value) -> uint32_t {
              uint32_t result = 0;
              for (unsigned int bit = 0; bit < 32; ++bit) {
                  result = (result << 1) | (value & 1u);
                  value >>= 1;
              }
              return result;
          },
          generate_byteswap_input<uint32_t>,
          volk_32u_reverse_32u_get_func_desc() } })),
    ElementwiseName());


template <typename In, typename Out>
struct ComplexUnaryCase {
    std::string name;
    void (*kernel)(Out*, const In*, unsigned int, const char*);
    Out (*reference)(In);
    void (*generate)(std::span<In>, std::mt19937&);
    volk_func_desc_t description;
};

template <typename In, typename Out>
class ComplexUnaryTestBase
    : public VolkTestImpl<std::tuple<std::string, size_t, ComplexUnaryCase<In, Out>>>
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
        std::transform(input.begin(), input.end(), expected.begin(), kernel.reference);
    }
    void run()
    {
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.kernel(output.data(),
                      input.data(),
                      static_cast<unsigned int>(this->vector_length),
                      this->implementation_name.c_str());
        if constexpr (std::is_integral_v<Out> || is_complex<Out>::value) {
            EXPECT_TRUE(AreIntegerArraysEqual<Out>(expected, output));
        } else {
            const auto equal = [](const Out lhs, const Out rhs) {
                return std::abs(lhs - rhs) <= Out(2e-6) * (Out(1) + std::abs(lhs));
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
class Complex8Real8Test : public ComplexUnaryTestBase<lv_8sc_t, int8_t>
{
};
TEST_P(Complex8Real8Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Complex8Real8Test,
    testing::ValuesIn(kernel_cases(std::vector<ComplexUnaryCase<lv_8sc_t, int8_t>>{
        { "volk_8ic_deinterleave_real_8i",
          volk_8ic_deinterleave_real_8i_manual,
          [](const lv_8sc_t value) -> int8_t { return value.real(); },
          generate_complex8_input,
          volk_8ic_deinterleave_real_8i_get_func_desc() } })),
    ElementwiseName());
class Complex8Real16Test : public ComplexUnaryTestBase<lv_8sc_t, int16_t>
{
};
TEST_P(Complex8Real16Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Complex8Real16Test,
    testing::ValuesIn(kernel_cases(std::vector<ComplexUnaryCase<lv_8sc_t, int16_t>>{
        { "volk_8ic_deinterleave_real_16i",
          volk_8ic_deinterleave_real_16i_manual,
          [](const lv_8sc_t value) -> int16_t {
              return static_cast<int16_t>(static_cast<int>(value.real()) * 128);
          },
          generate_complex8_input,
          volk_8ic_deinterleave_real_16i_get_func_desc() } })),
    ElementwiseName());
class Complex16Real16Test : public ComplexUnaryTestBase<lv_16sc_t, int16_t>
{
};
TEST_P(Complex16Real16Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Complex16Real16Test,
    testing::ValuesIn(kernel_cases(std::vector<ComplexUnaryCase<lv_16sc_t, int16_t>>{
        { "volk_16ic_deinterleave_real_16i",
          volk_16ic_deinterleave_real_16i_manual,
          [](const lv_16sc_t value) -> int16_t { return value.real(); },
          generate_complex16_input,
          volk_16ic_deinterleave_real_16i_get_func_desc() } })),
    ElementwiseName());
class Complex16Real8Test : public ComplexUnaryTestBase<lv_16sc_t, int8_t>
{
};
TEST_P(Complex16Real8Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Complex16Real8Test,
    testing::ValuesIn(kernel_cases(std::vector<ComplexUnaryCase<lv_16sc_t, int8_t>>{
        { "volk_16ic_deinterleave_real_8i",
          volk_16ic_deinterleave_real_8i_manual,
          [](const lv_16sc_t value) -> int8_t {
              return static_cast<int8_t>(value.real() >> 8);
          },
          generate_complex16_input,
          volk_16ic_deinterleave_real_8i_get_func_desc() } })),
    ElementwiseName());
class ComplexFloatRealTest : public ComplexUnaryTestBase<lv_32fc_t, float>
{
};
TEST_P(ComplexFloatRealTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexFloatRealTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexUnaryCase<lv_32fc_t, float>>{
        { "volk_32fc_deinterleave_real_32f",
          volk_32fc_deinterleave_real_32f_manual,
          [](const lv_32fc_t value) -> float { return value.real(); },
          generate_complex_float_input,
          volk_32fc_deinterleave_real_32f_get_func_desc() },
        { "volk_32fc_deinterleave_imag_32f",
          volk_32fc_deinterleave_imag_32f_manual,
          [](const lv_32fc_t value) -> float { return value.imag(); },
          generate_complex_float_input,
          volk_32fc_deinterleave_imag_32f_get_func_desc() },
        { "volk_32fc_magnitude_32f",
          volk_32fc_magnitude_32f_manual,
          [](const lv_32fc_t value) -> float {
              return std::sqrt(value.real() * value.real() + value.imag() * value.imag());
          },
          generate_complex_float_input,
          volk_32fc_magnitude_32f_get_func_desc() },
        { "volk_32fc_magnitude_squared_32f",
          volk_32fc_magnitude_squared_32f_manual,
          [](const lv_32fc_t value) -> float {
              return value.real() * value.real() + value.imag() * value.imag();
          },
          generate_complex_float_input,
          volk_32fc_magnitude_squared_32f_get_func_desc() } })),
    ElementwiseName());
class ComplexDoubleRealTest : public ComplexUnaryTestBase<lv_32fc_t, double>
{
};
TEST_P(ComplexDoubleRealTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexDoubleRealTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexUnaryCase<lv_32fc_t, double>>{
        { "volk_32fc_deinterleave_real_64f",
          volk_32fc_deinterleave_real_64f_manual,
          [](const lv_32fc_t value) -> double {
              return static_cast<double>(value.real());
          },
          generate_complex_float_input,
          volk_32fc_deinterleave_real_64f_get_func_desc() } })),
    ElementwiseName());
class ComplexConjugateTest : public ComplexUnaryTestBase<lv_32fc_t, lv_32fc_t>
{
};
TEST_P(ComplexConjugateTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexConjugateTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexUnaryCase<lv_32fc_t, lv_32fc_t>>{
        { "volk_32fc_conjugate_32fc",
          volk_32fc_conjugate_32fc_manual,
          [](const lv_32fc_t value) -> lv_32fc_t { return std::conj(value); },
          generate_complex_float_input,
          volk_32fc_conjugate_32fc_get_func_desc() } })),
    ElementwiseName());
class Complex16MagnitudeTest : public ComplexUnaryTestBase<lv_16sc_t, int16_t>
{
};
TEST_P(Complex16MagnitudeTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    Complex16MagnitudeTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexUnaryCase<lv_16sc_t, int16_t>>{
        { "volk_16ic_magnitude_16i",
          volk_16ic_magnitude_16i_manual,
          [](const lv_16sc_t value) -> int16_t {
              const float real = static_cast<float>(value.real()) / 32767.0f;
              const float imag = static_cast<float>(value.imag()) / 32767.0f;
              return static_cast<int16_t>(
                  std::rint(std::sqrt(real * real + imag * imag) * 32767.0f));
          },
          generate_complex16_magnitude_input,
          volk_16ic_magnitude_16i_get_func_desc() } })),
    ElementwiseName());

} // namespace
