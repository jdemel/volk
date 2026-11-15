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
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {
template <typename T>
struct BinaryCase {
    std::string name;
    void (*kernel)(T*, const T*, const T*, unsigned int, const char*);
    std::function<T(T, T)> reference;
    volk_func_desc_t description;
    void (*generate)(std::span<T>, std::span<T>, std::mt19937&);
};

std::vector<BinaryCase<float>> float_kernels()
{
    return {
        { "volk_32f_x2_add_32f",
          volk_32f_x2_add_32f_manual,
          [](const float vec0, const float vec1) { return vec0 + vec1; },
          volk_32f_x2_add_32f_get_func_desc(),
          generate_binary_input<float> },
        { "volk_32f_x2_subtract_32f",
          volk_32f_x2_subtract_32f_manual,
          [](const float vec0, const float vec1) { return vec0 - vec1; },
          volk_32f_x2_subtract_32f_get_func_desc(),
          generate_binary_input<float> },
        { "volk_32f_x2_multiply_32f",
          volk_32f_x2_multiply_32f_manual,
          [](const float vec0, const float vec1) { return vec0 * vec1; },
          volk_32f_x2_multiply_32f_get_func_desc(),
          generate_binary_input<float> },
        { "volk_32f_x2_divide_32f",
          volk_32f_x2_divide_32f_manual,
          [](const float vec0, const float vec1) { return vec0 / vec1; },
          volk_32f_x2_divide_32f_get_func_desc(),
          generate_division_input<float> },
        { "volk_32f_x2_pow_32f",
          volk_32f_x2_pow_32f_manual,
          [](const float exponent, const float base) { return std::pow(base, exponent); },
          volk_32f_x2_pow_32f_get_func_desc(),
          generate_binary_power_input },
        { "volk_32f_x2_min_32f",
          volk_32f_x2_min_32f_manual,
          [](const float vec0, const float vec1) { return std::min(vec0, vec1); },
          volk_32f_x2_min_32f_get_func_desc(),
          generate_binary_input<float> },
        { "volk_32f_x2_max_32f",
          volk_32f_x2_max_32f_manual,
          [](const float vec0, const float vec1) { return std::max(vec0, vec1); },
          volk_32f_x2_max_32f_get_func_desc(),
          generate_binary_input<float> },
    };
}

std::vector<BinaryCase<double>> double_kernels()
{
    return {
        { "volk_64f_x2_add_64f",
          volk_64f_x2_add_64f_manual,
          [](const double vec0, const double vec1) { return vec0 + vec1; },
          volk_64f_x2_add_64f_get_func_desc(),
          generate_binary_input<double> },
        { "volk_64f_x2_multiply_64f",
          volk_64f_x2_multiply_64f_manual,
          [](const double vec0, const double vec1) { return vec0 * vec1; },
          volk_64f_x2_multiply_64f_get_func_desc(),
          generate_binary_input<double> },
        { "volk_64f_x2_min_64f",
          volk_64f_x2_min_64f_manual,
          [](const double vec0, const double vec1) { return std::min(vec0, vec1); },
          volk_64f_x2_min_64f_get_func_desc(),
          generate_binary_input<double> },
        { "volk_64f_x2_max_64f",
          volk_64f_x2_max_64f_manual,
          [](const double vec0, const double vec1) { return std::max(vec0, vec1); },
          volk_64f_x2_max_64f_get_func_desc(),
          generate_binary_input<double> },
    };
}

template <typename T>
T saturated_add(const T vec0, const T vec1)
{
    const int sum = static_cast<int>(vec0) + static_cast<int>(vec1);
    return static_cast<T>(std::clamp(sum,
                                     static_cast<int>(std::numeric_limits<T>::min()),
                                     static_cast<int>(std::numeric_limits<T>::max())));
}

template <typename T>
std::vector<BinaryCase<T>> integer_kernels();

template <>
std::vector<BinaryCase<int8_t>> integer_kernels<int8_t>()
{
    return { { "volk_8i_x2_add_saturated_8i",
               volk_8i_x2_add_saturated_8i_manual,
               [](const int8_t vec0, const int8_t vec1) {
                   return saturated_add(vec0, vec1);
               },
               volk_8i_x2_add_saturated_8i_get_func_desc(),
               generate_saturated_signed_input<int8_t> } };
}
template <>
std::vector<BinaryCase<uint8_t>> integer_kernels<uint8_t>()
{
    return { { "volk_8u_x2_add_saturated_8u",
               volk_8u_x2_add_saturated_8u_manual,
               [](const uint8_t vec0, const uint8_t vec1) {
                   return saturated_add(vec0, vec1);
               },
               volk_8u_x2_add_saturated_8u_get_func_desc(),
               generate_integer_binary_input<uint8_t> } };
}
template <>
std::vector<BinaryCase<int16_t>> integer_kernels<int16_t>()
{
    return { { "volk_16i_x2_add_saturated_16i",
               volk_16i_x2_add_saturated_16i_manual,
               [](const int16_t vec0, const int16_t vec1) {
                   return saturated_add(vec0, vec1);
               },
               volk_16i_x2_add_saturated_16i_get_func_desc(),
               generate_saturated_signed_input<int16_t> } };
}
template <>
std::vector<BinaryCase<uint16_t>> integer_kernels<uint16_t>()
{
    return { { "volk_16u_x2_add_saturated_16u",
               volk_16u_x2_add_saturated_16u_manual,
               [](const uint16_t vec0, const uint16_t vec1) {
                   return saturated_add(vec0, vec1);
               },
               volk_16u_x2_add_saturated_16u_get_func_desc(),
               generate_integer_binary_input<uint16_t> } };
}
template <>
std::vector<BinaryCase<int32_t>> integer_kernels<int32_t>()
{
    return {
        { "volk_32i_x2_and_32i",
          volk_32i_x2_and_32i_manual,
          [](const int32_t vec0, const int32_t vec1) { return vec0 & vec1; },
          volk_32i_x2_and_32i_get_func_desc(),
          generate_bitwise_input },
        { "volk_32i_x2_or_32i",
          volk_32i_x2_or_32i_manual,
          [](const int32_t vec0, const int32_t vec1) { return vec0 | vec1; },
          volk_32i_x2_or_32i_get_func_desc(),
          generate_bitwise_input },
    };
}

template <typename T>
class BinaryTestBase : public VolkTestImpl<std::tuple<std::string, size_t, BinaryCase<T>>>
{
protected:
    void initialize_data() override
    {
        vec0_storage.resize(this->vector_length + this->offset);
        vec1_storage.resize(this->vector_length + this->offset);
        output_storage.resize(this->vector_length + this->offset);
        expected.resize(this->vector_length);
        vec0 = std::span(vec0_storage).subspan(this->offset, this->vector_length);
        vec1 = std::span(vec1_storage).subspan(this->offset, this->vector_length);
        output = std::span(output_storage).subspan(this->offset, this->vector_length);
        std::mt19937 engine(0xdeadbeef);
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.generate(vec0, vec1, engine);
        std::transform(
            vec0.begin(), vec0.end(), vec1.begin(), expected.begin(), kernel.reference);
    }

    void run()
    {
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.kernel(output.data(),
                      vec0.data(),
                      vec1.data(),
                      static_cast<unsigned int>(this->vector_length),
                      this->implementation_name.c_str());
        if constexpr (std::is_integral_v<T>) {
            EXPECT_TRUE(AreIntegerArraysEqual<T>(expected, output));
        } else {
            const auto equal = [](const T lhs, const T rhs) {
                const T tolerance = std::is_same_v<T, float> ? T(2e-6) : T(2e-12);
                return std::abs(lhs - rhs) <= tolerance * (T(1) + std::abs(lhs));
            };
            EXPECT_TRUE(compare_volk_arrays<T>(
                expected, output, equal, describe_volk_difference<T>));
        }
    }

    volk::vector<T> vec0_storage;
    volk::vector<T> vec1_storage;
    volk::vector<T> output_storage;
    volk::vector<T> expected;
    std::span<T> vec0;
    std::span<T> vec1;
    std::span<T> output;
};

class BinaryFloatTest : public BinaryTestBase<float>
{
};
class BinaryDoubleTest : public BinaryTestBase<double>
{
};

class BinaryInt8Test : public BinaryTestBase<int8_t>
{
};
class BinaryUInt8Test : public BinaryTestBase<uint8_t>
{
};
class BinaryInt16Test : public BinaryTestBase<int16_t>
{
};
class BinaryUInt16Test : public BinaryTestBase<uint16_t>
{
};
class BinaryInt32Test : public BinaryTestBase<int32_t>
{
};

TEST_P(BinaryFloatTest, run)
{
    report_runtime([this] { run(); });
}

INSTANTIATE_TEST_SUITE_P(Elementwise,
                         BinaryFloatTest,
                         testing::ValuesIn(kernel_cases(float_kernels())),
                         ElementwiseName());
TEST_P(BinaryDoubleTest, run)
{
    report_runtime([this] { run(); });
}

INSTANTIATE_TEST_SUITE_P(Elementwise,
                         BinaryDoubleTest,
                         testing::ValuesIn(kernel_cases(double_kernels())),
                         ElementwiseName());
TEST_P(BinaryInt8Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         BinaryInt8Test,
                         testing::ValuesIn(kernel_cases(integer_kernels<int8_t>())),
                         ElementwiseName());

TEST_P(BinaryUInt8Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         BinaryUInt8Test,
                         testing::ValuesIn(kernel_cases(integer_kernels<uint8_t>())),
                         ElementwiseName());

TEST_P(BinaryInt16Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         BinaryInt16Test,
                         testing::ValuesIn(kernel_cases(integer_kernels<int16_t>())),
                         ElementwiseName());

TEST_P(BinaryUInt16Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         BinaryUInt16Test,
                         testing::ValuesIn(kernel_cases(integer_kernels<uint16_t>())),
                         ElementwiseName());

TEST_P(BinaryInt32Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         BinaryInt32Test,
                         testing::ValuesIn(kernel_cases(integer_kernels<int32_t>())),
                         ElementwiseName());


struct MixedCase {
    std::string name;
    void (*kernel)(double*, const float*, const double*, unsigned int, const char*);
    std::function<double(float, double)> reference;
    volk_func_desc_t description;
    void (*generate)(std::span<float>, std::span<double>, std::mt19937&);
};

std::vector<MixedCase> mixed_kernels()
{
    return {
        { "volk_32f_64f_add_64f",
          volk_32f_64f_add_64f_manual,
          [](const float value0, const double value1) {
              return static_cast<double>(value0) + value1;
          },
          volk_32f_64f_add_64f_get_func_desc(),
          generate_mixed_binary_input },
        { "volk_32f_64f_multiply_64f",
          volk_32f_64f_multiply_64f_manual,
          [](const float value0, const double value1) {
              return static_cast<double>(value0) * value1;
          },
          volk_32f_64f_multiply_64f_get_func_desc(),
          generate_mixed_binary_input },
    };
}

class MixedBinaryTest : public VolkTestImpl<std::tuple<std::string, size_t, MixedCase>>
{
protected:
    void initialize_data() override
    {
        vec0_storage.resize(vector_length + offset);
        vec1_storage.resize(vector_length + offset);
        output_storage.resize(vector_length + offset);
        expected.resize(vector_length);
        vec0 = std::span(vec0_storage).subspan(offset, vector_length);
        vec1 = std::span(vec1_storage).subspan(offset, vector_length);
        output = std::span(output_storage).subspan(offset, vector_length);
        std::mt19937 engine(0xdeadbeef);
        const auto& kernel = std::get<2>(GetParam());
        kernel.generate(vec0, vec1, engine);
        std::transform(
            vec0.begin(), vec0.end(), vec1.begin(), expected.begin(), kernel.reference);
    }

    void run()
    {
        const auto& kernel = std::get<2>(GetParam());
        kernel.kernel(output.data(),
                      vec0.data(),
                      vec1.data(),
                      static_cast<unsigned int>(vector_length),
                      implementation_name.c_str());
        const auto equal = [](const double lhs, const double rhs) {
            return std::abs(lhs - rhs) <= 2e-12 * (1.0 + std::abs(lhs));
        };
        EXPECT_TRUE(compare_volk_arrays<double>(
            expected, output, equal, describe_volk_difference<double>));
    }
    volk::vector<float> vec0_storage;
    volk::vector<double> vec1_storage;
    volk::vector<double> output_storage;
    volk::vector<double> expected;
    std::span<float> vec0;
    std::span<double> vec1;
    std::span<double> output;
};

TEST_P(MixedBinaryTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         MixedBinaryTest,
                         testing::ValuesIn(kernel_cases(mixed_kernels())),
                         ElementwiseName());


template <typename In0, typename In1>
struct ComplexBinaryCase {
    std::string name;
    void (*kernel)(lv_32fc_t*, const In0*, const In1*, unsigned int, const char*);
    lv_32fc_t (*reference)(In0, In1);
    void (*generate)(std::span<In0>, std::span<In1>, std::mt19937&);
    volk_func_desc_t description;
};
template <typename In0, typename In1>
class ComplexBinaryTestBase
    : public VolkTestImpl<std::tuple<std::string, size_t, ComplexBinaryCase<In0, In1>>>
{
protected:
    void initialize_data() override
    {
        vec0_storage.resize(this->vector_length + this->offset);
        vec1_storage.resize(this->vector_length + this->offset);
        output_storage.resize(this->vector_length + this->offset);
        expected.resize(this->vector_length);
        vec0 = std::span(vec0_storage).subspan(this->offset, this->vector_length);
        vec1 = std::span(vec1_storage).subspan(this->offset, this->vector_length);
        output = std::span(output_storage).subspan(this->offset, this->vector_length);
        std::mt19937 engine(0xdeadbeef);
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.generate(vec0, vec1, engine);
        std::transform(
            vec0.begin(), vec0.end(), vec1.begin(), expected.begin(), kernel.reference);
    }
    void run()
    {
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.kernel(output.data(),
                      vec0.data(),
                      vec1.data(),
                      static_cast<unsigned int>(this->vector_length),
                      this->implementation_name.c_str());
        const auto equal = [](const lv_32fc_t lhs, const lv_32fc_t rhs) {
            return std::abs(lhs.real() - rhs.real()) <=
                       2e-5f * (1.0f + std::abs(lhs.real())) &&
                   std::abs(lhs.imag() - rhs.imag()) <=
                       2e-5f * (1.0f + std::abs(lhs.imag()));
        };
        EXPECT_TRUE(compare_volk_arrays<lv_32fc_t>(
            expected, output, equal, describe_volk_difference<lv_32fc_t>));
    }
    volk::vector<In0> vec0_storage;
    volk::vector<In1> vec1_storage;
    volk::vector<lv_32fc_t> output_storage, expected;
    std::span<In0> vec0;
    std::span<In1> vec1;
    std::span<lv_32fc_t> output;
};
class ComplexBinaryTest : public ComplexBinaryTestBase<lv_32fc_t, lv_32fc_t>
{
};
class ComplexRealBinaryTest : public ComplexBinaryTestBase<lv_32fc_t, float>
{
};
class InterleaveBinaryTest : public ComplexBinaryTestBase<float, float>
{
};
TEST_P(ComplexBinaryTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexBinaryTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexBinaryCase<lv_32fc_t, lv_32fc_t>>{
        { "volk_32fc_x2_add_32fc",
          volk_32fc_x2_add_32fc_manual,
          [](const lv_32fc_t value0, const lv_32fc_t value1) -> lv_32fc_t {
              return value0 + value1;
          },
          generate_complex_binary_input,
          volk_32fc_x2_add_32fc_get_func_desc() },
        { "volk_32fc_x2_divide_32fc",
          volk_32fc_x2_divide_32fc_manual,
          [](const lv_32fc_t value0, const lv_32fc_t value1) -> lv_32fc_t {
              return value0 / value1;
          },
          generate_complex_division_input,
          volk_32fc_x2_divide_32fc_get_func_desc() },
        { "volk_32fc_x2_multiply_conjugate_32fc",
          volk_32fc_x2_multiply_conjugate_32fc_manual,
          [](const lv_32fc_t value0, const lv_32fc_t value1) -> lv_32fc_t {
              return value0 * std::conj(value1);
          },
          generate_complex_binary_input,
          volk_32fc_x2_multiply_conjugate_32fc_get_func_desc() } })),
    ElementwiseName());
TEST_P(ComplexRealBinaryTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexRealBinaryTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexBinaryCase<lv_32fc_t, float>>{
        { "volk_32fc_32f_add_32fc",
          volk_32fc_32f_add_32fc_manual,
          [](const lv_32fc_t value0, const float value1) -> lv_32fc_t {
              return value0 + value1;
          },
          generate_complex_real_input,
          volk_32fc_32f_add_32fc_get_func_desc() },
        { "volk_32fc_32f_multiply_32fc",
          volk_32fc_32f_multiply_32fc_manual,
          [](const lv_32fc_t value0, const float value1) -> lv_32fc_t {
              return value0 * value1;
          },
          generate_complex_real_input,
          volk_32fc_32f_multiply_32fc_get_func_desc() } })),
    ElementwiseName());
TEST_P(InterleaveBinaryTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    InterleaveBinaryTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexBinaryCase<float, float>>{
        { "volk_32f_x2_interleave_32fc",
          volk_32f_x2_interleave_32fc_manual,
          [](const float value0, const float value1) -> lv_32fc_t {
              return lv_32fc_t(value0, value1);
          },
          generate_binary_input<float>,
          volk_32f_x2_interleave_32fc_get_func_desc() } })),
    ElementwiseName());

template <typename In>
struct ComplexIntegerBinaryCase {
    std::string name;
    void (*kernel)(lv_16sc_t*, const In*, const In*, unsigned int, const char*);
    lv_16sc_t (*reference)(In, In);
    void (*generate)(std::span<In>, std::mt19937&);
    volk_func_desc_t description;
};
template <typename In>
class ComplexIntegerBinaryTestBase
    : public VolkTestImpl<std::tuple<std::string, size_t, ComplexIntegerBinaryCase<In>>>
{
protected:
    void initialize_data() override
    {
        vec0_storage.resize(this->vector_length + this->offset);
        vec1_storage.resize(this->vector_length + this->offset);
        output_storage.resize(this->vector_length + this->offset);
        expected.resize(this->vector_length);
        vec0 = std::span(vec0_storage).subspan(this->offset, this->vector_length);
        vec1 = std::span(vec1_storage).subspan(this->offset, this->vector_length);
        output = std::span(output_storage).subspan(this->offset, this->vector_length);
        const auto& kernel = std::get<2>(this->GetParam());
        std::mt19937 engine(0xdeadbeef);
        kernel.generate(vec0, engine);
        kernel.generate(vec1, engine);
        std::transform(
            vec0.begin(), vec0.end(), vec1.begin(), expected.begin(), kernel.reference);
    }
    void run()
    {
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.kernel(output.data(),
                      vec0.data(),
                      vec1.data(),
                      static_cast<unsigned int>(this->vector_length),
                      this->implementation_name.c_str());
        EXPECT_TRUE(AreIntegerArraysEqual<lv_16sc_t>(expected, output));
    }
    volk::vector<In> vec0_storage, vec1_storage;
    volk::vector<lv_16sc_t> output_storage, expected;
    std::span<In> vec0, vec1;
    std::span<lv_16sc_t> output;
};
class ComplexInteger16BinaryTest : public ComplexIntegerBinaryTestBase<lv_16sc_t>
{
};
class ComplexInteger8BinaryTest : public ComplexIntegerBinaryTestBase<lv_8sc_t>
{
};
TEST_P(ComplexInteger16BinaryTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexInteger16BinaryTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexIntegerBinaryCase<lv_16sc_t>>{
        { "volk_16ic_x2_multiply_16ic",
          volk_16ic_x2_multiply_16ic_manual,
          [](const lv_16sc_t value0, const lv_16sc_t value1) -> lv_16sc_t {
              const int64_t real = int64_t(value0.real()) * value1.real() -
                                   int64_t(value0.imag()) * value1.imag();
              const int64_t imag = int64_t(value0.real()) * value1.imag() +
                                   int64_t(value0.imag()) * value1.real();
              return { static_cast<int16_t>(real), static_cast<int16_t>(imag) };
          },
          generate_complex16_input,
          volk_16ic_x2_multiply_16ic_get_func_desc() } })),
    ElementwiseName());
TEST_P(ComplexInteger8BinaryTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(
    Elementwise,
    ComplexInteger8BinaryTest,
    testing::ValuesIn(kernel_cases(std::vector<ComplexIntegerBinaryCase<lv_8sc_t>>{
        { "volk_8ic_x2_multiply_conjugate_16ic",
          volk_8ic_x2_multiply_conjugate_16ic_manual,
          [](const lv_8sc_t value0, const lv_8sc_t value1) -> lv_16sc_t {
              const int real =
                  int(value0.real()) * value1.real() + int(value0.imag()) * value1.imag();
              const int imag =
                  int(value0.imag()) * value1.real() - int(value0.real()) * value1.imag();
              return { static_cast<int16_t>(
                           std::min(real, int(std::numeric_limits<int16_t>::max()))),
                       static_cast<int16_t>(imag) };
          },
          generate_complex8_input,
          volk_8ic_x2_multiply_conjugate_16ic_get_func_desc() } })),
    ElementwiseName());

} // namespace
