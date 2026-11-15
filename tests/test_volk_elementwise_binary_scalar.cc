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
#include <complex>
#include <random>
#include <span>

namespace {
struct ComplexScalarCase {
    std::string name;
    lv_32fc_t scalar;
    volk_func_desc_t description;
};
std::vector<ComplexScalarCase> complex_scalar_cases()
{
    const auto descriptor =
        volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_get_func_desc();
    return { { "volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_zero",
               { 0.0f, 0.0f },
               descriptor },
             { "volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_complex",
               { 0.5f, -2.0f },
               descriptor },
             { "volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_unit",
               { 1.0f, 0.0f },
               descriptor } };
}
class ComplexBinaryScalarTest
    : public VolkTestImpl<std::tuple<std::string, size_t, ComplexScalarCase>>
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
        generate_complex_float_input(vec0, engine);
        generate_complex_float_input(vec1, engine);
        const lv_32fc_t scalar = std::get<2>(GetParam()).scalar;
        std::transform(vec0.begin(),
                       vec0.end(),
                       vec1.begin(),
                       expected.begin(),
                       [scalar](const lv_32fc_t value0, const lv_32fc_t value1) {
                           return value0 + std::conj(value1) * scalar;
                       });
    }
    void run()
    {
        const auto& kernel = std::get<2>(GetParam());
        volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_manual(
            output.data(),
            vec0.data(),
            vec1.data(),
            &kernel.scalar,
            static_cast<unsigned int>(vector_length),
            implementation_name.c_str());
        const auto equal = [](const lv_32fc_t lhs, const lv_32fc_t rhs) {
            return std::abs(lhs.real() - rhs.real()) <=
                       2e-5f * (1.0f + std::abs(lhs.real())) &&
                   std::abs(lhs.imag() - rhs.imag()) <=
                       2e-5f * (1.0f + std::abs(lhs.imag()));
        };
        EXPECT_TRUE(compare_volk_arrays<lv_32fc_t>(
            expected, output, equal, describe_volk_difference<lv_32fc_t>));
    }
    volk::vector<lv_32fc_t> vec0_storage, vec1_storage, output_storage, expected;
    std::span<lv_32fc_t> vec0, vec1, output;
};
TEST_P(ComplexBinaryScalarTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         ComplexBinaryScalarTest,
                         testing::ValuesIn(kernel_cases(complex_scalar_cases())),
                         ElementwiseName());

struct RealScalarCase {
    std::string name;
    float scalar;
    volk_func_desc_t description;
};

std::vector<RealScalarCase> interleave_cases()
{
    const auto descriptor = volk_32f_x2_s32f_interleave_16ic_get_func_desc();
    return { { "volk_32f_x2_s32f_interleave_16ic_fraction", 0.5f, descriptor },
             { "volk_32f_x2_s32f_interleave_16ic_positive", 2.0f, descriptor },
             { "volk_32f_x2_s32f_interleave_16ic_negative", -2.0f, descriptor } };
}
class InterleaveScalarTest
    : public VolkTestImpl<std::tuple<std::string, size_t, RealScalarCase>>
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
        generate_interleave_input(vec0, vec1, engine);
        const float scalar = std::get<2>(GetParam()).scalar;
        std::transform(vec0.begin(),
                       vec0.end(),
                       vec1.begin(),
                       expected.begin(),
                       [scalar](const float value0, const float value1) {
                           return lv_16sc_t(
                               static_cast<int16_t>(std::rint(value0 * scalar)),
                               static_cast<int16_t>(std::rint(value1 * scalar)));
                       });
    }
    void run()
    {
        volk_32f_x2_s32f_interleave_16ic_manual(output.data(),
                                                vec0.data(),
                                                vec1.data(),
                                                std::get<2>(GetParam()).scalar,
                                                static_cast<unsigned int>(vector_length),
                                                implementation_name.c_str());
        EXPECT_TRUE(AreIntegerArraysEqual<lv_16sc_t>(expected, output));
    }
    volk::vector<float> vec0_storage, vec1_storage;
    volk::vector<lv_16sc_t> output_storage, expected;
    std::span<float> vec0, vec1;
    std::span<lv_16sc_t> output;
};
TEST_P(InterleaveScalarTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         InterleaveScalarTest,
                         testing::ValuesIn(kernel_cases(interleave_cases())),
                         ElementwiseName());

std::vector<RealScalarCase> multiply8_cases()
{
    const auto descriptor = volk_8ic_x2_s32f_multiply_conjugate_32fc_get_func_desc();
    return { { "volk_8ic_x2_s32f_multiply_conjugate_32fc_fraction", 0.5f, descriptor },
             { "volk_8ic_x2_s32f_multiply_conjugate_32fc_positive", 3.0f, descriptor },
             { "volk_8ic_x2_s32f_multiply_conjugate_32fc_negative", -2.0f, descriptor } };
}
class Multiply8ScalarTest
    : public VolkTestImpl<std::tuple<std::string, size_t, RealScalarCase>>
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
        generate_complex8_input(vec0, engine);
        generate_complex8_input(vec1, engine);
        const float scalar = std::get<2>(GetParam()).scalar;
        std::transform(vec0.begin(),
                       vec0.end(),
                       vec1.begin(),
                       expected.begin(),
                       [scalar](const lv_8sc_t value0, const lv_8sc_t value1) {
                           const std::complex<float> first(value0.real(), value0.imag());
                           const std::complex<float> second(value1.real(), value1.imag());
                           return first * std::conj(second) * (1.0f / scalar);
                       });
    }
    void run()
    {
        volk_8ic_x2_s32f_multiply_conjugate_32fc_manual(
            output.data(),
            vec0.data(),
            vec1.data(),
            std::get<2>(GetParam()).scalar,
            static_cast<unsigned int>(vector_length),
            implementation_name.c_str());
        const auto equal = [](const lv_32fc_t lhs, const lv_32fc_t rhs) {
            return std::abs(lhs.real() - rhs.real()) <=
                       2e-6f * (1.0f + std::abs(lhs.real())) &&
                   std::abs(lhs.imag() - rhs.imag()) <=
                       2e-6f * (1.0f + std::abs(lhs.imag()));
        };
        EXPECT_TRUE(compare_volk_arrays<lv_32fc_t>(
            expected, output, equal, describe_volk_difference<lv_32fc_t>));
    }
    volk::vector<lv_8sc_t> vec0_storage, vec1_storage;
    volk::vector<lv_32fc_t> output_storage, expected;
    std::span<lv_8sc_t> vec0, vec1;
    std::span<lv_32fc_t> output;
};
TEST_P(Multiply8ScalarTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         Multiply8ScalarTest,
                         testing::ValuesIn(kernel_cases(multiply8_cases())),
                         ElementwiseName());

} // namespace
