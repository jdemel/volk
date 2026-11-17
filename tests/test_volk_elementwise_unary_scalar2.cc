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
struct RangeCase {
    std::string name;
    void (*kernel)(float*, const float*, float, float, unsigned int, const char*);
    std::function<float(float, float, float)> reference;
    float lower;
    float upper;
    volk_func_desc_t description;
    void (*generate)(std::span<float>, std::mt19937&, float, float);
};

std::vector<RangeCase> range_kernels()
{
    std::vector<RangeCase> cases;
    for (const auto& [label, lower, upper] :
         { std::tuple{ "symmetric", -4.0f, 4.0f },
           std::tuple{ "asymmetric", -3.0f, 5.0f } }) {
        cases.push_back(
            { std::string("volk_32f_s32f_x2_clamp_32f_") + label,
              volk_32f_s32f_x2_clamp_32f_manual,
              [](const float value, const float minimum, const float maximum) {
                  return std::clamp(value, minimum, maximum);
              },
              lower,
              upper,
              volk_32f_s32f_x2_clamp_32f_get_func_desc(),
              generate_range_input });
        cases.push_back(
            { std::string("volk_32f_s32f_s32f_mod_range_32f_") + label,
              volk_32f_s32f_s32f_mod_range_32f_manual,
              [](const float value, const float minimum, const float maximum) {
                  const float width = maximum - minimum;
                  if (value < minimum) {
                      return maximum - std::fmod(minimum - value, width);
                  }
                  if (value > maximum) {
                      return minimum + std::fmod(value - maximum, width);
                  }
                  return value;
              },
              lower,
              upper,
              volk_32f_s32f_s32f_mod_range_32f_get_func_desc(),
              generate_range_input });
    }
    return cases;
}

class RangeFloatTest : public VolkTestImpl<std::tuple<std::string, size_t, RangeCase>>
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
        kernel.generate(input, engine, kernel.lower, kernel.upper);
        std::transform(
            input.begin(), input.end(), expected.begin(), [&kernel](const float value) {
                return kernel.reference(value, kernel.lower, kernel.upper);
            });
    }

    void run()
    {
        const auto& kernel = std::get<2>(GetParam());
        kernel.kernel(output.data(),
                      input.data(),
                      kernel.lower,
                      kernel.upper,
                      static_cast<unsigned int>(vector_length),
                      implementation_name.c_str());
        const auto equal = [](const float lhs, const float rhs) {
            return std::abs(lhs - rhs) <= 1e-5f;
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

TEST_P(RangeFloatTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         RangeFloatTest,
                         testing::ValuesIn(kernel_cases(range_kernels())),
                         ElementwiseName());


struct SpectralDensityCase {
    std::string name;
    float normalization;
    float bandwidth;
    volk_func_desc_t description;
};
std::vector<SpectralDensityCase> density_cases()
{
    const auto descriptor = volk_32fc_s32f_x2_power_spectral_density_32f_get_func_desc();
    return {
        { "volk_32fc_s32f_x2_power_spectral_density_32f_unit", 2.0f, 1.0f, descriptor },
        { "volk_32fc_s32f_x2_power_spectral_density_32f_scaled", 2.0f, 4.0f, descriptor }
    };
}
class SpectralDensityTest
    : public VolkTestImpl<std::tuple<std::string, size_t, SpectralDensityCase>>
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
        generate_spectrum_input(input, engine);
        const auto& kernel = std::get<2>(GetParam());
        std::transform(input.begin(),
                       input.end(),
                       expected.begin(),
                       [&kernel](const lv_32fc_t value) {
                           return 10.0f *
                                  std::log10(std::norm(value) /
                                             (kernel.normalization *
                                              kernel.normalization * kernel.bandwidth));
                       });
    }
    void run()
    {
        const auto& kernel = std::get<2>(GetParam());
        volk_32fc_s32f_x2_power_spectral_density_32f_manual(
            output.data(),
            input.data(),
            kernel.normalization,
            kernel.bandwidth,
            static_cast<unsigned int>(vector_length),
            implementation_name.c_str());
        EXPECT_TRUE(
            AreFloatingPointArraysEqualWithTolerance<float>(expected, output, 1e-3f));
    }
    volk::vector<lv_32fc_t> input_storage;
    volk::vector<float> output_storage, expected;
    std::span<lv_32fc_t> input;
    std::span<float> output;
};
TEST_P(SpectralDensityTest, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         SpectralDensityTest,
                         testing::ValuesIn(kernel_cases(density_cases())),
                         ElementwiseName());

} // namespace
