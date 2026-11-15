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
#include <tuple>

namespace {
class InplaceScalarTest : public VolkTestImpl<std::tuple<std::string, size_t, float>>
{
protected:
    void initialize_data() override
    {
        storage.resize(vector_length + offset);
        expected.resize(vector_length);
        values = std::span(storage).subspan(offset, vector_length);
        std::mt19937 engine(0xdeadbeef);
        std::normal_distribution<float> distribution(0.0f, 1000000.0f);
        generate_normal_input(values, engine, distribution);
        const float scalar = std::get<2>(GetParam());
        std::transform(values.begin(),
                       values.end(),
                       expected.begin(),
                       [scalar](const float value) { return value / scalar; });
    }
    void run()
    {
        const float scalar = std::get<2>(GetParam());
        volk_32f_s32f_normalize_manual(values.data(),
                                       scalar,
                                       static_cast<unsigned int>(vector_length),
                                       implementation_name.c_str());
        const auto equal = [](const float lhs, const float rhs) {
            return std::abs(lhs - rhs) <= 2e-6f * (1.0f + std::abs(lhs));
        };
        EXPECT_TRUE(compare_volk_arrays<float>(
            expected, values, equal, describe_volk_difference<float>));
    }
    volk::vector<float> storage;
    volk::vector<float> expected;
    std::span<float> values;
};

TEST_P(InplaceScalarTest, run)
{
    report_runtime([this] { run(); });
}
struct ScalarName {
    std::string
    operator()(const testing::TestParamInfo<InplaceScalarTest::ParamType>& info) const
    {
        const auto& [implementation, size, scalar] = info.param;
        return fmt::format("{}_{}_{}",
                           implementation,
                           size,
                           scalar == -2.0f  ? "negative"
                           : scalar == 0.5f ? "fraction"
                                            : "positive");
    }
};
INSTANTIATE_TEST_SUITE_P(
    volk_32f_s32f_normalize,
    InplaceScalarTest,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32f_s32f_normalize_get_func_desc())),
                     testing::ValuesIn(default_vector_sizes),
                     testing::Values(-2.0f, 0.5f, 3.0f)),
    ScalarName());
} // namespace
