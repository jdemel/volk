/* -*- c++ -*- */
/*
 * Copyright 2022 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_test.h"
#include <gtest/gtest.h>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <algorithm>
#include <array>
#include <numeric>
#include <span>


class volk_32f_x3_sum_of_poly_32f_test : public VolkTest
{
protected:
    void initialize_data() override
    {
        const size_t length = vector_length;

        vec0_storage.resize(length + offset);
        center_storage.resize(5 + offset);
        cutoff_storage.resize(1 + offset);
        result_storage.resize(1 + offset);
        vec0 = std::span(vec0_storage).subspan(offset);
        center_points = std::span(center_storage).subspan(offset);
        cutoff = std::span(cutoff_storage).subspan(offset);
        result = std::span(result_storage).subspan(offset);
        constexpr std::array values{ -3.0f, -1.5f, -1.25f, 0.0f, 1.0f, 3.0f };
        size_t input_index = 0;
        std::generate(vec0.begin(), vec0.end(), [&input_index, &values] {
            return values.at(input_index++ % values.size());
        });
        constexpr std::array centers{ 4.4f, 2.1f, 0.3f, 0.05f, 4.1f };
        std::copy(centers.begin(), centers.end(), center_points.begin());
        cutoff_storage.at(offset) = -1.5f;

        const float cutoff_value = cutoff_storage.at(offset);
        const float c0 = centers.at(0);
        const float c1 = centers.at(1);
        const float c2 = centers.at(2);
        const float c3 = centers.at(3);
        expected = std::accumulate(
            vec0.begin(),
            vec0.end(),
            0.0,
            [cutoff_value, c0, c1, c2, c3](double sum, double value) {
                value = std::max(value, static_cast<double>(cutoff_value));
                const double sq = value * value;
                const double cube = sq * value;
                const double quartic = value * cube;
                return sum + (value * c0 + sq * c1 + cube * c2 + quartic * c3);
            });
        expected += static_cast<double>(centers.at(4)) * static_cast<double>(length);

        // Long float reductions accumulate more rounding error than the double reference.
        const double relative_tolerance = length >= 65536 ? 5e-5 : 1e-5;
        tolerance = std::max(std::abs(expected) * relative_tolerance, 1e-5);
    }

    void run()
    {
        volk_32f_x3_sum_of_poly_32f_manual(result.data(),
                                           vec0.data(),
                                           center_points.data(),
                                           cutoff.data(),
                                           static_cast<unsigned int>(vector_length),
                                           implementation_name.c_str());
        EXPECT_NEAR(result_storage.at(offset), expected, tolerance);
    }

    volk::vector<float> vec0_storage;
    volk::vector<float> center_storage;
    volk::vector<float> cutoff_storage;
    volk::vector<float> result_storage;
    std::span<float> vec0;
    std::span<float> center_points;
    std::span<float> cutoff;
    std::span<float> result;
    double expected = 0.0;
    double tolerance = 1.0e-5;
};

TEST_P(volk_32f_x3_sum_of_poly_32f_test, run)
{
    report_runtime([this] { run(); });
}

INSTANTIATE_TEST_SUITE_P(
    volk_32f_x3_sum_of_poly_32f,
    volk_32f_x3_sum_of_poly_32f_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32f_x3_sum_of_poly_32f_get_func_desc())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());
