/* -*- c++ -*- */
/*
 * Copyright 2022 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk/volk_complex.h"
#include "volk_test.h"
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <complex>
#include <span>

class volk_32fc_x2_multiply_32fc_test : public VolkTest
{
protected:
    void initialize_data() override
    {
        const size_t length = vector_length;
        vec0_storage.resize(length + offset);
        vec1_storage.resize(length + offset);
        result_storage.resize(length + offset);
        vec0 = std::span(vec0_storage).subspan(offset);
        vec1 = std::span(vec1_storage).subspan(offset);
        result = std::span(result_storage).subspan(offset);
        for (size_t i = 0; i < length; ++i) {
            vec0[i] =
                std::complex<float>(static_cast<float>(static_cast<double>(i) * 3.14),
                                    static_cast<float>(static_cast<double>(i) * 0.45));
            vec1[i] =
                std::complex<float>(static_cast<float>(static_cast<double>(i) * -2.78),
                                    static_cast<float>(static_cast<double>(i) * 5.44));
        }

        expected = volk::vector<lv_32fc_t>(length);
        for (size_t i = 0; i < length; ++i) {
            expected[i] = vec0[i] * vec1[i];
        }
    }

    void run()
    {
        volk_32fc_x2_multiply_32fc_manual(result.data(),
                                          vec0.data(),
                                          vec1.data(),
                                          static_cast<unsigned int>(vector_length),
                                          implementation_name.c_str());
        EXPECT_TRUE(
            AreComplexFloatingPointArraysAlmostEqual<lv_32fc_t>(expected, result));
    }

    volk::vector<lv_32fc_t> vec0_storage;
    volk::vector<lv_32fc_t> vec1_storage;
    volk::vector<lv_32fc_t> result_storage;
    std::span<lv_32fc_t> vec0;
    std::span<lv_32fc_t> vec1;
    std::span<lv_32fc_t> result;
    volk::vector<lv_32fc_t> expected;
};

TEST_P(volk_32fc_x2_multiply_32fc_test, run)
{
    report_runtime([this] { run(); });
}


INSTANTIATE_TEST_SUITE_P(
    volk_32fc_x2_multiply_32fc,
    volk_32fc_x2_multiply_32fc_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32fc_x2_multiply_32fc_get_func_desc())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());
