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
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {
template <std::unsigned_integral T>
T byteswap_reference(const T value)
{
    uint64_t result = 0;
    uint64_t remaining = value;
    for (size_t i = 0; i < sizeof(T); ++i) {
        result = (result << 8) | (remaining & 0xffu);
        remaining >>= 8;
    }
    return static_cast<T>(result);
}

template <typename T>
struct InplaceCase {
    std::string name;
    void (*kernel)(T*, unsigned int, const char*);
    volk_func_desc_t description;
    void (*generate)(std::span<T>, std::mt19937&);
};

template <typename T>
std::vector<InplaceCase<T>> inplace_kernels();

template <>
std::vector<InplaceCase<uint16_t>> inplace_kernels<uint16_t>()
{
    return { { "volk_16u_byteswap",
               volk_16u_byteswap_manual,
               volk_16u_byteswap_get_func_desc(),
               generate_byteswap_input<uint16_t> } };
}
template <>
std::vector<InplaceCase<uint32_t>> inplace_kernels<uint32_t>()
{
    return { { "volk_32u_byteswap",
               volk_32u_byteswap_manual,
               volk_32u_byteswap_get_func_desc(),
               generate_byteswap_input<uint32_t> } };
}
template <>
std::vector<InplaceCase<uint64_t>> inplace_kernels<uint64_t>()
{
    return { { "volk_64u_byteswap",
               volk_64u_byteswap_manual,
               volk_64u_byteswap_get_func_desc(),
               generate_byteswap_input<uint64_t> } };
}

template <typename T>
class InplaceTestBase
    : public VolkTestImpl<std::tuple<std::string, size_t, InplaceCase<T>>>
{
protected:
    void initialize_data() override
    {
        storage.resize(this->vector_length + this->offset);
        expected.resize(this->vector_length);
        values = std::span(storage).subspan(this->offset, this->vector_length);
        std::mt19937 engine(0xdeadbeef);
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.generate(values, engine);
        std::transform(values.begin(), values.end(), expected.begin(), [](const T value) {
            return byteswap_reference(value);
        });
    }
    void run()
    {
        const auto& kernel = std::get<2>(this->GetParam());
        kernel.kernel(values.data(),
                      static_cast<unsigned int>(this->vector_length),
                      this->implementation_name.c_str());
        EXPECT_TRUE(AreIntegerArraysEqual<T>(expected, values));
    }
    volk::vector<T> storage;
    volk::vector<T> expected;
    std::span<T> values;
};

class InplaceUInt16Test : public InplaceTestBase<uint16_t>
{
};
class InplaceUInt32Test : public InplaceTestBase<uint32_t>
{
};
class InplaceUInt64Test : public InplaceTestBase<uint64_t>
{
};

TEST_P(InplaceUInt16Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         InplaceUInt16Test,
                         testing::ValuesIn(kernel_cases(inplace_kernels<uint16_t>())),
                         ElementwiseName());
TEST_P(InplaceUInt32Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         InplaceUInt32Test,
                         testing::ValuesIn(kernel_cases(inplace_kernels<uint32_t>())),
                         ElementwiseName());
TEST_P(InplaceUInt64Test, run)
{
    report_runtime([this] { run(); });
}
INSTANTIATE_TEST_SUITE_P(Elementwise,
                         InplaceUInt64Test,
                         testing::ValuesIn(kernel_cases(inplace_kernels<uint64_t>())),
                         ElementwiseName());
} // namespace
