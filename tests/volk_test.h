/* -*- c++ -*- */
/*
 * Copyright 2022 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <string_view>
#include <volk/volk.h>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <span>
#include <tuple>

static constexpr std::array<size_t, 22> default_vector_sizes{
    0,  1,  3,  4,  5,  7,  8,   9,   15,  16,   17,
    31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 131071
};

std::vector<std::string> get_kernel_implementation_name_list(const volk_func_desc_t desc);

bool is_aligned_implementation_name(const std::string& name);

std::tuple<std::vector<std::string>, std::vector<std::string>>
separate_implementations_by_alignment(const std::vector<std::string>& names);

std::vector<std::string>
get_aligned_kernel_implementation_names(const volk_func_desc_t desc);
std::vector<std::string>
get_unaligned_kernel_implementation_names(const volk_func_desc_t desc);

struct generate_volk_test_name {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
    {
        return fmt::format("{}_{}", std::get<0>(info.param), std::get<1>(info.param));
    }
};

template <typename Parameter = std::tuple<std::string, size_t>>
class VolkTestImpl : public ::testing::TestWithParam<Parameter>
{
protected:
    void SetUp() override
    {
        initialize_test(this->GetParam());
        initialize_data();
    }

    virtual void initialize_data() {}

    void initialize_test(const Parameter& param)
    {
        implementation_name = std::get<0>(param);
        vector_length = std::get<1>(param);
        is_aligned_implementation = is_aligned_implementation_name(implementation_name);
        offset = is_aligned_implementation ? 0 : 1;
    }

    template <class Test>
    void report_runtime(Test&& test) const
    {
        fmt::print("test {} implementation: {:>12}, size={} ...",
                   is_aligned_implementation ? "aligned" : "unaligned",
                   implementation_name,
                   vector_length);
        const auto start = std::chrono::steady_clock::now();
        test();
        const std::chrono::duration<double> elapsed =
            std::chrono::steady_clock::now() - start;
        fmt::print("\tduration={}\n", elapsed);
    }

    std::string implementation_name;
    bool is_aligned_implementation;
    size_t vector_length;
    size_t offset;
};

using VolkTest = VolkTestImpl<>;

template <typename T>
struct is_complex : std::false_type {
};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {
};

template <class T, class Equal, class Describe>
::testing::AssertionResult compare_volk_arrays(std::span<const T> expected,
                                               std::span<const T> actual,
                                               Equal equal,
                                               Describe describe)
{
    auto result = ::testing::AssertionFailure();
    if (expected.size() != actual.size()) {
        return result << "expected result size=" << expected.size()
                      << " differs from actual size=" << actual.size();
    }
    size_t errors = 0;
    std::string_view separator = " ";
    for (size_t i = 0; i < expected.size(); ++i) {
        if (not equal(expected[i], actual[i])) {
            if (errors == 0) {
                result << "Differences found:";
            }
            if (errors < 3) {
                result << separator;
                describe(result, expected[i], actual[i], i);
                separator = ",\n";
            }
            ++errors;
        }
    }
    if (errors > 0) {
        return result << separator << errors << " differences in total";
    }
    return ::testing::AssertionSuccess();
}

template <typename T>
void describe_volk_difference(::testing::AssertionResult& result,
                              const T& lhs,
                              const T& rhs,
                              const size_t index)
{
    result << lhs << " != " << rhs << " @ " << index;
}

template <typename T>
::testing::AssertionResult AreIntegerArraysEqual(std::span<const T> expected,
                                                 std::span<const T> actual)
{
    const auto equal = [](const T lhs, const T rhs) { return lhs == rhs; };
    return compare_volk_arrays(expected, actual, equal, describe_volk_difference<T>);
}

template <class T>
::testing::AssertionResult
AreComplexFloatingPointArraysAlmostEqual(std::span<const T> expected,
                                         std::span<const T> actual)
{
    const auto equal = [](const T& lhs, const T& rhs) {
        return ::testing::internal::FloatingPoint(lhs.real())
                   .AlmostEquals(::testing::internal::FloatingPoint(rhs.real())) and
               ::testing::internal::FloatingPoint(lhs.imag())
                   .AlmostEquals(::testing::internal::FloatingPoint(rhs.imag()));
    };
    return compare_volk_arrays(expected, actual, equal, describe_volk_difference<T>);
}

template <class T>
::testing::AssertionResult
AreComplexFloatingPointArraysEqualWithAbsoluteError(std::span<const T> expected,
                                                    std::span<const T> actual,
                                                    const float absolute_error = 1.0e-7f)
{
    const auto equal_component = [absolute_error](const auto lhs, const auto rhs) {
        return ::testing::internal::DoubleNearPredFormat(
            "expected", "actual", "absolute_error", lhs, rhs, absolute_error);
    };
    const auto equal = [equal_component](const T& lhs, const T& rhs) {
        return equal_component(lhs.real(), rhs.real()) and
               equal_component(lhs.imag(), rhs.imag());
    };
    return compare_volk_arrays(expected, actual, equal, describe_volk_difference<T>);
}

template <class T>
::testing::AssertionResult
AreFloatingPointArraysEqualWithAbsoluteError(std::span<const T> expected,
                                             std::span<const T> actual,
                                             const float absolute_error = 1.0e-7f)
{
    const auto equal = [absolute_error](const T lhs, const T rhs) {
        return (::testing::internal::FloatingPoint(lhs).is_nan() and
                ::testing::internal::FloatingPoint(rhs).is_nan()) or
               ::testing::internal::DoubleNearPredFormat(
                   "expected", "actual", "absolute_error", lhs, rhs, absolute_error);
    };
    const auto describe =
        [absolute_error](auto& result, const T lhs, const T rhs, const size_t i) {
            result << fmt::format("@{}: {} != {}, error: {} > {}",
                                  i,
                                  lhs,
                                  rhs,
                                  std::abs(lhs - rhs),
                                  absolute_error);
        };
    return compare_volk_arrays(expected, actual, equal, describe);
}
