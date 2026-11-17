/* -*- c++ -*- */
/*
 * * Copyright 2026 Johannes Sterz Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_test.h"
#include <array>
#include <complex>
#include <limits>

TEST(VolkComparisons, RealAbsoluteErrorSpecialValues)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const auto check = [](float a, float b) {
        return AreFloatingPointArraysEqualWithTolerance<float>(
            std::array{ a }, std::array{ b }, 0.5f);
    };
    EXPECT_TRUE(check(nan, nan));
    EXPECT_FALSE(check(nan, 1.0f));
    EXPECT_FALSE(check(1.0f, nan));
    EXPECT_FALSE(check(nan, inf));
    EXPECT_TRUE(check(inf, inf));
    EXPECT_TRUE(check(-inf, -inf));
    EXPECT_FALSE(check(inf, -inf));
    EXPECT_FALSE(check(inf, 1.0f));
    EXPECT_TRUE(check(1.0f, 1.5f));
    EXPECT_FALSE(check(1.0f, 1.5001f));
}

TEST(VolkComparisons, ComplexAbsoluteErrorSpecialValues)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const auto check = [](std::complex<float> a, std::complex<float> b) {
        return AreComplexFloatingPointArraysEqualWithAbsoluteError<std::complex<float>>(
            std::array{ a }, std::array{ b }, 0.5f);
    };
    EXPECT_FALSE(check({ nan, 0 }, { nan, 0 }));
    EXPECT_FALSE(check({ 0, nan }, { 0, 1 }));
    EXPECT_TRUE(check({ inf, -inf }, { inf, -inf }));
    EXPECT_FALSE(check({ inf, 0 }, { -inf, 0 }));
    EXPECT_FALSE(check({ 0, inf }, { 0, 1 }));
    EXPECT_TRUE(check({ 1, 2 }, { 1.5f, 1.5f }));
    EXPECT_FALSE(check({ 1, 2 }, { 1.5001f, 2 }));
}

TEST(VolkComparisons, ComplexUlpsAndReporting)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float near = std::nextafter(1.0f, 2.0f);
    const std::array expected{ std::complex<float>{ 1, 2 } };
    EXPECT_TRUE(AreComplexFloatingPointArraysAlmostEqual<std::complex<float>>(
        expected, std::array{ std::complex<float>{ near, 2 } }));
    EXPECT_FALSE(AreComplexFloatingPointArraysAlmostEqual<std::complex<float>>(
        expected, std::array{ std::complex<float>{ nan, 2 } }));
    const std::array different{ std::complex<float>{ 3, 4 },
                                std::complex<float>{ 3, 4 },
                                std::complex<float>{ 3, 4 },
                                std::complex<float>{ 3, 4 } };
    const std::array reference{ std::complex<float>{ 1, 2 },
                                std::complex<float>{ 1, 2 },
                                std::complex<float>{ 1, 2 },
                                std::complex<float>{ 1, 2 } };
    const auto failure = AreComplexFloatingPointArraysAlmostEqual<std::complex<float>>(
        reference, different);
    const auto absolute_failure =
        AreComplexFloatingPointArraysEqualWithAbsoluteError<std::complex<float>>(
            reference, different);
    EXPECT_FALSE(failure);
    EXPECT_FALSE(absolute_failure);
    EXPECT_EQ(std::string(failure.message()), std::string(absolute_failure.message()));
    EXPECT_NE(std::string(failure.message()).find("4 differences in total"),
              std::string::npos);
    EXPECT_EQ(std::string(failure.message()).find(" @ 3"), std::string::npos);
    const auto size_failure =
        AreComplexFloatingPointArraysAlmostEqual<std::complex<float>>(expected,
                                                                      different);
    EXPECT_FALSE(size_failure);
    EXPECT_NE(std::string(size_failure.message()).find("differs from actual size"),
              std::string::npos);
}

TEST(VolkComparisons, IntegerEqualityAndReporting)
{
    const std::array expected{ 1, 2, 3, 4 };
    const std::array different{ 5, 6, 7, 8 };
    EXPECT_TRUE(AreIntegerArraysEqual<int>(expected, expected));
    const auto failure = AreIntegerArraysEqual<int>(expected, different);
    EXPECT_FALSE(failure);
    EXPECT_NE(std::string(failure.message()).find("1 != 5 @ 0"), std::string::npos);
    EXPECT_NE(std::string(failure.message()).find("4 differences in total"),
              std::string::npos);
    EXPECT_EQ(std::string(failure.message()).find(" @ 3"), std::string::npos);
    const auto size_failure =
        AreIntegerArraysEqual<int>(expected, std::span(different).first(1));
    EXPECT_FALSE(size_failure);
    EXPECT_NE(std::string(size_failure.message()).find("differs from actual size"),
              std::string::npos);
}
