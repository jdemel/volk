/* -*- c++ -*- */
/*
 * Copyright 2026 Johannes Sterz Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#pragma once

#include "volk_test.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

template <typename T>
void generate_normal_input(std::span<T> values,
                           std::mt19937& engine,
                           std::normal_distribution<T>& distribution)
{
    std::generate(values.begin(), values.end(), [&engine, &distribution] {
        return distribution(engine);
    });
}

template <typename T>
void generate_uniform_integer_input(std::span<T> values,
                                    std::mt19937& engine,
                                    std::uniform_int_distribution<int64_t>& distribution)
{
    std::generate(values.begin(), values.end(), [&engine, &distribution] {
        return static_cast<T>(distribution(engine));
    });
}

template <typename T, size_t N>
void set_binary_edges(std::span<T> vec0,
                      std::span<T> vec1,
                      const std::array<std::pair<T, T>, N>& edges)
{
    for (size_t i = 0; i < std::min(vec0.size(), edges.size()); ++i) {
        vec0[i] = edges[i].first;
        vec1[i] = edges[i].second;
    }
}

template <std::floating_point T>
void generate_binary_input(std::span<T> vec0, std::span<T> vec1, std::mt19937& engine)
{
    std::normal_distribution<T> distribution(T(0), T(1000000));
    generate_normal_input(vec0, engine, distribution);
    generate_normal_input(vec1, engine, distribution);
    const std::array edges{ std::pair{ T(0), T(0) },
                            std::pair{ T(1), T(-1) },
                            std::pair{ T(-1), T(1) },
                            std::pair{ T(1), T(1) } };
    set_binary_edges(vec0, vec1, edges);
}

template <std::floating_point T>
void generate_division_input(std::span<T> vec0, std::span<T> vec1, std::mt19937& engine)
{
    generate_binary_input(vec0, vec1, engine);
    std::transform(vec1.begin(), vec1.end(), vec1.begin(), [](const T value) {
        return std::abs(value) < T(1) ? std::copysign(T(1), value) : value;
    });
}

template <std::integral T>
void generate_integer_binary_input(std::span<T> vec0,
                                   std::span<T> vec1,
                                   std::mt19937& engine)
{
    std::uniform_int_distribution<int64_t> distribution(std::numeric_limits<T>::min(),
                                                        std::numeric_limits<T>::max());
    generate_uniform_integer_input(vec0, engine, distribution);
    generate_uniform_integer_input(vec1, engine, distribution);
    const T low = std::numeric_limits<T>::min();
    const T high = std::numeric_limits<T>::max();
    const std::array edges{ std::pair{ high, high }, std::pair{ low, low },
                            std::pair{ high, T(1) }, std::pair{ low, T(1) },
                            std::pair{ high, low },  std::pair{ T(0), high },
                            std::pair{ T(0), T(0) } };
    set_binary_edges(vec0, vec1, edges);
}

template <std::signed_integral T>
void generate_saturated_signed_input(std::span<T> vec0,
                                     std::span<T> vec1,
                                     std::mt19937& engine)
{
    generate_integer_binary_input(vec0, vec1, engine);
    if (vec0.size() > 7) {
        vec0[7] = std::numeric_limits<T>::min();
        vec1[7] = T(-1);
    }
}

inline void generate_bitwise_input(std::span<int32_t> vec0,
                                   std::span<int32_t> vec1,
                                   std::mt19937& engine)
{
    generate_integer_binary_input(vec0, vec1, engine);
    if (vec0.size() > 7) {
        vec0[7] = -1;
        vec1[7] = 0x55555555;
    }
    if (vec0.size() > 8) {
        vec0[8] = 0x55555555;
        vec1[8] = ~int32_t(0x55555555);
    }
}

inline void generate_mixed_binary_input(std::span<float> vec0,
                                        std::span<double> vec1,
                                        std::mt19937& engine)
{
    std::normal_distribution<float> float_distribution(0.0f, 1000000.0f);
    std::normal_distribution<double> double_distribution(0.0, 1000000.0);
    generate_normal_input(vec0, engine, float_distribution);
    generate_normal_input(vec1, engine, double_distribution);
    const std::array edges{ std::pair{ 0.0f, 0.0 },
                            std::pair{ 1.0f, -1.0 },
                            std::pair{ -1.0f, 1.0 },
                            std::pair{ 1.0f, 1.0 } };
    for (size_t i = 0; i < std::min(vec0.size(), edges.size()); ++i) {
        vec0[i] = edges[i].first;
        vec1[i] = edges[i].second;
    }
}

inline void generate_range_input(std::span<float> values,
                                 std::mt19937& engine,
                                 const float lower,
                                 const float upper)
{
    std::normal_distribution<float> distribution(0.0f, 32.0f);
    generate_normal_input(values, engine, distribution);
    const float width = upper - lower;
    const std::array edges{
        lower,
        std::nextafter(lower, -std::numeric_limits<float>::infinity()),
        std::nextafter(lower, std::numeric_limits<float>::infinity()),
        upper,
        std::nextafter(upper, -std::numeric_limits<float>::infinity()),
        std::nextafter(upper, std::numeric_limits<float>::infinity()),
        lower - width,
        upper + width,
        (lower + upper) / 2.0f
    };
    std::copy_n(edges.begin(), std::min(values.size(), edges.size()), values.begin());
}

template <std::unsigned_integral T>
void generate_byteswap_input(std::span<T> values, std::mt19937& engine)
{
    std::uniform_int_distribution<uint64_t> distribution(0,
                                                         std::numeric_limits<T>::max());
    std::generate(values.begin(), values.end(), [&engine, &distribution] {
        return static_cast<T>(distribution(engine));
    });
    const std::array<T, 5> edges{ T(0),
                                  T(1),
                                  std::numeric_limits<T>::max(),
                                  static_cast<T>(uint64_t(0x0123456789abcdef)),
                                  static_cast<T>(uint64_t(0xfedcba9876543210)) };
    std::copy_n(edges.begin(), std::min(values.size(), edges.size()), values.begin());
}

inline void generate_complex8_input(std::span<lv_8sc_t> values, std::mt19937& engine)
{
    std::uniform_int_distribution<int> distribution(std::numeric_limits<int8_t>::min(),
                                                    std::numeric_limits<int8_t>::max());
    std::generate(values.begin(), values.end(), [&engine, &distribution] {
        return lv_8sc_t(static_cast<int8_t>(distribution(engine)),
                        static_cast<int8_t>(distribution(engine)));
    });
    if (!values.empty()) {
        values[0] = lv_8sc_t(std::numeric_limits<int8_t>::min(),
                             std::numeric_limits<int8_t>::max());
    }
}

inline void generate_complex16_input(std::span<lv_16sc_t> values, std::mt19937& engine)
{
    std::uniform_int_distribution<int> distribution(std::numeric_limits<int16_t>::min(),
                                                    std::numeric_limits<int16_t>::max());
    std::generate(values.begin(), values.end(), [&engine, &distribution] {
        return lv_16sc_t(static_cast<int16_t>(distribution(engine)),
                         static_cast<int16_t>(distribution(engine)));
    });
    if (!values.empty()) {
        values[0] = lv_16sc_t(std::numeric_limits<int16_t>::min(),
                              std::numeric_limits<int16_t>::max());
    }
}

inline void generate_complex_float_input(std::span<lv_32fc_t> values,
                                         std::mt19937& engine)
{
    std::normal_distribution<float> distribution(0.0f, 1000.0f);
    std::generate(values.begin(), values.end(), [&engine, &distribution] {
        return lv_32fc_t(distribution(engine), distribution(engine));
    });
    if (!values.empty()) {
        values[0] = lv_32fc_t(0.0f, 0.0f);
    }
}

inline void generate_interleave_input(std::span<float> vec0,
                                      std::span<float> vec1,
                                      std::mt19937& engine)
{
    std::normal_distribution<float> distribution(0.0f, 1000.0f);
    generate_normal_input(vec0, engine, distribution);
    generate_normal_input(vec1, engine, distribution);
    const std::array edges{ std::pair{ 0.25f, -0.25f },
                            std::pair{ 0.75f, -0.75f },
                            std::pair{ 0.0f, 1.0f },
                            std::pair{ -127.5f, 127.5f } };
    set_binary_edges(vec0, vec1, edges);
}

inline void generate_slicer_input(std::span<float> values, std::mt19937& engine)
{
    std::normal_distribution<float> distribution(0.0f, 1000000.0f);
    generate_normal_input(values, engine, distribution);
    const std::array edges{ 0.0f,
                            -0.0f,
                            std::numeric_limits<float>::min(),
                            -std::numeric_limits<float>::min(),
                            1.0f,
                            -1.0f };
    std::copy_n(edges.begin(), std::min(values.size(), edges.size()), values.begin());
}

inline void generate_complex_binary_input(std::span<lv_32fc_t> vec0,
                                          std::span<lv_32fc_t> vec1,
                                          std::mt19937& engine)
{
    generate_complex_float_input(vec0, engine);
    generate_complex_float_input(vec1, engine);
}

inline void generate_complex_division_input(std::span<lv_32fc_t> vec0,
                                            std::span<lv_32fc_t> vec1,
                                            std::mt19937& engine)
{
    generate_complex_binary_input(vec0, vec1, engine);
    std::transform(vec1.begin(), vec1.end(), vec1.begin(), [](const lv_32fc_t value) {
        return std::abs(value) < 1.0f ? lv_32fc_t(1.0f, 0.0f) : value;
    });
}

inline void generate_complex_real_input(std::span<lv_32fc_t> vec0,
                                        std::span<float> vec1,
                                        std::mt19937& engine)
{
    generate_complex_float_input(vec0, engine);
    std::normal_distribution<float> distribution(0.0f, 1000.0f);
    generate_normal_input(vec1, engine, distribution);
}

inline void generate_scalar_input(std::span<float> values, std::mt19937& engine)
{
    std::normal_distribution<float> distribution(0.0f, 1000000.0f);
    generate_normal_input(values, engine, distribution);
}

inline void generate_power_input(std::span<float> values, std::mt19937& engine)
{
    std::normal_distribution<float> distribution(0.0f, 100.0f);
    generate_normal_input(values, engine, distribution);
    std::transform(values.begin(), values.end(), values.begin(), [](const float value) {
        return std::max(std::abs(value), 0.125f);
    });
    const std::array edges{ 1.0f, 0.5f, 2.0f, 0.125f };
    std::copy_n(edges.begin(), std::min(values.size(), edges.size()), values.begin());
}

inline void generate_binary_power_input(std::span<float> exponent,
                                        std::span<float> base,
                                        std::mt19937& engine)
{
    std::uniform_real_distribution<float> base_distribution{ 1.0e-2f, 100.0f };
    std::uniform_real_distribution<float> exponent_distribution{ -18.0f, 18.0f };
    std::generate(base.begin(), base.end(), [&] { return base_distribution(engine); });
    std::generate(
        exponent.begin(), exponent.end(), [&] { return exponent_distribution(engine); });
}

inline void generate_complex16_magnitude_input(std::span<lv_16sc_t> values,
                                               std::mt19937& engine)
{
    // Keep magnitudes representable in int16_t.
    std::uniform_int_distribution<int> distribution(-23169, 23169);
    std::generate(values.begin(), values.end(), [&engine, &distribution] {
        return lv_16sc_t(static_cast<int16_t>(distribution(engine)),
                         static_cast<int16_t>(distribution(engine)));
    });
    const std::array edges{
        lv_16sc_t(32767, 0), lv_16sc_t(0, -32767), lv_16sc_t(0, 0), lv_16sc_t(3, 4)
    };
    std::copy_n(edges.begin(), std::min(values.size(), edges.size()), values.begin());
}

inline void generate_spectrum_input(std::span<lv_32fc_t> values, std::mt19937& engine)
{
    generate_complex_float_input(values, engine);
    std::transform(
        values.begin(), values.end(), values.begin(), [](const lv_32fc_t value) {
            return std::norm(value) < 1.0f ? lv_32fc_t(1.0f, 0.0f) : value;
        });
}

template <typename Case>
std::vector<std::tuple<std::string, size_t, Case>>
kernel_cases(const std::vector<Case>& kernels)
{
    std::vector<std::tuple<std::string, size_t, Case>> cases;
    for (const auto& kernel : kernels) {
        for (const auto& implementation :
             get_kernel_implementation_name_list(kernel.description)) {
            for (const auto size : default_vector_sizes) {
                cases.emplace_back(implementation, size, kernel);
            }
        }
    }
    return cases;
}

struct ElementwiseName {
    template <typename Parameter>
    std::string operator()(const testing::TestParamInfo<Parameter>& info) const
    {
        const auto& [implementation, size, kernel] = info.param;
        return fmt::format("{}_{}_{}", kernel.name, implementation, size);
    }
};
