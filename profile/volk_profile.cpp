
/* -*- c++ -*- */
/*
 * Copyright 2021 Johannes Demel
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "volk_machines.h"
#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <algorithm>
#include <complex>
#include <random>
#include <vector>

typedef std::complex<float> complexf;


static std::vector<std::string> get_arch_list(volk_func_desc_t desc)
{
    std::vector<std::string> archlist;

    for (size_t i = 0; i < desc.n_impls; i++) {
        archlist.push_back(std::string(desc.impl_names[i]));
    }

    return archlist;
}

float get_random_float(const float mean = 0.0, const float variance = 10.0)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    std::normal_distribution<float> dist{ mean, variance };
    return dist(mersenne_engine);
}

std::vector<float> initialize_random_float_vector(const unsigned size,
                                                  const float variance = 10.0)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    std::normal_distribution<float> dist{ 0, variance };
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::vector<float> vec(size);
    std::generate(vec.begin(), vec.end(), gen);
    return vec;
}


std::vector<complexf> initialize_random_complex_vector(const unsigned size,
                                                       const float variance = 10.0)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    std::normal_distribution<float> dist{ 0, variance };
    auto gen = [&dist, &mersenne_engine]() {
        return std::complex<float>(dist(mersenne_engine), dist(mersenne_engine));
    };

    std::vector<std::complex<float>> vec(size);
    std::generate(vec.begin(), vec.end(), gen);
    return vec;
}


static void BM_volk_32fc_x2_multiply_32fc(benchmark::State& state,
                                          const std::string& impl_name,
                                          const size_t vector_length)
{
    bool is_aligned = false;
    size_t ref_length = vector_length;
    if (impl_name.find("a_") != std::string::npos) {
        is_aligned = true;
        ref_length += 1;
    }
    auto func_ptr = volk_32fc_x2_multiply_32fc_func_manual(impl_name.c_str());

    volk::vector<complexf> result(ref_length);
    auto in0std = initialize_random_complex_vector(ref_length);
    auto in1std = initialize_random_complex_vector(ref_length);
    volk::vector<complexf> in0(in0std.begin(), in0std.end());
    volk::vector<complexf> in1(in1std.begin(), in1std.end());

    auto res_ptr = result.data();
    auto in0_ptr = in0.data();
    auto in1_ptr = in1.data();
    if (not is_aligned) {
        res_ptr += 1;
        in0_ptr += 1;
        in1_ptr += 1;
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(in0_ptr);
        benchmark::DoNotOptimize(in1_ptr);
        benchmark::DoNotOptimize(res_ptr);
        func_ptr(res_ptr, in0_ptr, in1_ptr, vector_length);
        benchmark::DoNotOptimize(in0_ptr);
        benchmark::DoNotOptimize(in1_ptr);
        benchmark::DoNotOptimize(res_ptr);
    }
    state.counters["SampleRate"] = benchmark::Counter(vector_length * state.iterations(),
                                                      benchmark::Counter::kIsRate);
}

static void BM_volk_32fc_x2_multiply_32fc_inplace(benchmark::State& state,
                                          const std::string& impl_name,
                                          const size_t vector_length)
{
    bool is_aligned = false;
    size_t ref_length = vector_length;
    if (impl_name.find("a_") != std::string::npos) {
        is_aligned = true;
        ref_length += 1;
    }
    auto func_ptr = volk_32fc_x2_multiply_32fc_func_manual(impl_name.c_str());

    // volk::vector<complexf> result(ref_length);
    auto in0std = initialize_random_complex_vector(ref_length);
    auto in1std = initialize_random_complex_vector(ref_length);
    volk::vector<complexf> in0(in0std.begin(), in0std.end());
    volk::vector<complexf> in1(in1std.begin(), in1std.end());

    // auto res_ptr = result.data();
    auto in0_ptr = in0.data();
    auto in1_ptr = in1.data();
    if (not is_aligned) {
        // res_ptr += 1;
        in0_ptr += 1;
        in1_ptr += 1;
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(in0_ptr);
        benchmark::DoNotOptimize(in1_ptr);
        // benchmark::DoNotOptimize(res_ptr);
        func_ptr(in0_ptr, in0_ptr, in1_ptr, vector_length);
        benchmark::DoNotOptimize(in0_ptr);
        benchmark::DoNotOptimize(in1_ptr);
        // benchmark::DoNotOptimize(res_ptr);
    }
    state.counters["SampleRate"] = benchmark::Counter(vector_length * state.iterations(),
                                                      benchmark::Counter::kIsRate);
}


static void BM_volk_32f_s32f_multiply_32f(benchmark::State& state,
                                          const std::string& impl_name,
                                          const size_t vector_length)
{
    bool is_aligned = false;
    size_t ref_length = vector_length;
    if (impl_name.find("a_") != std::string::npos) {
        is_aligned = true;
        ref_length += 1;
    }
    auto func_ptr = volk_32f_s32f_multiply_32f_func_manual(impl_name.c_str());

    volk::vector<float> result(ref_length);
    auto in0std = initialize_random_float_vector(ref_length);
    volk::vector<float> in0(in0std.begin(), in0std.end());
    float scalar = get_random_float();

    auto res_ptr = result.data();
    auto in0_ptr = in0.data();
    if (not is_aligned) {
        res_ptr += 1;
        in0_ptr += 1;
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(in0_ptr);
        benchmark::DoNotOptimize(scalar);
        benchmark::DoNotOptimize(res_ptr);
        func_ptr(res_ptr, in0_ptr, scalar, vector_length);
        benchmark::DoNotOptimize(in0_ptr);
        benchmark::DoNotOptimize(scalar);
        benchmark::DoNotOptimize(res_ptr);
    }
    state.counters["SampleRate"] = benchmark::Counter(vector_length * state.iterations(),
                                                      benchmark::Counter::kIsRate);
}

static void BM_volk_32f_s32f_multiply_32f_inplace(benchmark::State& state,
                                                  const std::string& impl_name,
                                                  const size_t vector_length)
{
    bool is_aligned = false;
    size_t ref_length = vector_length;
    if (impl_name.find("a_") != std::string::npos) {
        is_aligned = true;
        ref_length += 1;
    }
    auto func_ptr = volk_32f_s32f_multiply_32f_func_manual(impl_name.c_str());

    // volk::vector<float> result(ref_length);
    auto in0std = initialize_random_float_vector(ref_length);
    volk::vector<float> in0(in0std.begin(), in0std.end());
    float scalar = get_random_float();

    // auto res_ptr = result.data();
    auto in0_ptr = in0.data();
    if (not is_aligned) {
        // res_ptr += 1;
        in0_ptr += 1;
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(in0_ptr);
        benchmark::DoNotOptimize(scalar);
        // benchmark::DoNotOptimize(res_ptr);
        func_ptr(in0_ptr, in0_ptr, scalar, vector_length);
        benchmark::DoNotOptimize(in0_ptr);
        benchmark::DoNotOptimize(scalar);
        // benchmark::DoNotOptimize(res_ptr);
    }
    state.counters["SampleRate"] = benchmark::Counter(vector_length * state.iterations(),
                                                      benchmark::Counter::kIsRate);
}


std::string get_argument_value(const std::string& arg_text, const std::string& arg_name)
{
    auto flag_name = std::string("--") + arg_name + std::string("=");
    auto flag_pos = arg_text.find(flag_name);
    if (flag_pos == std::string::npos) {
        return std::string("");
    } else {
        return arg_text.substr(flag_pos + flag_name.size());
    }
}


int parse_int_argument_value(const std::string& arg_value)
{
    return std::stoi(arg_value);
}

size_t find_vector_length_argument(int argc, char** argv)
{
    size_t vector_length = 131071;
    for (int i = 0; i < argc; ++i) {
        auto s = get_argument_value(std::string(argv[i]), std::string("vlen"));
        if (s.size() > 0) {
            vector_length = parse_int_argument_value(s);
            return vector_length;
        }
    }
    return vector_length;
};

int main(int argc, char** argv)
{
    size_t vector_length = find_vector_length_argument(argc, argv);
    fmt::print("benchmark vector length: {}\n", vector_length);

    {
        auto function_description = volk_32f_s32f_multiply_32f_get_func_desc();
        auto archs = get_arch_list(function_description);
        for (const auto& arch : archs) {
            // fmt::print("{}\n", arch);
            auto name =
                fmt::format("volk_32f_s32f_multiply_32f/{}/{}", arch, vector_length);
            benchmark::RegisterBenchmark(
                name.c_str(), BM_volk_32f_s32f_multiply_32f, arch, vector_length);
        }
    }

    {
        auto function_description = volk_32f_s32f_multiply_32f_get_func_desc();
        auto archs = get_arch_list(function_description);
        for (const auto& arch : archs) {
            // fmt::print("{}\n", arch);
            auto name = fmt::format(
                "volk_32f_s32f_multiply_32f_inplace/{}/{}", arch, vector_length);
            benchmark::RegisterBenchmark(
                name.c_str(), BM_volk_32f_s32f_multiply_32f_inplace, arch, vector_length);
        }
    }


    {
        auto function_description = volk_32fc_x2_multiply_32fc_get_func_desc();
        auto archs = get_arch_list(function_description);
        for (const auto& arch : archs) {
            // fmt::print("{}\n", arch);
            auto name =
                fmt::format("volk_32fc_x2_multiply_32fc/{}/{}", arch, vector_length);
            benchmark::RegisterBenchmark(
                name.c_str(), BM_volk_32fc_x2_multiply_32fc, arch, vector_length);
        }
    }

    {
        auto function_description = volk_32fc_x2_multiply_32fc_get_func_desc();
        auto archs = get_arch_list(function_description);
        for (const auto& arch : archs) {
            // fmt::print("{}\n", arch);
            auto name =
                fmt::format("volk_32fc_x2_multiply_32fc_inplace/{}/{}", arch, vector_length);
            benchmark::RegisterBenchmark(
                name.c_str(), BM_volk_32fc_x2_multiply_32fc_inplace, arch, vector_length);
        }
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}

// BENCHMARK_MAIN();