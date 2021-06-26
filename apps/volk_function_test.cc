/* -*- c++ -*- */
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <volk/volk_prefs.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

typedef std::complex<float> complexf;

// https://www.fluentcpp.com/2019/05/24/how-to-fill-a-cpp-collection-with-random-values/
// I prefer the C++11 version for now.
template <class T>
class RandomComplexGaussianNumber
{
public:
    RandomComplexGaussianNumber(T mean, T sigma)
        : random_engine_{ std::random_device{}() }, distribution_{ mean, sigma }
    {
    }
    std::complex<T> operator()()
    {
        return std::complex<T>(distribution_(random_engine_),
                               distribution_(random_engine_));
    }

private:
    std::mt19937 random_engine_;
    std::normal_distribution<T> distribution_;
};


int main(int argc, char* argv[])
{
    volk_free_preferences();
    volk_initialize_preferences();
    auto prefs = volk_get_arch_prefs();
    auto nprefs = volk_get_num_arch_prefs();
    // for (unsigned i = 0; i < nprefs; i++) {
    //     std::cout << i << "\t" << prefs[i].name << std::endl;
    // }
    volk_free_preferences();
    const float mean = 1.3f;
    const float sigma = 0.7f;
    const unsigned num_points = 16;
    volk::vector<complexf> result(num_points);
    volk::vector<complexf> input0;
    volk::vector<complexf> input1;
    std::generate_n(std::back_inserter(input0),
                    num_points,
                    RandomComplexGaussianNumber<float>(mean, sigma));
    std::generate_n(std::back_inserter(input1),
                    num_points,
                    RandomComplexGaussianNumber<float>(mean, sigma));

    for (unsigned i = 0; i < 5; i++) {
        std::cout << i << std::endl;
        volk_32fc_x2_multiply_32fc(
            result.data(), input0.data(), input1.data(), result.size());
    }


    for (unsigned i = 0; i < result.size(); ++i) {
        std::cout << input0[i] << "\t" << input1[i] << "\t" << result[i] << std::endl;
    }
}