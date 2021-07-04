/* -*- c++ -*- */
/*
 * Copyright 2021 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#include <volk/qa_volk.h>
#include <volk/volk_alloc.hh>
#include <algorithm>
#include <exception>
#include <random>
#include <cmath>
#include <cstdlib>
#include <complex>

template <typename T>
volk::vector<T> initialize_random_vector(const unsigned vector_length)
{

    // We use static in order to instantiate the random engine
    // and the distribution once only.
    // It may provoke some thread-safety issues.
    static std::uniform_int_distribution<T> distribution(std::numeric_limits<T>::min(),
                                                         std::numeric_limits<T>::max());
    static std::default_random_engine generator;

    volk::vector<T> data(vector_length);
    std::generate(data.begin(), data.end(), []() { return distribution(generator); });
    return data;
}

template <>
volk::vector<lv_8sc_t> initialize_random_vector<lv_8sc_t>(const unsigned vector_length)
{
    static std::uniform_int_distribution<int8_t> distribution(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    static std::default_random_engine generator;

    volk::vector<lv_8sc_t> data(vector_length);
    std::generate(data.begin(), data.end(), []() {
        return lv_8sc_t(distribution(generator), distribution(generator));
    });
    return data;
}

template <>
volk::vector<lv_16sc_t> initialize_random_vector<lv_16sc_t>(const unsigned vector_length)
{
    static std::uniform_int_distribution<short> distribution(
        std::numeric_limits<short>::min(), std::numeric_limits<short>::max());
    static std::default_random_engine generator;

    volk::vector<lv_16sc_t> data(vector_length);
    std::generate(data.begin(), data.end(), []() {
        return lv_16sc_t(distribution(generator), distribution(generator));
    });
    return data;
}

template <>
volk::vector<lv_32fc_t> initialize_random_vector<lv_32fc_t>(const unsigned vector_length)
{
    static std::normal_distribution<float> distribution(0.0f, 1.0f);
    static std::default_random_engine generator;

    volk::vector<lv_32fc_t> data(vector_length);
    std::generate(data.begin(), data.end(), []() {
        return lv_32fc_t(distribution(generator), distribution(generator));
    });
    return data;
}

template <>
volk::vector<float> initialize_random_vector<float>(const unsigned vector_length)
{
    static std::normal_distribution<float> distribution(0.0f, 1.0f);
    static std::default_random_engine generator;

    volk::vector<float> data(vector_length);
    std::generate(data.begin(), data.end(), []() { return distribution(generator); });
    return data;
}

template <>
volk::vector<double> initialize_random_vector<double>(const unsigned vector_length)
{
    static std::normal_distribution<double> distribution(0.0f, 1.0f);
    static std::default_random_engine generator;

    volk::vector<double> data(vector_length);
    std::generate(data.begin(), data.end(), []() { return distribution(generator); });
    return data;
}


volk_qa_test::volk_qa_test(std::string name) : kernel_name(name) {}

std::vector<std::string>
volk_qa_test::get_arch_list(volk_func_desc_t desc)
{
    std::vector<std::string> archlist;

    for (size_t i = 0; i < desc.n_impls; i++) {
        archlist.push_back(std::string(desc.impl_names[i]));
    }
    archs = archlist;
    return archlist;
}

// void volk_qa_test::set_vector_length(const unsigned vector_length)
// {
//     throw std::logic_error(
//         "This function should always be overridden! Something unintended happened!");
// }

// clang-format off
%for kern in kernels:
%if 'num_points' in kern.args[-1][1]:

void
qa_${kern.name}::set_vector_length(const unsigned vector_length) {
    for(const auto arch : archs){
        arch_vectors[arch] = std::make_unique<data_vectors>();
    }
%for arg in kern.annotated_args:
    %if 'vector' in arg[0]:
    arch_vectors["generic"]->${arg[2]} = initialize_random_vector<${arg[1]}>(vector_length);
    %endif
%endfor
    for(const auto arch : archs){
        if("generic" != arch){
        %for arg in kern.annotated_args:
            %if 'vector' in arg[0]:
            arch_vectors[arch]->${arg[2]} = volk::vector<${arg[1]}>(
                arch_vectors["generic"]->${arg[2]}.begin(),
                arch_vectors["generic"]->${arg[2]}.end());
            %endif
        %endfor
        }
    }
}

volk_func_desc_t 
qa_${kern.name}::get_function_description(){
    return ${kern.name}_get_func_desc();
}

bool
qa_${kern.name}::check_results(const std::string& arch){
    if(std::string("generic") == arch){ return true; }
    bool equal = true;
    const float tolerance = 1e-3;
    
    %for arg in kern.annotated_args[:-1]:
    %if 'vector' in arg[0]:
    for(unsigned i = 0; i < arch_vectors["generic"]->${arg[2]}.size(); i++){
        const ${arg[1]} diff = ${arg[1]}(
                arch_vectors["generic"]->${arg[2]}[i] - 
                arch_vectors[arch]->${arg[2]}[i]
            );
        %if arg[1] in ("float", "double", "lv_32fc_t"):
        if(std::abs(diff) > tolerance){
        %else:
        if(diff != 0){
        %endif
            equal = false;
        }
    }
    %endif
    %endfor
    return equal;
}

void
qa_${kern.name}::execute(const unsigned vector_length, const std::string& arch) {
    ${kern.name}_manual( 
    %for arg in kern.annotated_args[:-1]:
        %if 'vector' in arg[0]:
        arch_vectors[arch]->${arg[2]}.data(), 
        %else:
        ${arg[2]},
        %endif
    %endfor         
        vector_length, arch.c_str());
}

%endif
%endfor
//clang-format on

