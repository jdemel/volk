/* -*- c++ -*- */
/*
 * Copyright 2012-2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
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

#include <stdbool.h> // for bool, false, true
#include <fstream>   // IWYU pragma: keep
#include <iostream>  // for operator<<, basic_ostream, endl, char...
#include <map>       // for map, map<>::iterator, _Rb_tree_iterator
#include <string>    // for string, operator<<
#include <utility>   // for pair
#include <vector>    // for vector
#include <cstdlib>

// #include "kernel_tests.h"      // for init_test_list
// #include "qa_utils.h"          // for volk_test_case_t, volk_test_results_t
#include "volk/volk_complex.h" // for lv_32fc_t
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <volk/volk.h>
#include <volk/qa_volk.h>


int main(int argc, char* argv[])
{
    if (argc < 2){
        std::cerr << "Missing required argument 'kernel name'!\n";
        std::exit(EXIT_FAILURE);
        // return 2; // Command line usage error: https://stackoverflow.com/a/40484670
    }
    for (int i = 0; i < argc; i++) {
        fmt::print("{}\t{}\n", i, std::string(argv[i]));
    }

    const auto kernel_name = std::string(argv[1]);
    const unsigned vector_length = (argc > 2) ? std::stoi(std::string(argv[2])) : 131071;
    const unsigned iterations = (argc > 3) ? std::stoi(std::string(argv[3])) : 1;

    fmt::print("Run VOLK QA '{}'(it={}, vlen={})\n", kernel_name, iterations, vector_length);
    auto tests = initialize_test_list();
    for (auto& t: tests){
        auto archlist = t->get_arch_list(t->get_function_description());
        fmt::print("{} -> {}\n", t->name(), archlist);
        t->set_vector_length(vector_length);
        for(const auto& arch: archlist){
            fmt::print("\t{}\n", arch);
            t->execute(vector_length, arch);
        }
        for(const auto& arch: archlist){
            auto res = t->check_results(arch);
            fmt::print("\t{} correct: \n", arch, res);
        }
    }

    // bool qa_ret_val = 0;

    // float def_tol = 1e-6;
    // lv_32fc_t def_scalar = 327.0;
    // bool def_benchmark_mode = true;
    // std::string def_kernel_regex = "";

    // volk_test_params_t test_params(
    //     def_tol, def_scalar, def_vlen, def_iter, def_benchmark_mode, def_kernel_regex);
    // std::vector<volk_test_case_t> test_cases = init_test_list(test_params);
    // std::vector<volk_test_results_t> results;

        // const auto kernel_name = std::string(argv[1]);
        // for (unsigned int ii = 0; ii < test_cases.size(); ++ii) {
        //     if (kernel_name == test_cases[ii].name()) {
        //         volk_test_case_t test_case = test_cases[ii];
        //         return run_volk_tests(test_case.desc(),
        //                               test_case.kernel_ptr(),
        //                               test_case.name(),
        //                               test_case.test_parameters(),
        //                               &results,
        //                               test_case.puppet_master_name());
        //     }
        // }
        // std::cerr << "Did not run a test for kernel: " << kernel_name << " !"
        //           << std::endl;
        // return 0;



    // return qa_ret_val;
    std::exit(EXIT_SUCCESS);
}
