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

#ifndef INCLUDED_QA_VOLK_H
#define INCLUDED_QA_VOLK_H


#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <memory>
#include <map>


class volk_qa_test
{
private:
    std::string kernel_name;

protected:
    std::vector<std::string> archs;

public:
    volk_qa_test(std::string name = std::string("prototype"));
    ~volk_qa_test() = default;

    std::string name() { return kernel_name; }
    virtual void set_vector_length(const unsigned vector_length) = 0;
    virtual volk_func_desc_t get_function_description() = 0;
    virtual void execute(const unsigned vector_length, const std::string& arch) = 0;
    virtual bool check_results(const std::string& arch) = 0;
    std::vector<std::string> get_arch_list(volk_func_desc_t desc);
};

// clang-format off
%for kern in kernels:
%if 'num_points' in kern.args[-1][1]:

class qa_${kern.name} : public volk_qa_test 
{
private:
    struct data_vectors
    {
%for arg in kern.annotated_args[:-1]:
        %if 'vector' in arg[0]:
        volk::vector<${arg[1]}> ${arg[2]};
        %endif
%endfor
    };

    std::map<std::string, std::unique_ptr<data_vectors>> arch_vectors;
    
%for arg in kern.annotated_args[:-1]:
    %if 'vector' not in arg[0]:
    ${arg[1]} ${arg[2]};
    %endif
%endfor
public:
    qa_${kern.name}() : volk_qa_test(std::string("${kern.name}")) {};
    void set_vector_length(const unsigned vector_length) override;
    bool check_results(const std::string& arch) override;
    volk_func_desc_t get_function_description();
    void execute(const unsigned vector_length, const std::string& arch);
};
%endif
%endfor
//clang-format on


static std::vector<std::unique_ptr<volk_qa_test>> initialize_test_list()
{
    std::vector<std::unique_ptr<volk_qa_test>> tests = std::vector<std::unique_ptr<volk_qa_test>>();
    // clang-format off
%for kern in kernels:
%if 'num_points' in kern.args[-1][1]:
    tests.push_back(std::make_unique<qa_${kern.name}>());
%endif
% endfor
    // clang-format on
    return tests;
}


#endif /*INCLUDED_QA_VOLK_H*/
