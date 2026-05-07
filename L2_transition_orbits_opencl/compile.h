#ifndef __OPENCL_CL_H
#define __OPENCL_CL_H
#include <CL/cl.h>
#endif
int create_source(const char* kernel_source_filename, const char* equation_source_filename, char* def_str, int max_source_size, int workrange_size, int workgroup_size, int* dimension, char* result_buffer);
cl_program build_program(char* source, cl_context context, cl_device_id device, char* output_buffer, int max_output_size, int* error_code);
