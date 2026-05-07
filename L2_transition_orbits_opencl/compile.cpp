#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 6386 6385 6387 6001 6031 6011 4018)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "utils_opencl.h"
#include "compile.h"

int create_source(const char* kernel_source_filename, const char* equation_source_filename, char* def_str, int max_source_size, int workrange_size, int workgroup_size, int* dimension, char* result_buffer)
{
	FILE* kernel_source;
	FILE* equation_source;
	char* kernel_str;
	char* eq_str;
	char* n_def_str_begin_ptr;
	char n_define_string[100];
	int kernel_size, eq_size;
	char ch;
	int i;
	if (!(kernel_source = fopen(kernel_source_filename, "r")))
	{
		return 10;
	}
	if (!(equation_source = fopen(equation_source_filename, "r")))
	{
		return 11;
	}
	kernel_str = (char*)malloc(max_source_size * sizeof(char));
	kernel_size = fread(kernel_str, sizeof(char), max_source_size, kernel_source);
	fread(kernel_str, sizeof(char), kernel_size, kernel_source);
	eq_str = (char*)malloc(max_source_size * sizeof(char));
	eq_size = fread(eq_str, sizeof(char), max_source_size, equation_source);
	fread(eq_str, sizeof(char), eq_size, equation_source);
	kernel_str[kernel_size - 1] = '\0';
	eq_str[eq_size - 1] = '\0';
	fclose(kernel_source);
	fclose(equation_source);
	if (!(n_def_str_begin_ptr = strstr(eq_str, "#define N")))
	{
		return 2;
	}
	ch = n_def_str_begin_ptr[0];
	i = 0;
	while (ch != '\n')
	{
		n_define_string[i] = ch;
		i++;
		ch = n_def_str_begin_ptr[i];
	}
	n_define_string[i] = '\0';
	sscanf(n_define_string, "#define N %d", dimension);
	snprintf(result_buffer, 2 * max_source_size, "#define WORKRANGE_SIZE %d\n#define WORKGROUP_SIZE %d\n%s\n%s\n%s\n", workrange_size, workgroup_size, def_str, eq_str, kernel_str);
	return 0;
}

cl_program build_program(char* source, cl_context context, cl_device_id device, char* output_buffer, int max_output_size, int* error_code)
{
	size_t source_size, len, binary_size;
	int ier, shft, full_shft;
	cl_program program;
	char* binary_buffer[1];
	cl_int ret = CL_SUCCESS;
	FILE* binary_out;
	full_shft = 0;
	source_size = strlen(source);
	program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_size, &ier);
	if (ier)
	{
		*error_code = ier;
		shft = snprintf(output_buffer + full_shft, max_output_size - full_shft, "Error in clCreateProgramWithSource: %s\n", error_string(ier));
		full_shft += shft;
		if (full_shft > max_output_size)
		{
			fprintf(stderr, "Warning: Buffer size is too small to write compiler output!\n");
		}
		return program;
	}
	ier = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (ier)
	{
		*error_code = ier;
		if (max_output_size - full_shft > 0)
		{
			shft = snprintf(output_buffer + full_shft, max_output_size - full_shft, "Error in clBuildProgram: %s\n\n", error_string(ier));
		}
		full_shft += shft;
	}
	ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
	if (len > max_output_size - full_shft)
	{
		fprintf(stderr, "Warning: Buffer size is too small to write compiler output!\n");
	}
	else
	{
		ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, output_buffer + full_shft, NULL);
	}
	if (ier)
	{
		return program;
	}
	*error_code = 0;
	return program;
}
