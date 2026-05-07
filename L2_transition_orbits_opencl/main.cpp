#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 6386 6385 6387 6001 6031 6011 4018 4703)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include <sys/stat.h>
#include "config.h"
#include "legendre.h"
#include "gauss.h"
#include "select_device.h"
#include "utils_opencl.h"
#include "compile.h"
#include "io.h"

#define MAX_SOURCE_SIZE 1048576
#define MAX_COMPILER_OUTPUT_SIZE 1048576
#define CONFIG_NAME "set.config"
#define NUM_PARAMS 10

void report_error(int, const char*, int);

int main(int argc, char* argv[])
{
	int N;

	int i, j, jj;
	int ier;
	char cl_name[100], long_cl_name[100], eq_name[100], long_eq_name[100];
	FILE* fsource;
	FILE* fequation;
	char* source;
	char* equation;
	char* raw_source;
	size_t source_size, value_size, equation_size;
	char* device_name;
	cl_device_type device_type;
	cl_device_id device;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;

	unsigned int t0, t1, t3, t4, t5, t5_1, t6, t7, t8, t9, t10, t11, t12;

	const int num_params = NUM_PARAMS;
	const int collo_num_params = 1; // collo
	const char* params[NUM_PARAMS] = { "num_steps", "step", "M_steps", "N", "method", "output", "result_file", "data_file", "equation", "t0" };
	const char* collo_params[1] = { "log2S" }; // collo
	char* values[NUM_PARAMS];
	char* collo_values[1]; // collo
	int n, m, n_threads, nt, n0_bodies, n_bodies;
	double h, t_init;
	double* x0;
	double* xres;

	int binary = 0;
	char* result_name;
	char* datafile_name;

	int collo = 0;
	// Only for collocation method
	double* c;
	double* ct;
	double* ct_inv;
	double* ct_inv_t;
	double* bt;
	cl_mem c_mem_obj, ct_inv_t_mem_obj, bt_mem_obj;

	cl_mem x0_mem_obj, xres_mem_obj, n_mem_obj, m_mem_obj, h_mem_obj, t0_mem_obj;
	FILE* fptr;
	int log2s, s; // for collocation method

	size_t workrange_size[1], workgroup_size[1];

	t0 = clock();

	// Reading configuration file
	for (j = 0; j < num_params; j++)
	{
		values[j] = (char*)malloc(1000 * sizeof(char));
	}
	if (argc > 1)
	{
		ier = read_config(argv[1], num_params, params, values, NULL);
		if (ier)
		{
			fprintf(stderr, "Error: Cannot read configuration file %s\n", argv[1]);
			return 121;
		}
	}
	else
	{
		ier = read_config(CONFIG_NAME, num_params, params, values, NULL);
		if (ier)
		{
			fprintf(stderr, "Error: Cannot read configuration file %s\n", CONFIG_NAME);
			return 121;
		}
	}
	if (ier)
	{
		return 20;
	}
	n = atoi(values[0]);
	m = atoi(values[2]);
	nt = n / m;
	n0_bodies = atoi(values[3]);
	n_threads = (int)pow(2.0, (double)((int)log2(n0_bodies) + 1));
	if (n_threads < 64)
	{
		n_threads = 64;
	}
	n_bodies = n_threads;
	sscanf(values[1], "%lf", &h);
	sprintf(cl_name, "%s.cl", values[4]);
	sprintf(eq_name, "%s.cl", values[8]);
	sscanf(values[9], "%lf", &t_init);
	sprintf(long_cl_name, "../../OpenCL_Integrator/integrators/%s.cl", values[4]);
	sprintf(long_eq_name, "../../OpenCL_Integrator/equations/%s.cl", values[8]);
	if (!strcmp(values[5], "binary"))
	{
		binary = 1;
	}
	else if (strcmp(values[5], "text"))
	{
		fprintf(stderr, "Error: Invalid value: output = %s\n", values[5]);
		return 41;
	}
	result_name = (char*)malloc((strlen(values[6]) + 1) * sizeof(char));
	datafile_name = (char*)malloc((strlen(values[7]) + 1) * sizeof(char));
	sprintf(result_name, "%s", values[6]);
	sprintf(datafile_name, "%s", values[7]);

	if (!strcmp(values[4], "collo")) // for collocation method
	{
		collo = 1;
		for (j = 0; j < collo_num_params; j++)
		{
			collo_values[j] = (char*)malloc(1000 * sizeof(char));
		}
		if (argc > 1)
		{
			ier = read_config(argv[1], collo_num_params, collo_params, collo_values, NULL);
		}
		else
		{
			ier = read_config(CONFIG_NAME, collo_num_params, collo_params, collo_values, NULL);
		}
		if (ier)
		{
			return 30;
		}
		log2s = atoi(collo_values[0]);
		s = (int)(round(pow(2.0, log2s)));
		n_threads = (int)pow(2.0, (double)((int)log2(n0_bodies * s) + 1));
		if (n_threads < 64)
		{
			n_threads = 64;
		}
		n_bodies = n_threads / s;
		for (j = 0; j < collo_num_params; j++)
		{
			free(collo_values[j]);
		}
	}

	for (j = 0; j < num_params; j++)
	{
		free(values[j]);
	}
	t1 = clock();
	printf("Reading configuration file: %d ms\n\n", t1 - t0);
	printf("\n%d threads\n", n_threads);
	printf("num_steps = %d\nstep = %lf\nM_steps = %d\nN = %d\n", n, h, m, n0_bodies);
	printf("nt = %d\n\n", nt);
	printf("Kernel source: %s\n\n", cl_name);
	workrange_size[0] = n_threads;
	workgroup_size[0] = 64;

	// Preparing source
	source = (char*)malloc(2 * MAX_SOURCE_SIZE * sizeof(char));
	char def_str[100];
	if (collo)
	{
		snprintf(def_str, 99, "#define S %d", s);
	}
	else
	{
		def_str[0] = '\0';
	}
	if (ier = create_source(long_cl_name, long_eq_name, def_str, MAX_SOURCE_SIZE, (int)workrange_size[0], (int)workgroup_size[0], &N, source))
	{
		switch (ier)
		{
		case 10:
			fprintf(stderr, "Error: Failed to open file %s\n", cl_name);
			return 1000;
		case 11:
			fprintf(stderr, "Error: Failed to open file %s\n", eq_name);
			return 1001;
		case 2:
			fprintf(stderr, "Error: N must be defined in %s\n", eq_name);
			return 1020;
		}
	}

	// Preparing device
	if (argc > 1)
	{
		device = select_device(argv[1]);
	}
	else
	{
		device = select_device(CONFIG_NAME);
	}
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &value_size);
	device_name = (char*)malloc(value_size);
	clGetDeviceInfo(device, CL_DEVICE_NAME, value_size, device_name, NULL);
	printf("\nDevice \"%s\" will be used\n\n", device_name);
	clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	if (device_type != CL_DEVICE_TYPE_GPU)
	{
		fprintf(stderr, "Warning: Device %s is not GPU\n", device_name);
	}
	free(device_name);
	t3 = clock();
	printf("Collecting device information: %d ms\n\n", t3 - t1);

	// Creating context and command queue
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &ier); // Create OpenCL context
	report_error(ier, "clContextCreate", 3);
	command_queue = clCreateCommandQueue(context, device, 0, &ier); // Create command queue
	report_error(ier, "clCreateCommandQueue", 4);
	t4 = clock();

	// Preparing data
	if ((fptr = fopen(result_name, "r")) != NULL)
	{
		fprintf(stderr, "\nWarning: File %s already exists. It will be overwritten!\n\n", result_name);
		fclose(fptr);
		fptr = fopen(result_name, "w");
		fclose(fptr);
	}
	x0 = (double*)malloc(n_bodies * N * sizeof(double));
	if (read_data(datafile_name, n0_bodies, x0))
	{
		return 21;
	}
	free(datafile_name);
	for (j = n0_bodies; j < n_bodies; j++)
	{
		for (int jj = 0; jj < N; jj++)
		{
			x0[j * N + jj] = 0.0;
		}
	}
	xres = (double*)malloc(n_bodies * N * (nt + 1) * sizeof(double));
	t5 = clock();
	printf("Preparing data: %d ms\n\n", t5 - t4);

	// Preparing collocation method
	if (collo)
	{
		c = (double*)malloc(s * sizeof(double));
		ct = (double*)malloc(s * s * sizeof(double));
		ct_inv = (double*)malloc(s * s * sizeof(double));
		ct_inv_t = (double*)malloc(s * s * sizeof(double));
		bt = (double*)malloc(s * s * sizeof(double));
		lobatto(s, c);
		for (j = 0; j < s; j++)
		{
			for (jj = 1; jj <= s; jj++)
			{
				ct[(jj - 1) * s + j] = shifted_legendre_polynomial_derivative_value(jj, c[j]);
				bt[j * s + jj - 1] = shifted_legendre_polynomial_value(jj, c[j]) - (jj % 2 ? -1 : 1);
			}
		}
		invert(s, ct, ct_inv);
		transpose_matrix(s, s, ct_inv, ct_inv_t);
		t5_1 = clock();
		printf("Preparing collocation method: %d ms\n\n", t5_1 - t5);
	}

	t6 = clock();

	// Building program
	char* buffer = (char*)malloc(MAX_COMPILER_OUTPUT_SIZE * sizeof(char));
	program = build_program(source, context, device, buffer, MAX_COMPILER_OUTPUT_SIZE, &ier);
	fprintf(stderr, "\n%s\n", buffer);
	report_error(ier, "create_program_binary", 6);
	free(buffer);
	free(source);
	t7 = clock();
	printf("Building program: %d ms\n\n", t7 - t6);

	x0_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bodies * N * sizeof(double), NULL, &ier);
	report_error(ier, "clCreateBuffer", 5);
	xres_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bodies * N * (nt + 1) * sizeof(double), NULL, &ier);
	report_error(ier, "clCreateBuffer", 14);
	n_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ier);
	report_error(ier, "clCreateBuffer", 10);
	m_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ier);
	report_error(ier, "clCreateBuffer", 17);
	h_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double), NULL, &ier);
	report_error(ier, "clCreateBuffer", 18);
	t0_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double), NULL, &ier);
	report_error(ier, "clCreateBuffer", 1818);

	if (collo) // for collocation method
	{
		c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, s * sizeof(double), NULL, &ier);
		report_error(ier, "clCreateBuffer", 40);
		ct_inv_t_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, s * s * sizeof(double), NULL, &ier);
		report_error(ier, "clCreateBuffer", 41);
		bt_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, s * s * sizeof(double), NULL, &ier);
		report_error(ier, "clCreateBuffer", 51);
	}

	ier = clEnqueueWriteBuffer(command_queue, x0_mem_obj, CL_TRUE, 0, n_bodies * N * sizeof(double), x0, 0, NULL, NULL);
	report_error(ier, "clEnqueueWriteBuffer", 11);
	ier = clEnqueueWriteBuffer(command_queue, n_mem_obj, CL_TRUE, 0, sizeof(int), &n, 0, NULL, NULL);
	report_error(ier, "clEnqueueWriteBuffer", 15);
	ier = clEnqueueWriteBuffer(command_queue, m_mem_obj, CL_TRUE, 0, sizeof(int), &m, 0, NULL, NULL);
	report_error(ier, "clEnqueueWriteBuffer", 30);
	ier = clEnqueueWriteBuffer(command_queue, h_mem_obj, CL_TRUE, 0, sizeof(double), &h, 0, NULL, NULL);
	report_error(ier, "clEnqueueWriteBuffer", 31);
	ier = clEnqueueWriteBuffer(command_queue, t0_mem_obj, CL_TRUE, 0, sizeof(double), &t_init, 0, NULL, NULL);
	report_error(ier, "clEnqueueWriteBuffer", 3131);

	if (collo) //for collocation method
	{
		ier = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0, s * sizeof(double), c, 0, NULL, NULL);
		report_error(ier, "clEnqueueWriteBuffer", 42);
		ier = clEnqueueWriteBuffer(command_queue, ct_inv_t_mem_obj, CL_TRUE, 0, s * s * sizeof(double), ct_inv_t, 0, NULL, NULL);
		report_error(ier, "clEnqueueWriteBuffer", 43);
		ier = clEnqueueWriteBuffer(command_queue, bt_mem_obj, CL_TRUE, 0, s * s * sizeof(double), bt, 0, NULL, NULL);
		report_error(ier, "clEnqueueWriteBuffer", 52);
	}

	t8 = clock();
	// printf("Creating buffer in device memory and copying data: %d ms\n\n", t6 - t5);
	// printf("Copying data to the device memory: %d ms\n\n", t8 - t7);

	// Creating kernel
	kernel = clCreateKernel(program, "integrator", &ier);
	report_error(ier, "clCreateKernel", 7);
	// Setting kernel arguments
	ier = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&t0_mem_obj);
	report_error(ier, "clSetKernelArg", 888);
	ier = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&h_mem_obj);
	report_error(ier, "clSetKernelArg", 8);
	ier = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&n_mem_obj);
	report_error(ier, "clSetKernelArg", 16);
	ier = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&m_mem_obj);
	report_error(ier, "clSetKernelArg", 32);
	ier = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&x0_mem_obj);
	report_error(ier, "clSetKernelArg", 33);
	ier = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&xres_mem_obj);
	report_error(ier, "clSetKernelArg", 34);

	if (collo) // for collocation method
	{
		ier = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&c_mem_obj);
		report_error(ier, "clSwetKernelArg", 44);
		ier = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&ct_inv_t_mem_obj);
		report_error(ier, "clSetKernelArg", 45);
		ier = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&bt_mem_obj);
		report_error(ier, "clSetKernelArg", 53);
	}

	t9 = clock();
	// printf("Creating kernel and setting arguments: %d ms\n\n", t9 - t8);

	ier = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, workrange_size, workgroup_size, 0, NULL, NULL); // Create NDRange
	report_error(ier, "clEnqueueNDRangeKernel", 12);

	ier = clEnqueueReadBuffer(command_queue, xres_mem_obj, CL_TRUE, 0, n_bodies * N * (nt + 1) * sizeof(double), xres, 0, NULL, NULL); // Read result from device memory
	report_error(ier, "clEnqueueReadBuffer", 13);
	t10 = clock();
	printf("Calculation and reading result: %d ms\n\n", t10 - t9);

	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseMemObject(x0_mem_obj);
	clReleaseMemObject(n_mem_obj);
	clReleaseMemObject(m_mem_obj);
	clReleaseMemObject(h_mem_obj);
	clReleaseMemObject(t0_mem_obj);
	clReleaseMemObject(xres_mem_obj);
	t11 = clock();

	// Writing result
	if (binary)
	{
		binary_write(N, result_name, nt, n_bodies, n0_bodies, h * m, xres);
	}
	else
	{
		text_write(N, result_name, nt, n_bodies, n0_bodies, h * m, xres);
	}
	free(result_name);
	t12 = clock();
	printf("Writing result: %d ms\n\n", t12 - t11);

	printf("Performance: %f Msteps/s\n\n", (float)n_bodies * (float)n / (t10 - t9) / 1000.0F);

	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(x0);
	free(xres);
	return 0;
}

void report_error(int ier, const char* funcname, int exitcode)
{
	if (ier)
	{
		fprintf(stderr, "An error occured in %s: returned %d: \"%s\"\n", funcname, ier, error_string(ier));
		fprintf(stderr, "exit code: %d\n", exitcode);
		exit(exitcode);
	}
}
