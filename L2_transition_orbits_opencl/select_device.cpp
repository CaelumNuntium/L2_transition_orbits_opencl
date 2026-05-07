#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 6386 6385 6387 6001 6031 6011 4018)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "config.h"
#include "utils_opencl.h"
#include "select_device.h"

cl_device_id select_device(const char* config)
{
	int i, j;
	cl_uint num_platforms = 0;
	cl_uint* num_devices;
	cl_platform_id* platforms;
	cl_device_id** devices;
	cl_device_type device_type;
	size_t value_size;
	char* platform_name;
	char* device_name;

	int n_params = 2;
	const char* params[2] = { "platform", "device" };
	char* values[2]; // platform name and device name from configuration file

	cl_platform_id selected_platform = (cl_platform_id)0;
	cl_device_id selected_device = (cl_device_id)0;
	int manual_select = 0, device_is_selected = 0;

	for (i = 0; i < n_params; i++)
	{
		values[i] = (char*)malloc(1000 * sizeof(char));
	}
	if (read_config(config, 2, params, values, NULL))
	{
		manual_select = 1;
		fprintf(stderr, "Warning: Cannot find platform name and device name in the configuration file. Please, select manually.\n");
	}
	clGetPlatformIDs(0, NULL, &num_platforms); // Check out how many platforms are there
	printf("Platforms and devices:\n");
	platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
	devices = (cl_device_id**)malloc(num_platforms * sizeof(cl_device_id*));
	num_devices = (cl_uint*)malloc(num_platforms * sizeof(cl_uint));
	clGetPlatformIDs(num_platforms, platforms, NULL); // Get all platform ids
	for (i = 0; i < num_platforms; i++)
	{
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &value_size);
		platform_name = (char*)malloc(value_size * sizeof(char));
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, value_size, platform_name, NULL); // Get platform name
		if (manual_select)
		{
			printf("%d) %s\n", i, platform_name);
		}
		if (strstr(platform_name, values[0]) != NULL)
		{
			selected_platform = platforms[i];
		}
		free(platform_name);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, num_devices + i); // Check out how many devices there are on each platform
		devices[i] = (cl_device_id*)malloc(num_devices[i] * sizeof(cl_device_id));
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices[i], devices[i], NULL); // Get all device ids
		for (j = 0; j < num_devices[i]; j++)
		{
			clGetDeviceInfo(devices[i][j], CL_DEVICE_NAME, 0, NULL, &value_size);
			device_name = (char*)malloc(value_size);
			clGetDeviceInfo(devices[i][j], CL_DEVICE_NAME, value_size, device_name, NULL); // Get device name
			if (selected_platform == platforms[i] && strstr(device_name, values[1]) != NULL)
			{
				selected_device = devices[i][j];
				device_is_selected = 1;
			}
			clGetDeviceInfo(devices[i][j], CL_DEVICE_TYPE, sizeof(cl_device_type), (void*)&device_type, NULL); // Get device type
			if (manual_select)
			{
				printf("    %d. %s (%s)\n", j, device_name, device_type_string(device_type));
			}
			free(device_name);
		}
	}
	if (!device_is_selected && !manual_select)
	{
		fprintf(stderr, "Warning: Cannot find device %s on platform %s. Please, select device manually.\n", values[1], values[0]);
		manual_select = 1;
	}
	for (i = 0; i < n_params; i++)
	{
		free(values[i]);
	}
	while (manual_select)
	{
		int spn, sdn;
		printf("\nPlatform: ");
		scanf("%d", &spn);
		printf("\nDevice: ");
		scanf("%d", &sdn);
		if (spn >= 0 && spn < num_platforms && sdn >= 0 && sdn < num_devices[spn])
		{
			manual_select = 0;
			selected_device = devices[spn][sdn];
		}
		clGetPlatformInfo(platforms[spn], CL_PLATFORM_NAME, 0, NULL, &value_size);
		values[0] = (char*)malloc(value_size);
		clGetPlatformInfo(platforms[spn], CL_PLATFORM_NAME, value_size, values[0], NULL);
		clGetDeviceInfo(devices[spn][sdn], CL_DEVICE_NAME, 0, NULL, &value_size);
		values[1] = (char*)malloc(value_size);
		clGetDeviceInfo(devices[spn][sdn], CL_DEVICE_TYPE, sizeof(cl_device_type), (void*)&device_type, NULL);
		rewrite_config(config, 2, params, (const char**)values);
		for (i = 0; i < n_params; i++)
		{
			free(values[i]);
		}
	}
	free(platforms);
	for (int i = 0; i < num_platforms; i++)
	{
		free(devices[i]);
	}
	free(devices);
	return selected_device;
}
