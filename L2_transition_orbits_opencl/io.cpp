#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 6386 6385 6387 6001 6031 6011)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int read_data(const char* filename, int n, double* x)
{
	FILE* fp;
	int i, j;
	double* xcur;
	if ((fp = fopen(filename, "r")) == NULL)
	{
		fprintf(stderr, "Error: Cannot open file %s\n", filename);
		return 1;
	}
	for (i = 0; i < n; i++)
	{
		xcur = x + i * 6;
		for (j = 0; j < 6; j++)
		{
			fscanf(fp, "%lf", xcur + j);
		}
	}
	fclose(fp);
	return 0;
}

int text_write(int N, const char* filename, int nt, int n_threads, int n_bodies, double dt, double* x)
{
	FILE* out;
	double* xcur;
	double* xcurcur;
	if (!(out = fopen(filename, "w")))
	{
		fprintf(stderr, "Error: Cannot open file %s\n", filename);
		return 1;
	}
	for (int i = 0; i <= nt; i++)
	{
		xcur = x + i * n_threads * N;
		fprintf(out, "# t = %.10f\n", i * dt);
		for (int j = 0; j < n_bodies; j++)
		{
			xcurcur = xcur + N * j;
			for (int k = 0; k < N; k++)
			{
				fprintf(out, "%.16lf ", xcurcur[k]);
			}
			fprintf(out, "\n");
		}
		fprintf(out, "\n");
	}
	fclose(out);
}

int is_big_endian(void)
{
	uint64_t i = 0x0102030405060708;
	uint8_t* ptr = (uint8_t*)(&i);
	return ptr[0] == (uint8_t)1;
}

double change_byte_order(double value)
{
	uint8_t* ptr = (uint8_t*)(&value);
	double res = 0.0;
	uint8_t* res_ptr = (uint8_t*)(&res);
	res_ptr[7] = ptr[0];
	res_ptr[6] = ptr[1];
	res_ptr[5] = ptr[2];
	res_ptr[4] = ptr[3];
	res_ptr[3] = ptr[4];
	res_ptr[2] = ptr[5];
	res_ptr[1] = ptr[6];
	res_ptr[0] = ptr[7];
	return res;
}

int binary_write(int N, const char* filename, int nt, int n_threads, int n_bodies, double dt, double* x)
{
	FILE* out;
	double* xcur;
	double* xcurcur;
	char* end_data_block;
	int written_size;
	char* header;
	int end_data_size;
	double* write_buf;
	double* bufcur;
	if (!(out = fopen(filename, "wb")))
	{
		fprintf(stderr, "Error: Cannot open file %s\n", filename);
		return 1;
	}
	header = (char*)malloc((size_t)2880);
	sprintf(header, "SIMPLE  =                    T / conforms to FITS standard                      ");
	sprintf(header + 80, "BITPIX  =                  -64 / array data type                                ");
	sprintf(header + 160, "NAXIS   =                    3 / number of array dimensions                     ");
	sprintf(header + 240, "NAXIS1  = %20d                                                  ", N);
	sprintf(header + 320, "NAXIS2  = %20d                                                  ", n_bodies);
	sprintf(header + 400, "NAXIS3  = %20d                                                  ", nt + 1);
	sprintf(header + 480, "EXTEND  =                    T                                                  ");
	sprintf(header + 560, "T_STEP  = %20.16f                                                  ", dt);
	sprintf(header + 640, "END");
	for (int s = 643; s < 2880; s++)
	{
		header[s] = ' ';
	}
	fwrite(header, (size_t)1, (size_t)2880, out);
	free(header);
	written_size = 0;
	write_buf = (double*)malloc(n_bodies * N * sizeof(double));
	for (int i = 0; i <= nt; i++)
	{
		xcur = x + N * n_threads * i;
		if (!is_big_endian())
		{
			for (int j = 0; j < n_bodies; j++)
			{
				xcurcur = xcur + N * j;
				bufcur = write_buf + N * j;
				for (int k = 0; k < N; k++)
				{
					bufcur[k] = change_byte_order(xcurcur[k]);
				}
			}
			written_size += fwrite(write_buf, sizeof(double), (size_t)(n_bodies * N), out);
		}
		else
		{
			written_size += fwrite(xcur, sizeof(double), (size_t)(n_bodies * N), out);
		}
	}
	free(write_buf);
	if ((written_size * (int)sizeof(double)) % 2880 != 0)
	{
		end_data_size = 2880 - ((written_size + (int)sizeof(double)) % 2880);
		end_data_block = (char*)malloc(end_data_size);
		for (int i = 0; i < end_data_size; i++)
		{
			end_data_block[i] = '\0';
		}
		fwrite(end_data_block, (size_t)1, (size_t)end_data_size, out);
		free(end_data_block);
	}
	fclose(out);
	return 0;
}
