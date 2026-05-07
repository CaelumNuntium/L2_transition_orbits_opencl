#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 6386 6385 6387 6001 6031 6011)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int PRINT_WARNINGS = 0;

void gauss(int type, int n, double* a, double* b, double* x)
{
	double* aint;
	double* acur;
	double* aacur;
	int* e;
	int i, j, m, k, p, q;
	double a0, a1, eps, max, tmp;
	eps = 1e-7;
	m = n + 1;
	aint = (double*)malloc(m * n * sizeof(double));
	for (i = 0; i < n; i++)
	{
		acur = aint + i * m;
		aacur = a + i * n;
		for (j = 0; j < n; j++)
		{
			acur[j] = aacur[j];
		}
		acur[n] = b[i];
	}
	switch (type)
	{
	case 1:
		for (k = 0; k < n; k++)
		{
			aacur = aint + k * m;
			a0 = aacur[k];
			if (a0 < eps && a0 > -eps && PRINT_WARNINGS)
			{
				printf("WARNING: The leading element is close to 0!\n");
			}
			for (j = k; j < m; j++)
			{
				aacur[j] /= a0;
			}
			for (i = k + 1; i < n; i++)
			{
				acur = aint + i * m;
				a1 = acur[k];
				for (j = k; j < m; j++)
				{
					acur[j] -= aacur[j] * a1;
				}
			}
		}
		for (i = n - 1; i > -1; i--)
		{
			acur = aint + i * m;
			x[i] = acur[n];
			for (j = i + 1; j < n; j++)
			{
				x[i] -= acur[j] * x[j];
			}
		}
		break;
	case 2:
		for (k = 0; k < n; k++)
		{
			aacur = aint + k * m;
			a0 = aacur[k];
			if (a0 < eps && a0 > -eps && PRINT_WARNINGS)
			{
				printf("WARNING: The leading element is close to 0!\n");
			}
			for (j = k; j < m; j++)
			{
				aacur[j] /= a0;
			}
			for (i = 0; i < k; i++)
			{
				acur = aint + i * m;
				a1 = acur[k];
				for (j = k; j < m; j++)
				{
					acur[j] -= aacur[j] * a1;
				}
			}
			for (i = k + 1; i < n; i++)
			{
				acur = aint + i * m;
				a1 = acur[k];
				for (j = k; j < m; j++)
				{
					acur[j] -= aacur[j] * a1;
				}
			}
		}
		for (i = 0; i < n; i++)
		{
			x[i] = aint[i * m + n];
		}
		break;
	case 3:
		e = (int*)malloc(n * sizeof(int));
		for (k = 0; k < n; k++)
		{
			max = fabs(aint[k * m + k]);
			p = k;
			q = k;
			for (i = k; i < n; i++)
			{
				acur = aint + i * m;
				for (j = k; j < n; j++)
				{
					if (fabs(acur[j]) > max)
					{
						max = fabs(acur[j]);
						p = i;
						q = j;
					}
				}
			}
			e[k] = q;
			acur = aint + p * m;
			aacur = aint + k * m;
			for (i = 0; i < m; i++)
			{
				tmp = acur[i];
				acur[i] = aacur[i];
				aacur[i] = tmp;
			}
			for (i = 0; i < n; i++)
			{
				acur = aint + i * m;
				tmp = acur[q];
				acur[q] = acur[k];
				acur[k] = tmp;
			}
			a0 = aacur[k];
			for (j = k; j < m; j++)
			{
				aacur[j] /= a0;
			}
			for (i = 0; i < k; i++)
			{
				acur = aint + i * m;
				a1 = acur[k];
				for (j = k; j < m; j++)
				{
					acur[j] -= aacur[j] * a1;
				}
			}
			for (i = k + 1; i < n; i++)
			{
				acur = aint + i * m;
				a1 = acur[k];
				for (j = k; j < m; j++)
				{
					acur[j] -= aacur[j] * a1;
				}
			}
		}
		for (i = 0; i < n; i++)
		{
			x[i] = aint[i * m + n];
		}
		for (i = n - 1; i >= 0; i--)
		{
			tmp = x[i];
			x[i] = x[e[i]];
			x[e[i]] = tmp;
		}
		free(e);
		break;
	case 4:
		e = (int*)malloc(n * sizeof(int));
		for (k = 0; k < n; k++)
		{
			max = fabs(aint[k * m + k]);
			p = k;
			q = k;
			for (i = k; i < n; i++)
			{
				acur = aint + i * m;
				for (j = k; j < n; j++)
				{
					if (fabs(acur[j]) > max)
					{
						max = fabs(acur[j]);
						p = i;
						q = j;
					}
				}
			}
			e[k] = q;
			acur = aint + p * m;
			aacur = aint + k * m;
			for (i = 0; i < m; i++)
			{
				tmp = acur[i];
				acur[i] = aacur[i];
				aacur[i] = tmp;
			}
			for (i = 0; i < n; i++)
			{
				acur = aint + i * m;
				tmp = acur[q];
				acur[q] = acur[k];
				acur[k] = tmp;
			}
			a0 = aacur[k];
			for (j = k; j < m; j++)
			{
				aacur[j] /= a0;
			}
			for (i = k + 1; i < n; i++)
			{
				acur = aint + i * m;
				a1 = acur[k];
				for (j = k; j < m; j++)
				{
					acur[j] -= aacur[j] * a1;
				}
			}
		}
		for (i = n - 1; i > -1; i--)
		{
			acur = aint + i * m;
			x[i] = acur[n];
			for (j = i + 1; j < n; j++)
			{
				x[i] -= acur[j] * x[j];
			}
		}
		for (i = n - 1; i >= 0; i--)
		{
			tmp = x[i];
			x[i] = x[e[i]];
			x[e[i]] = tmp;
		}
		free(e);
		break;
	}
	free(aint);
}

void invert(int n, double* a, double* res)
{
	double* b;
	double* col;
	int i, j;
	b = (double*)malloc(n * sizeof(double));
	for (i = 0; i < n; i++)
	{
		b[i] = 0.0;
	}
	col = (double*)malloc(n * sizeof(double));
	for (i = 0; i < n; i++)
	{
		b[i] = 1.0;
		gauss(4, n, a, b, col);
		for (j = 0; j < n; j++)
		{
			res[j * n + i] = col[j];
		}
		b[i] = 0.0;
	}
	free(b);
	free(col);
}

void transpose_matrix(int rows, int cols, double* matrix, double* res)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			res[j * rows + i] = matrix[i * cols + j];
		}
	}
}
