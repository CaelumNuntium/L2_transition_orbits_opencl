#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 6386 6385 6387 6001 6031 6011)

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "legendre.h"

double legendre_polynomial_value(int n, double x)
{
	if (n == 0)
	{
		return 1.0;
	}
	else if (n == 1)
	{
		return x;
	}
	else if (n == 2)
	{
		return 1.5 * x * x - 0.5;
	}
	return (2.0 * n - 1.0) * legendre_polynomial_value(n - 1, x) * x / n - (n - 1.0) * legendre_polynomial_value(n - 2, x) / n;
}

double shifted_legendre_polynomial_value(int n, double x)
{
	return legendre_polynomial_value(n, 2.0 * x - 1.0);
}

double legendre_polynomial_derivative_value(int n, double x)
{
	if (n == 0)
	{
		return 0.0;
	}
	else if (n == 1)
	{
		return 1.0;
	}
	else if (n == 2)
	{
		return 3.0 * x;
	}
	return n * legendre_polynomial_value(n - 1, x) + x * legendre_polynomial_derivative_value(n - 1, x);
}

double shifted_legendre_polynomial_derivative_value(int n, double x)
{
	return 2.0 * legendre_polynomial_derivative_value(n, 2.0 * x - 1.0);
}

double shifted_legendre_polynomial_antiderivative_value(int n, double x)
{
	return shifted_legendre_polynomial_value(n + 1, x) * 0.5 / n - (2.0 * x - 1.0) * shifted_legendre_polynomial_value(n, x) * 0.5 / n;
}

double legendre_polynomial_antiderivative_value(int n, double x)
{
	return 2.0 * shifted_legendre_polynomial_antiderivative_value(n, 0.5 * (x + 1.0));
}

void polynom_division(double* p, int n, double* q, int m, double* res, double* r)
{
	double* ptmp;
	int i, j;
	ptmp = (double*)malloc((n + 1) * sizeof(double));
	for (i = 0; i <= n; i++)
	{
		ptmp[i] = p[i];
	}
	for (i = n - m; i >= 0; i--)
	{
		res[i] = ptmp[i + m] / q[m];
		for (j = i + m; j >= i; j--)
		{
			ptmp[j] -= (res[i] * q[j - i]);
		}
	}
	for (i = m - 1; i >= 0; i--)
	{
		r[i] = ptmp[i];
	}
	free(ptmp);
}

double bernoulli(double* p, int n, double eps, int* ier)
{
	const double NMAX = 1000000;
	double* y;
	int i, j;
	double res, eps0;
	if (n == 1)
	{
		ier[0] = 0;
		return -p[0] / p[1];
	}
	y = (double*)malloc((n + 1) * sizeof(double));
	for (i = 0; i < n; i++)
	{
		y[i] = (double)(rand() % 100) / 100.0 + 0.01;
	}
	ier[0] = 1;
	for (i = 0; i < NMAX; i++)
	{
		y[n] = 0;
		for (j = 0; j < n; j++)
		{
			y[n] -= y[j] * p[j];
		}
		y[n] /= p[n];
		eps0 = (y[n] / y[n - 1]) - (y[n - 1] / y[n - 2]);
		if ((eps0 < eps) && (eps0 > -eps))
		{
			ier[0] = 0;
			break;
		}
		for (j = 0; j < n; j++)
		{
			y[j] = y[j + 1];
		}
	}
	res = y[n] / y[n - 1];
	free(y);
	return res;
}

void polynom_antiderivative(int n, double* p, double* res)
{
	int i;
	res[0] = 0.0;
	for (i = 0; i <= n; i++)
	{
		res[i + 1] = p[i] / (i + 1);
	}
}

void shifted_legendre_coeff(int n, double* coeff)
{
	int i;
	if (n % 2 == 0)
	{
		coeff[0] = 1.0;
	}
	else
	{
		coeff[0] = -1.0;
	}
	for (i = 1; i <= n; i++)
	{
		coeff[i] = -coeff[i - 1] * (n + i) * (n - i + 1) / i / i;
	}
}

int lobatto_approach(int n, double* coeff)
{
	const double bernoulli_eps = 1e-5;
	double* pp;
	double* p;
	double* tmp;
	double* p_copy;
	double q[2], r[2];
	int i, ier, j;
	pp = (double*)malloc(n * sizeof(double));
	shifted_legendre_coeff(n - 1, pp); // pp - coefficients of shifted Legendre polynomial; deg(pp) = n - 1
	p = (double*)malloc((n + 1) * sizeof(double));
	polynom_antiderivative(n - 1, pp, p); // p - integral of shifted Legendre polynomial
	p_copy = (double*)malloc((n + 1) * sizeof(double));
	for (i = 0; i <= n; i++) // make a copy for using in Bernoulli method
	{
		p_copy[i] = p[i];
	}
	q[1] = 1.0;
	tmp = (double*)malloc((n + 1) * sizeof(double));
	for (i = 0; i < n; i++) // find roots using Bernoulli method
	{
		coeff[n - i - 1] = bernoulli(p_copy, n - i, bernoulli_eps, &ier);
		q[0] = -coeff[n - i - 1];
		polynom_division(p_copy, n - i, q, 1, tmp, r);
		for (j = 0; j <= n - i; j++)
		{
			p_copy[j] = tmp[j];
		}
	}
	free(p_copy);
	free(pp);
	free(tmp);
	free(p);
	return ier;
}

int lobatto(int n, double* coeff)
{
	const double eps = 1e-16;
	double* aprch;
	int i, m, ier;
	aprch = (double*)malloc(n * sizeof(double));
	if (ier = lobatto_approach(n, aprch))
	{
		return ier;
	}
	for (i = 0; i < n; i++)
	{
		coeff[i] = aprch[i];
		aprch[i] = 0.0;
	}
	m = n - 1;
	for (i = 1; i < m; i++)
	{
		while (fabs(coeff[i] - aprch[i]) > eps)
		{
			aprch[i] = coeff[i];
			coeff[i] = aprch[i] - shifted_legendre_polynomial_antiderivative_value(n - 1, aprch[i]) / shifted_legendre_polynomial_value(n - 1, aprch[i]);
		}
	}
	coeff[0] = 0.0;
	coeff[m] = 1.0;
	free(aprch);
	return 0;
}
