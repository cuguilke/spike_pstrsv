/*
	Title           :example_usage.c
	Description     :Take a look at this code for a quick test drive
	Author          :Ilke Cugu
	Date Created    :18-12-2019
	Date Modified   :18-12-2019
	version         :1.0
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <metis.h>
#include <math.h>
#include "mkl.h"
#include "spike_pstrsv.h"

typedef enum { false, true } bool;

char SINGLE_CALL = 1;

void sparseMatrixVectorMult(int m, double *a, MKL_INT *ia, MKL_INT *ja, double *x, double *b)
{
	double alpha = 1, beta = 1;
	const char transa = 'N';
	char matdescra[6] = {'G', 0, 0, 'C', 0, 0};
	mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, a, ja, ia, &ia[1], x, &beta, b);
}

double calc2Norm(double *x_real, double *x, int m)
{
	int i;
	double norm_2 = 0;
	for(i=0;i<m;i++)
		norm_2 += (x_real[i] - x[i])*(x_real[i] - x[i]);
	return sqrt(norm_2);
}

int main(int argc, char **argv)
{
	double *x; // Solution vector
	double *b; // Right-hand side vector 
	double *a; // Array containing non-zero elements of the matrix A.
	MKL_INT *ia; // Array of length m+1, containing indices of elements in the array a.
	MKL_INT *ja; // Array containing the column indices for each non-zero element of the matrix A.
	int i, t, m, m_, nthreads;
	double *b_copy, norm_2;
	char *matrixFileName, *endptr, uplo, symm;

	if(argc < 5)
	{
		printf("Usage: exampleUsage <{matrix_filename}.mtx> <thread count> <uplo{U,u:L,l}> <symmetric{Y,y:S,s:N,n}>\n\n");
		printf("  <uplo>      : {U,u} = upper triangle, {L,l} = lower triangle\n");
		printf("  <symmetric> : {Y,y} = symmteric, {S,s} = pattern symmteric, {N,n} = unsymmetric \n\n");
		printf("Important!: One must set the stack size as large as (at least 2GB) possible beforehand\n");
		return 0;
	}

	matrixFileName = argv[1];
	uplo = toupper(argv[3][0]);
	symm = argv[4][0];
	nthreads = strtol(argv[2], &endptr, 10);

	if(*endptr)
		printf("Conversion error, thread count is problematic. Non-convertible part: %s", endptr);
	else
	{
		// Read the matrix from a MatrixMarket file
		matrixMarketToCSR(matrixFileName, &a, &ia, &ja, &m, uplo, symm);
		m_ = m + 8 - (m % 8);
		x = (double*)mkl_malloc(sizeof(double)*m_, 64);
		b = (double*)mkl_malloc(sizeof(double)*m_, 64);
		b_copy = (double*)mkl_malloc(sizeof(double)*m_, 64);

		// Test case preparation
		printf("Test case preparation...\n");
		double *x_real;
		x_real = (double*)mkl_malloc(sizeof(double)*m_, 64);
		for(i=0;i<m;i++)
			x_real[i] = ((rand() % 71) + 1) + ((double)(i % 1000) / 10.0);
		memset(b, 0, sizeof(double)*m);
		sparseMatrixVectorMult(m, a, ia, ja, x_real, b);
		cblas_dcopy(m, b, 1, b_copy, 1); // since spike_pstrsv modify b vector

		// Prepare the x & b vectors
		memset(x, 0, sizeof(double)*m);
		cblas_dcopy(m, b, 1, b_copy, 1);

		printf("spike_pstrsv is running...\n");
		if(SINGLE_CALL)
		{
			// Single call - it first runs the preprocessor then the solver
			spike_pstrsv(uplo, m, a, ia, ja, b, x, nthreads);
		}
		else
		{
			// Multiple call - for an iterative solver, call preprocessor once then run the solver multiple times with different right-hand side vectors
			spike_pstrsv_preproc(a, ia, ja, m, nthreads, uplo);
			spike_pstrsv_solve(uplo, m, b_copy, x, nthreads);
		}

		// Check if the solution is correct
		norm_2 = calc2Norm(x_real, x, m);

		printf("spike_pstrsv residual: %e\nDone.\n", norm_2);

		mkl_free(x);
		mkl_free(b);
		mkl_free(a);
		mkl_free(ia);
		mkl_free(ja);
		mkl_free(x_real);
		mkl_free(b_copy);
	}
	return 0;
}
