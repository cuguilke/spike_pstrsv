/*
	Title           :runtime_analysis.c
	Description     :This tester is written to analyze computational cost of parallel sparse triangular linear system solvers in detail
	Author          :Ilke Cugu
	Date Created    :29-10-2016
	Date Modified   :18-12-2019
	version         :5.0
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <metis.h>
#include <math.h>
#include "mkl.h"
#include "misc.h"
#include "spike_pstrsv.h"
#include "spike_pstrsv_globals.h"

char ENABLE_MKL_PSTRSV = 1;

double totalPreprocessing_timer_pstrsv3 = 0;
double totalPreprocessing_timer_MKL = 0;

typedef enum { false, true } bool;

void sparseMatrixVectorMult(int m, double *a, MKL_INT *ia, MKL_INT *ja, double *x, double *b)
{
	double alpha = 1, beta = 1;
	const char transa = 'N';
	char matdescra[6] = {'G', 0, 0, 'C', 0, 0};
	mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, a, ja, ia, &ia[1], x, &beta, b);
}

void udsol_(int n, double *x, double *y, double *A, MKL_INT *JA, MKL_INT *IA)
{
	// Upper triangular solver: U*x = y
	double t;
	int k, j;
	x[n-1] = y[n-1] / A[IA[n-1]];
	for(k=n-2;k>=0;k--)
	{
		t = y[k];
		for(j=IA[k]+1;j<IA[k+1];j++)
			t -= A[j]*x[JA[j]];
		x[k] = t / A[IA[k]];
	}
}

void ldsol_(int n, double *x, double *y, double *A, MKL_INT *JA, MKL_INT *IA)
{
	// Lower triangular solver: L*x = y
	double t;
	int k, j;
	x[0] = y[0] / A[IA[0]];
	for(k=1;k<n;k++)
	{
		t = y[k];
		for(j=IA[k];j<IA[k+1]-1;j++)
			t -= A[j]*x[JA[j]];
		x[k] = t / A[IA[k+1]-1];
	}
}

void IntelMKL_STRSV(const char uplo, int m, double *a, MKL_INT *ia, MKL_INT *ja, double *b, double *x)
{
	/*
		From Intel MKL Sparse BLAS
		mkl_cspblas_dcsrtrsv(&uplo, &transa, &diag, &m, a, ia, ja, x, y);
		for double precision triangular system solution.
	*/
	const char transa = 'N';
	const char diag = 'N';
	mkl_cspblas_dcsrtrsv(&uplo, &transa, &diag, &m, a, ia, ja, b, x);
}

void IntelMKL_pSTRSV(struct matrix_descr descrA, sparse_matrix_t csrA, double *b, double *x)
{
	/*
		From Intel MKL Sparse BLAS Inspecter Executor
		double precision parallel triangular system solution.
	*/
	double alpha = 1;
	mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA, b, x);
}

void SPARSKIT_STRSV(const char uplo, int m, double *a, MKL_INT *ia, MKL_INT *ja, double *b, double *x)
{
	/*
		SPARSKIT Library
		Sparse Triangular System Solve
	*/
	if(uplo == 'u' || uplo == 'U')
		udsol_(m, x, b, a, ja, ia);
	else
		ldsol_(m, x, b, a, ja, ia);
}

double calc2Norm(double *x_real, double *x, int m)
{
	int i;
	double norm_2 = 0;
	for(i=0;i<m;i++)
		norm_2 += (x_real[i] - x[i])*(x_real[i] - x[i]);
	return sqrt(norm_2);
}

void log_(int id, int problemid, double norm)
{
	/*
		id: 0 -> Intel MKL dcsrtrsv (sequential)
		id: 1 -> Intel MKL dcsrtrsv (parallel)
		id: 2 -> SPARSKIT strsv
		id: 3 -> Proposed Parallel Solver
		problemid: 0 -> 2-norm != 0
	*/
	switch(problemid)
	{
		case 0:
			switch(id)
			{
				case 0:
					printf("!!!Intel MKL dcsrtrsv (sequential) 2-norm: %lf != 0!!!\n", norm);
					break;
				case 1:
					printf("!!!Intel MKL dcsrtrsv (parallel) 2-norm: %lf != 0!!!\n", norm);
					break;
				case 2:
					printf("!!!SPARSKIT udsol 2-norm: %lf != 0!!!\n", norm);
					break;
				case 3:
					printf("!!!spike_pstrsv 2-norm: %lf != 0!!!\n", norm);
					break;
			}
			break;
	}
}

void printRuntimeAnalysis(int nthreads, double total)
{
	// spike_pstrsv variables
	double avg_computePartialY = 0;
	double avg_waitBefore = 0;
	double avg_solveIndependent = 0;
	double avg_subtract_Rx = 0;
	double avg_waitAfter = 0;
	double avg_solveD2 = 0;
	double avg_solveD = 0;
	int i, count = 0;

	for(i=0;i<nthreads;i++)
	{
		if(computeY_timer[i] > 0)
		{
			avg_computePartialY += computeY_timer[i];
			count++;
		}
	}
	avg_computePartialY /= count;

	count = 0;
	for(i=0;i<nthreads;i++)
	{
		if(waitBefore_timer_pstrsv3[i] > 0)
		{
			avg_waitBefore += waitBefore_timer_pstrsv3[i];
			count++;
		}
	}
	avg_waitBefore /= count;

	count = 0;
	for(i=0;i<nthreads;i++)
	{
		if(solveIndependent_timer[i] > 0)
		{
			avg_solveIndependent += solveIndependent_timer[i];
			count++;
		}
	}
	avg_solveIndependent /= count;

	count = 0;
	for(i=0;i<nthreads;i++)
	{
		if(subtract_Rx_timer[i] > 0)
		{
			avg_subtract_Rx += subtract_Rx_timer[i];
			count++;
		}
	}
	avg_subtract_Rx /= count;

	count = 0;
	for(i=0;i<nthreads;i++)
	{
		if(waitAfter_timer_pstrsv3[i] > 0)
		{
			avg_waitAfter += waitAfter_timer_pstrsv3[i];
			count++;
		}
	}
	avg_waitAfter /= count;

	count = 0;
	for(i=0;i<nthreads;i++)
	{
		if(solveD2_timer[i] > 0)
		{
			avg_solveD2 += solveD2_timer[i];
			count++;
		}
	}
	avg_solveD2 /= count;

	count = 0;
	for(i=0;i<nthreads;i++)
	{
		if(solveD_timer[i] > 0)
		{
			avg_solveD += solveD_timer[i];
			count++;
		}
	}
	avg_solveD /= count;

	printf("--------------------------------------------------------\n");
	printf("|   		Runtime Analysis of spike_pstrsv   			\n");
	printf("|                                          			 	\n");
	printf("| Total preprocessing MKL runtime: %lf ms				\n", totalPreprocessing_timer_MKL * 1000);
	printf("| Total preprocessing pSTRSV runtime: %lf ms			\n", totalPreprocessing_timer_pstrsv3 * 1000);
	printf("| preprocessD runtime: %lf ms							\n", preprocessD_timer_pstrsv3 * 1000);
	printf("| preprocessR runtime: %lf ms							\n", preprocessR_timer_pstrsv3 * 1000);
	printf("| setDependents runtime: %lf ms							\n", setDependents_timer_pstrsv3 * 1000);
	printf("| preprocessS runtime: %lf ms							\n", preprocessS_timer_pstrsv3 * 1000);
	printf("| memoryAlloc runtime: %lf ms							\n", memoryAlloc_timer_pstrsv3 * 1000);
	printf("| preprocessMappings runtime: %lf ms					\n", preprocessMappings_timer_pstrsv3 * 1000);
	printf("| produceCompressedR runtime: %lf ms					\n", produceCompressedR_timer_pstrsv3 * 1000);
	printf("| produceS runtime: %lf ms								\n", produceS_timer_pstrsv3 * 1000);
	printf("| produceReducedSystem runtime: %lf ms					\n", produceReducedSystem_timer_pstrsv3 * 1000);
	printf("|                                          			 	\n");
	printf("--------------------------------------------------------\n");
	printf("|                                          			 	\n");
	printf("| Total runtime: %lf ms                    			 	\n", total);
	printf("| initSolverVariables runtime: %lf ms      	    	 	\n", init_vars_pstrsv3);
	printf("| Avg. computePartialY runtime: %lf ms      	    	\n", avg_computePartialY);
	printf("| Avg. waitBefore runtime: %lf ms         				\n", avg_waitBefore);
	printf("| Avg. solveIndependent runtime: %lf ms   				\n", avg_solveIndependent);
	printf("| Avg. reducedSystemSolver runtime: %lf ms   			\n", reduced_solver_timer_pstrsv3);
	printf("| Avg. subtract_Rx runtime: %lf ms 						\n", avg_subtract_Rx);
	printf("| Avg. waitAfter runtime: %lf ms         				\n", avg_waitAfter);
	printf("| Avg. solveD2 runtime: %lf ms 							\n", avg_solveD2);
	printf("| Avg. solveD runtime: %lf ms 							\n", avg_solveD);
	printf("|                                          				\n");
	printf("--------------------------------------------------------\n");
}

int main(int argc, char **argv)
{
	double *x; // Solution vector
	double *b; // Right-hand side vector 
	double *a; // Array containing non-zero elements of the matrix A.
	MKL_INT *ia; // Array of length m+1, containing indices of elements in the array a.
	MKL_INT *ja; // Array containing the column indices for each non-zero element of the matrix A.
	int i, t, m, m_, nthreads, iter_count = 1000;
	double start1, end1, start2, end2, start3, end3, start4, end4, elapsed_time, best_time1, best_time2, best_time3, best_time4, total_time1 = 0, total_time2 = 0, total_time3 = 0, total_time4 = 0, *b_copy, norm_2, norm_2_MKL;
	char *matrixFileName, *endptr, *mmname, *prefix, uplo, symm, runMETIS;
	if(argc < 6)
	{
		printf("Usage: runtimeAnalysis <{matrix_filename}.mtx> <thread count> <uplo{U,u:L,l}> <symmetric{Y,y:S,s:N,n}> <METIS usage{Y,y:N,n}>\n\n");
		printf("  <uplo>      : {U,u} = upper triangle, {L,l} = lower triangle\n");
		printf("  <symmetric> : {Y,y} = symmteric, {S,s} = pattern symmteric, {N,n} = unsymmetric \n\n");
		printf("Important!: One must set the stack size as large as (at least 2GB) possible beforehand\n");
		return 0;
	}

	matrixFileName = argv[1];
	uplo = toupper(argv[3][0]);
	symm = argv[4][0];
	runMETIS = argv[5][0];
	nthreads = strtol(argv[2], &endptr, 10);
	if(*endptr)
		printf("Conversion error, thread count is problematic. Non-convertible part: %s", endptr);
	else
	{
		// Preparation Part
		if(runMETIS == 'Y' || runMETIS == 'y')
			matrixMarketToCSR(matrixFileName, &a, &ia, &ja, &m, 'N', symm);
		else
			matrixMarketToCSR(matrixFileName, &a, &ia, &ja, &m, uplo, symm);
		m_ = m + 8 - (m % 8);
		x = (double*)mkl_malloc(sizeof(double)*m_, 64);
		b = (double*)mkl_malloc(sizeof(double)*m_, 64);
		b_copy = (double*)mkl_malloc(sizeof(double)*m_, 64);
		if(runMETIS == 'Y' || runMETIS == 'y')
		{
			matrixReordering(&a, &ia, &ja, m, nthreads, uplo, symm, matrixFileName);
			switch(toupper(symm))
			{
				case 'Y': prefix = "/home/cuguilke/Desktop/Matrices/Literature/METIS/Symmetric/"; break;
				case 'S': prefix = "/home/cuguilke/Desktop/Matrices/Literature/METIS/PatternSymmetric/"; break;
				case 'N': prefix = "/home/cuguilke/Desktop/Matrices/Literature/METIS/Unsymmetric/"; break;
			}
			mmname = strconcat(prefix, strconcat(pathToMatrixName(matrixFileName), "_METIS"));
			//CSRtoMatrixMarket(mmname, m, a, ia, ja);
			printf("Matrix reodering is completed.\n");
		}
		mkl_set_num_threads(nthreads);

		//START - PRODUCE X FOR TESTING and ASSIGN B
		double *x_real;
		x_real = (double*)mkl_malloc(sizeof(double)*m_, 64);
		for(i=0;i<m;i++)
			x_real[i] = ((rand() % 71) + 1) + ((double)(i % 1000) / 10.0);
		memset(b, 0, sizeof(double)*m);
		sparseMatrixVectorMult(m, a, ia, ja, x_real, b);
		cblas_dcopy(m, b, 1, b_copy, 1); // since parallel solvers modify b vector
		//END - PRODUCE X FOR TESTING and ASSIGN B
		
		/* --- Direct Solution with Intel MKL Sparse BLAS (sequential) --- */
		best_time1 = INF;
		for(i=0;i<=iter_count;i++)
		{
			memset(x, 0, sizeof(double)*m);
			if(i == 0)
				IntelMKL_STRSV(uplo, m, a, ia, ja, b, x); // WARM-UP
			else
			{
				start1 = omp_get_wtime();
				IntelMKL_STRSV(uplo, m, a, ia, ja, b, x);
				end1 = omp_get_wtime();
				elapsed_time = end1 - start1;
				total_time1 += elapsed_time;
				best_time1 = (elapsed_time < best_time1) ? elapsed_time : best_time1;
			}
		}
		norm_2 = calc2Norm(x_real, x, m);
		if(norm_2 > 0.0001) log_(0, 0, norm_2);
		/* --------------------------------------------------------------- */

		/* --- Direct Solution with Intel MKL Sparse BLAS (parallel) --- */
		best_time2 = INF;
		sparse_matrix_t csrA;
		struct matrix_descr descrA;
		descrA.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
		if(uplo == 'u' || uplo == 'U')
			descrA.mode = SPARSE_FILL_MODE_UPPER;
		else
			descrA.mode = SPARSE_FILL_MODE_LOWER;
		descrA.diag = SPARSE_DIAG_NON_UNIT;

		if(ENABLE_MKL_PSTRSV) 
		{
			// Build internal model
			start2 = omp_get_wtime();
			mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, m, m, ia, &ia[1], ja, a);

			// Preprocessing for Intel MKL Sparse BLAS Inspector Executor
			mkl_sparse_set_sv_hint(csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA, iter_count);
			mkl_sparse_optimize(csrA);
			totalPreprocessing_timer_MKL = omp_get_wtime() - start2;

			// Triangular solve
			for(i=0;i<=iter_count;i++)
			{
				memset(x, 0, sizeof(double)*m);
				if(i == 0)
					IntelMKL_pSTRSV(descrA, csrA, b, x); // WARM-UP
				else
				{
					start2 = omp_get_wtime();
					IntelMKL_pSTRSV(descrA, csrA, b, x);
					end2 = omp_get_wtime();
					elapsed_time = end2 - start2;
					total_time2 += elapsed_time;
					best_time2 = (elapsed_time < best_time2) ? elapsed_time : best_time2;
				}
			}
			norm_2 = calc2Norm(x_real, x, m);
			if(norm_2 > 0.0001) log_(1, 0, norm_2);
			norm_2_MKL = norm_2;
		}
		else
			best_time2 = 0;
		/* ------------------------------------------------------------- */

		/* --- Direct Solution with SPARSKIT --- */
		best_time3 = INF;
		for(i=0;i<=iter_count;i++)
		{
			memset(x, 0, sizeof(double)*m);
			if(i == 0)
				SPARSKIT_STRSV(uplo, m, a, ia, ja, b, x); // WARM-UP
			else
			{
				start3 = omp_get_wtime();
				SPARSKIT_STRSV(uplo, m, a, ia, ja, b, x);
				end3 = omp_get_wtime();
				elapsed_time = end3 - start3;
				total_time3 += elapsed_time;
				best_time3 = (elapsed_time < best_time3) ? elapsed_time : best_time3;
			}
		}
		norm_2 = calc2Norm(x_real, x, m);
		if(norm_2 > 0.0001) log_(2, 0, norm_2);
		/* ------------------------------------------------------------- */

		/* --- Parallel Solution of Sparse Triangular Linear System --- */
		best_time4 = INF;
		PSTRSV_VERBOSE = 1;
		mkl_set_num_threads(1);
		start4 = omp_get_wtime();
		_spike_pstrsv_preproc(a, ia, ja, m, nthreads, uplo);
		totalPreprocessing_timer_pstrsv3 = omp_get_wtime() - start4;
		for(i=0;i<=iter_count;i++)
		{
			memset(x, 0, sizeof(double)*m);
			cblas_dcopy(m, b, 1, b_copy, 1);

			if(i == 0)
				_spike_pstrsv_solve(uplo, m, b_copy, x, nthreads); // WARM-UP
			else
			{
				start4 = omp_get_wtime();
				_spike_pstrsv_solve(uplo, m, b_copy, x, nthreads);
				end4 = omp_get_wtime();
				elapsed_time = end4 - start4;
				total_time4 += elapsed_time;
				best_time4 = (elapsed_time < best_time4) ? elapsed_time : best_time4;
			}
		}
		norm_2 = calc2Norm(x_real, x, m);
		if(norm_2 > 0.0001) log_(3, 0, norm_2);
		printRuntimeAnalysis(nthreads, total_time4);
		/* --------------------------------------------------------------- */

		//printf("MKL residual: %e\nPSTRSV residual: %e\n", norm_2_MKL, norm_2);
		printf( "MKL Seq. Triangular Solver \t| Min. runtime: %.4f ms \t| Avg. runtime: %.4f ms\n"  \
				"MKL Par. Triangular Solver \t| Min. runtime: %.4f ms \t| Avg. runtime: %.4f ms\n"  \
				"SPARSKIT Triangular Solver \t| Min. runtime: %.4f ms \t| Avg. runtime: %.4f ms\n"  \
				"Parallel Triangular Solver \t| Min. runtime: %.4f ms \t| Avg. runtime: %.4f ms\n", \
				best_time1*1000, total_time1, best_time2*1000, total_time2, best_time3*1000, total_time3, best_time4*1000, total_time4);

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
