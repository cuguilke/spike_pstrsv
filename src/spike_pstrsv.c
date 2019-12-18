/*
	Title           :spike_pstrsv.c
	Description     :The proposed algorithm for the parallel solution of sparse triangular linear systems
	Author          :Ilke Cugu
	Date Created    :23-06-2017
	Date Modified   :18-12-2019
	version         :4.0.4
*/

#include <omp.h>
#include "misc.h"
#include "matrix_partitioning.h"
#include "matrix_reordering.h"
#include "spike_pstrsv_globals.h"
#include "spike_pstrsv_ops.h"
#include "spike_pstrsv.h"

/* ----------------- spike_pstrsv Timers ----------------- */
double preprocessD_timer_pstrsv3 = 0;
double preprocessR_timer_pstrsv3 = 0;
double setDependents_timer_pstrsv3 = 0;
double preprocessS_timer_pstrsv3 = 0;
double memoryAlloc_timer_pstrsv3 = 0;
double preprocessMappings_timer_pstrsv3 = 0;
double produceCompressedR_timer_pstrsv3 = 0;
double produceS_timer_pstrsv3 = 0;
double produceReducedSystem_timer_pstrsv3 = 0;
double init_vars_pstrsv3 = 0;
double reduced_solver_timer_pstrsv3 = 0;
double computeY_timer[MAXTHREADCOUNT];
double waitBefore_timer_pstrsv3[MAXTHREADCOUNT];
double solveIndependent_timer[MAXTHREADCOUNT];
double subtract_Rx_timer[MAXTHREADCOUNT];
double waitAfter_timer_pstrsv3[MAXTHREADCOUNT];
double solveD2_timer[MAXTHREADCOUNT];
double solveD_timer[MAXTHREADCOUNT];
double temp_timer[MAXTHREADCOUNT];
/* ------------------------------------------------------- */

void spike_pstrsv_preproc(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo)
{
	// Initialize timers for detailed runtime analysis
	initTimers_PSTRSV3(nthreads);
	
	// Initialize the global variables
	initGlobals_PSTRSV3(n, ia[n], nthreads);

	// Matrix partitioning & thread assignments
	matrixPartitioning(n, nthreads);

	// Produce D matrix of DS Factorization 
	preprocessD(a, ia, ja, nthreads);

	// Point out all dependent elements of A
	setDependents_PSTRSV3(a, ia, ja, n, nthreads, uplo);

	// Determine the needed nonzeros of S and fractions of R partitions to compute them
	// to be able to produce the small system of dependent elements. Then, allocate 
	// memory for the S matrix of DS factorization and the compressed R matrix.
	// If the memory constraints are met, it calls separateDsections function of spike_pstrsv
	// to prepare all necessary matrix parts of the original matrix A.
	preprocessS_PSTRSV3(a, ia, ja, n, nthreads, uplo);

	// Produce R = A - D + D3
	preprocessR_PSTRSV3(a, ia, ja, nthreads, uplo);

	// Start generating compressed R matrix with memory allocation and partitioning of the mappings
	preprocessMappings_PSTRSV3(nthreads);

	// Produce compressed dense matrix of fractions of R partitions
	produceCompressedR_PSTRSV3(n, nthreads, uplo);
	
	// Produce fraction of S of DS Factorization in a compressed format
	produceS_PSTRSV3_Parallel(nthreads, uplo);
	
	// Produce the reduced system, if necessary
	produceReducedSystem_PSTRSV3(n, nthreads, uplo);

	// If logs are enabled print detailed information
	if(PSTRSV_VERBOSE) logPSTRSV3(nthreads);	
}

void _spike_pstrsv_preproc(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo)
{
	// Initialize timers for detailed runtime analysis
	initTimers_PSTRSV3(nthreads);
	
	// Initialize the global variables
	initGlobals_PSTRSV3(n, ia[n], nthreads);

	// Matrix partitioning & thread assignments
	matrixPartitioning(n, nthreads);

	// Produce D matrix of DS Factorization 
	temp_timer[0] = omp_get_wtime();
	preprocessD(a, ia, ja, nthreads);
	preprocessD_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];

	// Point out all dependent elements of A
	temp_timer[0] = omp_get_wtime();
	setDependents_PSTRSV3(a, ia, ja, n, nthreads, uplo);
	setDependents_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];

	// Determine the needed nonzeros of S and fractions of R partitions to compute them
	// to be able to produce the small system of dependent elements. Then, allocate 
	// memory for the S matrix of DS factorization and the compressed R matrix.
	// If the memory constraints are met, it calls separateDsections function of spike_pstrsv
	// to prepare all necessary matrix parts of the original matrix A.
	temp_timer[0] = omp_get_wtime();
	preprocessS_PSTRSV3(a, ia, ja, n, nthreads, uplo);
	preprocessS_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];

	// Produce R = A - D + D3
	temp_timer[0] = omp_get_wtime();
	preprocessR_PSTRSV3(a, ia, ja, nthreads, uplo);
	preprocessR_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];

	// Start generating compressed R matrix with memory allocation and partitioning of the mappings
	temp_timer[0] = omp_get_wtime();
	preprocessMappings_PSTRSV3(nthreads);
	preprocessMappings_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];

	// Produce compressed dense matrix of fractions of R partitions
	temp_timer[0] = omp_get_wtime();
	produceCompressedR_PSTRSV3(n, nthreads, uplo);
	produceCompressedR_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];
	
	// Produce fraction of S of DS Factorization in a compressed format
	temp_timer[0] = omp_get_wtime();
	produceS_PSTRSV3_Parallel(nthreads, uplo);
	produceS_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];
	
	// Produce the reduced system, if necessary
	temp_timer[0] = omp_get_wtime();
	produceReducedSystem_PSTRSV3(n, nthreads, uplo);
	produceReducedSystem_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];

	// If logs are enabled print detailed information
	if(PSTRSV_VERBOSE) logPSTRSV3(nthreads);	
}

void spike_pstrsv(const char uplo, int m, double *a, MKL_INT *ia, MKL_INT *ja, double *b, double *x, int nthreads)
{
	// Preprocessing & solving
	spike_pstrsv_preproc(a, ia, ja, m, nthreads, uplo);
	spike_pstrsv_solve(uplo, m, b, x, nthreads);
}

void spike_pstrsv_solve(const char uplo, int m, double *b, double *x, int nthreads)
{
	/*
		Parallel Sparse Triangular System Solver using partial DS Factorization to solve.
		Specifically, it does not compute the whole S matrix, but instead only a small 
		portion of it to calculate dependent elements. Then, sparse matrix-vector mults
		to clear out out-of-partition nonzeros to solve partitions in parallel.
	*/
	int th_id;
	const char transa = 'N';
	const char diag = 'N';		
	char matdescra[4] = {'G', 0, 0, 'C'};
	double alpha = -1, beta = 1;
	omp_set_num_threads(nthreads);

	// Solve Triangular System in Parallel
	#pragma omp parallel private(th_id) shared(alpha, beta, m, b, x, transa, diag, matdescra, nthreads)
	{
		// Init private variables
		th_id = omp_get_thread_num();
		int rowCount, colCount;
		int k = partition_numRows[th_id]; 				// rows per D partition
		int partition_start = partition_starts[th_id];
		int partition_end = partition_ends[th_id];
		int start_D = starts_D[th_id];
		int end_D = ends_D[th_id];
		int start_D0 = starts_D0[th_id];
		int end_D0 = ends_D0[th_id];
		int start_D1 = starts_D1[th_id];
		int end_D1 = ends_D1[th_id];
		int start_D2 = starts_D2[th_id];
		int end_D2 = ends_D2[th_id];
		int start_R = starts_R[th_id];
		int end_R = ends_R[th_id];
		int start_interval = start_intervals[th_id];	// relative row index with respect to the partition_start
		int end_interval = end_intervals[th_id];		// relative row index with respect to the partition_start
		char hasDependence = hasDependences[th_id];
		char hasReflection = hasReflections[th_id];
		char isOptimized = isOptimizeds[th_id];
	
		// Upper Sparse Triangular Linear System Solution with Partial DS Factorization
		if(uplo == 'U' || uplo == 'u')
		{
			if(isReducedSystemNecessary)
			{	
				if(hasReflection || isOptimized)
				{
					// Solve D1 * y = b to find right-hand side vector of the reduced system and store y in x array
					rowCount = k - start_interval;
					__udsol(rowCount, &x[partition_start + start_interval], &b[partition_start + start_interval], T2, &IT2[start_D1], JT2);
				}

				// Wait for all threads to finish producing partial S
				#pragma omp barrier
				
				// Sequential computation & solution of the reduced system
				produceReducedY_PSTRSV3(reducedX, x);
				#pragma omp single
				{
					__usol2(numRowsReducedSystem, reducedX, T, IT, JT);
				}
				updateX(reducedX, x);

				// First, subtract R * x from the right-hand side vector, if applicable
				if(hasDependence) // whether the thread has dependent elements (sparse R matrix)
					// Compute b = b - R * x	(if hasReflection -> R = R_old + D3)
					mkl_dcsrmv(&transa, &k, &m, &alpha, matdescra, &D[ID[start_R]], &JD[ID[start_R]], &ID[start_R], &ID[start_R + 1], x, &beta, &b[partition_start]);

				#pragma omp flush

				// Then, solve D2 * x = b, if applicable
				if(hasReflection || isOptimized)
				{
					// Solve D2 * x = b
					rowCount = end_interval;
					if(rowCount > 0)
						__udsol(rowCount, &x[partition_start], &b[partition_start], T2, &IT2[start_D2], JT2);
				} 
				else // no intersection means no need to split the D matrix
					// Solve D * x = b
					__udsol(k, &x[partition_start], &b[partition_start], D, &ID[start_D], JD);
			}
			else // perfect parallelism
				__udsol(k, &x[partition_start], &b[partition_start], D, &ID[start_D], JD);
		}
		else if(uplo == 'L' || uplo == 'l') // Lower Sparse Triangular Linear System Solution with Partial DS Factorization
		{
			//TODO: Implement lower triangular support
		}
	}
}

void _spike_pstrsv_solve(const char uplo, int m, double *b, double *x, int nthreads)
{
	/*
		Parallel Sparse Triangular System Solver using partial DS Factorization to solve.
		Specifically, it does not compute the whole S matrix, but instead only a small 
		portion of it to calculate dependent elements. Then, sparse matrix-vector mults
		to clear out out-of-partition nonzeros to solve partitions in parallel.
	*/
	temp_timer[0] = omp_get_wtime();
	int th_id;
	const char transa = 'N';
	const char diag = 'N';		
	char matdescra[4] = {'G', 0, 0, 'C'};
	double alpha = -1, beta = 1;
	omp_set_num_threads(nthreads);
	init_vars_pstrsv3 += omp_get_wtime() - temp_timer[0];

	// Solve Triangular System in Parallel
	#pragma omp parallel private(th_id) shared(alpha, beta, m, b, x, transa, diag, matdescra, nthreads)
	{
		// Init private variables
		th_id = omp_get_thread_num();
		int rowCount, colCount;
		int k = partition_numRows[th_id]; 				// rows per D partition
		int partition_start = partition_starts[th_id];
		int partition_end = partition_ends[th_id];
		int start_D = starts_D[th_id];
		int end_D = ends_D[th_id];
		int start_D0 = starts_D0[th_id];
		int end_D0 = ends_D0[th_id];
		int start_D1 = starts_D1[th_id];
		int end_D1 = ends_D1[th_id];
		int start_D2 = starts_D2[th_id];
		int end_D2 = ends_D2[th_id];
		int start_R = starts_R[th_id];
		int end_R = ends_R[th_id];
		int start_interval = start_intervals[th_id];	// relative row index with respect to the partition_start
		int end_interval = end_intervals[th_id];		// relative row index with respect to the partition_start
		char hasDependence = hasDependences[th_id];
		char hasReflection = hasReflections[th_id];
		char isOptimized = isOptimizeds[th_id];
	
		// Upper Sparse Triangular Linear System Solution with Partial DS Factorization
		if(uplo == 'U' || uplo == 'u')
		{
			if(isReducedSystemNecessary)
			{	
				if(hasReflection || isOptimized)
				{
					// Solve D1 * y = b to find right-hand side vector of the reduced system and store y in x array
					temp_timer[th_id] = omp_get_wtime();
					rowCount = k - start_interval;
					__udsol(rowCount, &x[partition_start + start_interval], &b[partition_start + start_interval], T2, &IT2[start_D1], JT2);
					computeY_timer[th_id] += omp_get_wtime() - temp_timer[th_id];
				}

				// Wait for all threads to finish producing partial S
				temp_timer[th_id] = omp_get_wtime();
				#pragma omp barrier
				waitBefore_timer_pstrsv3[th_id] += omp_get_wtime() - temp_timer[th_id];
				
				// Sequential computation & solution of the reduced system
				produceReducedY_PSTRSV3(reducedX, x);
				#pragma omp single
				{
					temp_timer[th_id] = omp_get_wtime();
					__usol2(numRowsReducedSystem, reducedX, T, IT, JT);
					reduced_solver_timer_pstrsv3 += omp_get_wtime() - temp_timer[th_id];
				}
				updateX(reducedX, x);

				// First, subtract R * x from the right-hand side vector, if applicable
				if(hasDependence) // whether the thread has dependent elements (sparse R matrix)
				{
					// Compute b = b - R * x	(if hasReflection -> R = R_old + D3)
					temp_timer[th_id] = omp_get_wtime();
					mkl_dcsrmv(&transa, &k, &m, &alpha, matdescra, &D[ID[start_R]], &JD[ID[start_R]], &ID[start_R], &ID[start_R + 1], x, &beta, &b[partition_start]);
					subtract_Rx_timer[th_id] += omp_get_wtime() - temp_timer[th_id];
				}

				temp_timer[th_id] = omp_get_wtime();
				#pragma omp flush
				waitAfter_timer_pstrsv3[th_id] += omp_get_wtime() - temp_timer[th_id];

				// Then, solve D2 * x = b, if applicable
				if(hasReflection || isOptimized)
				{
					// Solve D2 * x = b
					temp_timer[th_id] = omp_get_wtime();
					rowCount = end_interval;
					if(rowCount > 0)
						__udsol(rowCount, &x[partition_start], &b[partition_start], T2, &IT2[start_D2], JT2);
					solveD2_timer[th_id] += omp_get_wtime() - temp_timer[th_id];
				} 
				else // no intersection means no need to split the D matrix
				{
					// Solve D * x = b
					temp_timer[th_id] = omp_get_wtime();
					__udsol(k, &x[partition_start], &b[partition_start], D, &ID[start_D], JD);
					solveD_timer[th_id] += omp_get_wtime() - temp_timer[th_id];
				}
			}
			else // perfect parallelism
			{
				temp_timer[th_id] = omp_get_wtime();
				__udsol(k, &x[partition_start], &b[partition_start], D, &ID[start_D], JD);
				solveIndependent_timer[th_id] += omp_get_wtime() - temp_timer[th_id];
			}
		}
		else if(uplo == 'L' || uplo == 'l') // Lower Sparse Triangular Linear System Solution with Partial DS Factorization
		{
			//TODO: Implement lower triangular support
		}
	}
}