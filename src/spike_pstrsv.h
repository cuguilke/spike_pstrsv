#ifndef SPIKE_PSTRSV_H   /* Include guard */
#define SPIKE_PSTRSV_H

#include "mkl.h"
//----------------------------------------------------------------------------------------------//

void spike_pstrsv_preproc(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	For performance profiling of spike_pstrsv_preproc
*/
void _spike_pstrsv_preproc(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	Interface to provide preprocessing & solution bundle
*/
void spike_pstrsv(const char uplo, int m, double *a, MKL_INT *ia, MKL_INT *ja, double *b, double *x, int nthreads);

//----------------------------------------------------------------------------------------------//

/*
	Parallel Sparse Triangular System Solver using partial DS Factorization to solve.
	Specifically, it does not compute the whole S matrix, but instead only a small 
	portion of it to calculate dependent elements. Then, sparse matrix-vector mults
	to clear out out-of-partition nonzeros to solve partitions in parallel.
*/
void spike_pstrsv_solve(const char uplo, int m, double *b, double *x, int nthreads);

//----------------------------------------------------------------------------------------------//

/*
	For performance profiling of spike_pstrsv_solve
*/
void _spike_pstrsv_solve(const char uplo, int m, double *b, double *x, int nthreads);

//----------------------------------------------------------------------------------------------//
#endif