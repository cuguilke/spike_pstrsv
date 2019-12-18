#ifndef SPIKE_PSTRSV_OPS_H   /* Include guard */
#define SPIKE_PSTRSV_OPS_H

#include "mkl.h"
//----------------------------------------------------------------------------------------------//

void initGlobals_PSTRSV3(int n, int nnz, int nthreads);

//----------------------------------------------------------------------------------------------//

void initTimers_PSTRSV3(int nthreads);

//----------------------------------------------------------------------------------------------//

void free_PSTRSV3();

//----------------------------------------------------------------------------------------------//

/*
	Provides a map of all dependents elements in A matrix
	by marking the nonzeros outside of the partition area 
	for each thread. Also indicates general dependency in
	the A matrix.
*/
void setDependents_PSTRSV3(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	Returns the row index of the nnz of R matrix at the bottom
	for the upper and at the top for the lower triangular case. 
*/
int getIntervalEnd(double *a, MKL_INT *ia, MKL_INT *ja, int start, int end, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	pSTRSV3 is built on the idea of computing only a fraction
	of the S matrix, so this function determines the start and 
	end points of the necessary parts of the S matrix. Then,
	it produces small triangular matrices that are used to 
	calculate S matrix fractions out of D partitions and stores
	them in T2, IT2, and JT2 arrays in CSR format.
	
	In short, the small system only needs intersection points'
	S matrix values, to be able to point them out, call this
	function before producing the small system.
*/
int preprocessS_PSTRSV3(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	Checks whether we can achieve better load-balance by also
	partitioning D_i matrices with no reflections that are waiting
	for the full D_i * x_i = b_i solution while other threads are
	solving the D1_i * x_i = b_i at the beginning of pSTRSV3.
*/
void optimizeLoadBalance(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	Separates D_section1 (D^{(m;b)}) & D_section2 (D^{(t;m)})
	Note1: when this functions is called D_section0 is already separated
	Note2: when this functions is called {starts|ends}_intervals are set
	Use the first part of T2, IT2, JT2 to store D_section0 (triangular)
	Use the middle part of T2, IT2, JT2 to store D_section1 (triangular)
	Use the rest of T2, IT2, JT2 to store D_section2 (triangular)
	D_section0 -> starts_D0 & ends_D0
	D_section1 -> starts_D1 & ends_D1
	D_section2 -> starts_D2 & ends_D2

	In detail:
	Separate D sections and store them as three matrices in T, IT, JT arrays:
		1) First part of the array holds the matrix resutled from consecutively
		placing the areas of particular D section intervals of each thread
		2) First-Middle part of the array holds the matrix resulted from placing
		D1[t] matrices for t = 0..nthreads-1 consecutively
		2) Second-Middle part of the array holds the matrix resulted from placing
		D2[t] matrices for t = 0..nthreads-1 consecutively
	For matrix operations, coressponding indices per computation are:
		1) Upper triangular:
			start_index = partition_starts[t]
			interval_start = start_index + start_intervals[t]
			interval_end = start_index + end_intervals[t]
			D0 * x = b 	-> T2[IT2[starts_D0[t]]] * x[interval_start] = b[interval_start]
			D1 * x = b	-> T2[IT2[starts_D1[t]]] * x[interval_end] 	 = b[interval_end]
			D2 * x = b 	-> T2[IT2[starts_D2[t]]] * x[start_index] 	 = b[start_index]
		2) Lower triangular:
			start_index = partition_start
			interval_start = start_index + start_intervals[t]
			D0 * x = b 	-> T2[IT2[starts_D0[t]]] * x[interval_start] = b[interval_start]
			D1 * x = b	-> T2[IT2[starts_D1[t]]] * x[start_index] 	 = b[start_index]
			D2 * x = b 	-> T2[IT2[starts_D2[t]]] * x[interval_start] = b[interval_start]
*/ 
void separateDsections_PSTRSV3(int nthreads, const char uplo, int rowD1);

//----------------------------------------------------------------------------------------------//

/*
	Computes start & end indices of each R matrix partition, and
	produces R matrix. It is stored in the leftover space at the
	end of D, ID, and JD arrays. It is guaranteed that the memory
	is enough to store R matrix iff A matrix does not require more
	rows than MAXSIZE & nonzeros than MAXNNZS.
	Call this function once, then the solver can be called n times.

	Update:
	R_new = R_old + D3
	Performance Improvement:
	Replace: b -= R_old * x & b -= D3 * x
	with:	 b -= R_new * x
	to make the necessary calculation in one function call
*/
void preprocessR_PSTRSV3(double *a, MKL_INT *ia, MKL_INT *ja, int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

void preprocessMappings_PSTRSV3(int nthreads);

//----------------------------------------------------------------------------------------------//

/*
	Since the matrix R is stored in CSR format,
	it does not waste memory space storing zeros.
	However, when it is used to produce S matrix,
	which is a dense matrix, there is a strong
	possibility to end up wasting lots of memory.
	Therefore, this routine compresses the sparse
	R matrix into a matrix where each column has
	at least one nonzero element. In addition, it
	only contains an interval of R partitions for
	each thread, and does not contain anything if
	the particular thread does not have any inter-
	section elements of partial S to be calculated.
*/
void produceCompressedR_PSTRSV3(int n, int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	Produces partial S of DS Factorization sequentially.
	(S = D'A -> DS = A)
	Since D partitions are triangular systems, 
	there is no need to calculate the inverse 
	of D sections. Instead, triangular solvers
	calculate dense S parts.
*/
void produceS_PSTRSV3(int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	Produces partial S of DS Factorization in parallel.
	(S = D'A -> DS = A)
	Since D partitions are triangular systems, 
	there is no need to calculate the inverse 
	of D sections. Instead, triangular solvers
	calculate dense S parts.
*/
void produceS_PSTRSV3_Parallel(int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	Produces the small system of dependent elements
	and stores it in T, IT, and JT with CSR format.
*/
void produceReducedSystem_PSTRSV3(int n, int nthreads, char const uplo);

//----------------------------------------------------------------------------------------------//

/*
	Produces reduced right-hand side vector containing only the
	corresponding elements of the original one with respect to
	mapReducedX.
*/
extern inline void produceReducedY_PSTRSV3(double *reducedRHS, double *originalRHS);

//----------------------------------------------------------------------------------------------//

/*
	Updates the original solution vector using the reducedX.
	mapReducedX is used to map reducedX values into the original x.
*/
extern inline void updateX(double *reducedX, double *originalX);

//----------------------------------------------------------------------------------------------//

/*
	Upper triangular solver: U*x = y
*/
extern inline void __udsol(int n, double *x, double *y, double *A, MKL_INT *IA, MKL_INT *JA);

//----------------------------------------------------------------------------------------------//

/*
	Upper triangular solver: U*X = B where X will be directly stored in B
*/
extern inline void __udsol2(int rowCount, int colCount, double *X, double *A, MKL_INT *IA, MKL_INT *JA, int stepsize);

//----------------------------------------------------------------------------------------------//

/*
	Upper triangular solver: U*x = b for the unit diagonal matrix U
*/
extern inline void __usol(int n, double *x, double *b, double *A, MKL_INT *IA, MKL_INT *JA);

//----------------------------------------------------------------------------------------------//

/*
	Upper triangular solver: U*x = b for the unit diagonal matrix U
	with single vector for both the right-hand side & the solution
*/
extern inline void __usol2(int n, double *x, double *A, MKL_INT *IA, MKL_INT *JA);

//----------------------------------------------------------------------------------------------//

/*
	Lower triangular solver: L*x = y
*/
extern inline void __ldsol(int n,double *x, double *y, double *A, MKL_INT *IA, MKL_INT *JA);

//----------------------------------------------------------------------------------------------//

/*
	Parallel memory initializer
*/
void pmemset_d(double *array, int x, int n, int nthreads);
void pmemset_i(MKL_INT *array, int x, int n, int nthreads);

//----------------------------------------------------------------------------------------------//

/*
	PSTRSV3 logger
*/
void logPSTRSV3(int nthreads);

//----------------------------------------------------------------------------------------------//
#endif