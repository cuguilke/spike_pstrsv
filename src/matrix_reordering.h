#ifndef MATRIX_REORDERING_H   /* Include guard */
#define MATRIX_REORDERING_H

#include "mkl.h"
//----------------------------------------------------------------------------------------------//

void matrixDuplication(int numRows, double *a, MKL_INT *ia, MKL_INT *ja, double **a_dup, MKL_INT **ia_dup, MKL_INT **ja_dup);

//----------------------------------------------------------------------------------------------//

/*
	If matrix A is not symmetric, call this routine
	It computes A' = A + A^T
	Returns symmetric matrix A' by updating the
	input pointers a, ia and ja
*/
void matrixSymmetrization(double **a, MKL_INT **ia, MKL_INT **ja, int numRows);

//----------------------------------------------------------------------------------------------//

/*
	Reorders the symmetric matrix A via METIS
	If uplo != 'N' or 'n', this routine only 
	returns the desired triangular matrix by
	by updating the input pointers a, ia and ja
	Else, returns the full reordered matrix by
	updating the input pointers a, ia and ja
*/
void matrixReordering(double **a, MKL_INT **ia, MKL_INT **ja, int numRows, int nthreads, const char uplo, const char symm, const char *matrixFileName);

//----------------------------------------------------------------------------------------------//

/*
	Iterates through CSR formatted sparse matrix and
	looks for dependent elements across solver threads.
	If there is no dependency, no need to reorder:
		returns 1
	else, reordering is necessary:
		returns 0
*/
int matrixAnalysis(int numRows, int nthreads, double **a, MKL_INT **ia, MKL_INT **ja, const char uplo, const char symm, const char *matrixFileName);

//----------------------------------------------------------------------------------------------//
#endif