#ifndef MATRIX_PARTITIONING_H   /* Include guard */
#define MATRIX_PARTITIONING_H

#include "mkl.h"
//----------------------------------------------------------------------------------------------//

void matrixPartitioning(int numRows, int nthreads);

//----------------------------------------------------------------------------------------------//

/*
	Computes start & end indices of each D matrix partition, and
	produces D matrix. Each partition matrix Di, columns are left
	aligned. Meaning:
		- newCol[i] = oldCol[i] - partition_start[t]
	Call this function once, then the solver can be called n times.
*/
void preprocessD(double *a, MKL_INT *ia, MKL_INT *ja, int nthreads);

//----------------------------------------------------------------------------------------------//

/*
	Produces R = A - D
	Computes start & end indices of each R matrix partition, and
	produces R matrix. It is stored in the leftover space at the
	end of D, ID, and JD arrays. It is guaranteed that the memory
	is enough to store R matrix iff A matrix does not require more
	rows than MAXSIZE & nonzeros than MAXNNZS.
	Call this function once, then the solver can be called n times.
*/
void preprocessR(double *a, MKL_INT *ia, MKL_INT *ja, int nthreads, const char uplo);

//----------------------------------------------------------------------------------------------//

/*
	Returns id of the owner thread of the partition that the given 
	column belongs to.
*/
int getOwnerThread(int col, int nthreads);

//----------------------------------------------------------------------------------------------//
#endif