#ifndef MISC_H  /* Include guard */
#define MISC_H

#include "mkl.h"
//----------------------------------------------------------------------------------------------//

void printCSRMatrix(char *matrixname, double *a, MKL_INT *ia, MKL_INT *ja, int m);

//----------------------------------------------------------------------------------------------//

void printCOOMatrix(double *Av, int *Ai, int *Aj, int nnz, int m);

//----------------------------------------------------------------------------------------------//

int compare(const void *pa, const void *pb);

//----------------------------------------------------------------------------------------------//

/*
	Written specifically for: 
		object: matrix
		format: coordinate
		field: double || real || integer
	of Matrix Market Format. 
	This function returns 0-based indexed matrix with CSR format.
	Set uplo variable to make a matrix triangular:
		uplo: 'U' or 'u' -> for getting upper triangular
		uplo: 'L' or 'l' -> for getting lower triangular
		uplo: 'N' or 'n' -> take matrix as it is
	Set symm variable to properly read symmetric matrices with MM format:
		symm: 'Y' or 'y' -> real symmetric matrix
		symm: 'S' or 's' -> nonzeros pattern symmetric matrix
		symm: 'N' or 'n' -> unsymmetric matrix
*/
void matrixMarketToCSR(const char *filename, double **a, MKL_INT **ia, MKL_INT **ja, int *numRows, const char uplo, const char symm);

//----------------------------------------------------------------------------------------------//

/*
	It returns the name of the matrix for a given file path
	Example:
		input: /home/tyrion/casterlyRock.mtx
		output: casterlyRock
*/
char* pathToMatrixName(const char *filepath);

//----------------------------------------------------------------------------------------------//

/*
	Wrapper for string concatenation function
*/
char* strconcat(const char *str1, const char *str2);

//----------------------------------------------------------------------------------------------//

/*
	Converts matrices in compressed sparse row format 
	into matrix market format and writes it on a file.
	This function automatically adds ".mtx" to the end 
	of the matrixName, so provide names accordingly.
*/
void CSRtoMatrixMarket(const char *matrixName, int n, double *a, MKL_INT *ia, MKL_INT *ja);

//----------------------------------------------------------------------------------------------//

/*
	Converts matrices in coordinate format into
	matrix market format and writes it on a file.
	This function automatically adds ".mtx" to end  
	of the matrixName, so provide names accordingly.
*/
void COOtoMatrixMarket(const char *matrixName, int n, int nnz, double *Av, int *Ai, int *Aj);

//----------------------------------------------------------------------------------------------//
#endif