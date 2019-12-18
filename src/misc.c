/*
	Title           :misc.c
	Description     :Miscellaneous functions
	Author          :Ilke Cugu
	Date Created    :26-06-2017
	Date Modified   :01-07-2018
	version         :2.0
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "misc.h"

void printCSRMatrix(char *matrixname, double *a, MKL_INT *ia, MKL_INT *ja, int m)
{ 
	int i, j, iRindex;
	double **matrix;
	printf("%s: ", matrixname);
	for(i=0;i<ia[m]-ia[0];i++)
		printf("%lf ", a[i]);
	printf("\nJ%s: ", matrixname);
	for(i=0;i<ia[m]-ia[0];i++)
		printf("%d ", ja[i]);
	printf("\nI%s: ", matrixname);
	for(i=0;i<=m;i++)
		printf("%d ", ia[i]);
	printf("\n");
	matrix = (double**)malloc(sizeof(double*)*m);
	for(i=0;i<m;i++)
		matrix[i] = (double*)malloc(sizeof(double)*m);
	for(i=0;i<m;i++)
		for(j=0;j<m;j++)
			matrix[i][j] = 0;
	iRindex = 0;
	if(m < 128)
	{
		for(i=0;i<m;i++)
		{
			for(j=ia[i];j<ia[i+1];j++)
				matrix[i][ja[j - ia[0]]] = a[j - ia[0]];
		}
		printf("Matrix:\n\n");
		for(i=0;i<m;i++)
		{
			for(j=0;j<m;j++)
			{
				//printf("%lf ", matrix[i][j]);
				if(matrix[i][j])
					printf("%d ", (int)matrix[i][j]);
				else
					printf("- ");
			}
			printf("\n");
		}
		printf("\n");
	}
	else printf("You should probably not try to print a matrix with that much rows.\n");
	for(i=0;i<m;i++)
		free(matrix[i]);
	free(matrix);
}

void printCOOMatrix(double *Av, int *Ai, int *Aj, int nnz, int m)
{
	int i,j;
	double **matrix;
	matrix = (double**)malloc(sizeof(double*)*m);
	for(i=0;i<m;i++)
		matrix[i] = (double*)malloc(sizeof(double)*m);
	for(i=0;i<m;i++)
		for(j=0;j<m;j++)
			matrix[i][j] = 0;
	for(i=0;i<nnz;i++)
		matrix[Ai[i]][Aj[i]] = Av[i];
	printf("Reordered Matrix:\n\n");
	for(i=0;i<m;i++)
	{
		for(j=0;j<m;j++)
		{
			//printf("%lf ", matrix[i][j]);
			if(matrix[i][j])
				printf("%d ", (int)matrix[i][j]);
			else
				printf("- ");
		}
		printf("\n");
	}
	printf("\n");
}

int compare(const void *pa, const void *pb)
{
	const double *a = *(const double **)pa;
	const double *b = *(const double **)pb;
	if(a[0] > b[0])
		return 1;
	else if(a[0] < b[0])
		return -1;
	else
	{
		if(a[1] > b[1])
			return 1;
		else
			return -1;
	}
}

void matrixMarketToCSR(const char *filename, double **a, MKL_INT **ia, MKL_INT **ja, int *numRows, const char uplo, const char symm)
{
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
	char c;
	int m; // the number of rows in the matrix.
	int n; // the number of columns in the matrix.
	int nonzeros, i, row, column, ia_index, nnz_triangular = 0, i_triangular, check; 
	double **coords;
	FILE *file;
	file = fopen(filename, "r");
	check = fscanf(file, "%c", &c);
	while(c == '%')
	{
		while (c != '\n')
			check = fscanf(file, "%c", &c);
		check = fscanf(file, "%c", &c);
	}
	fseek(file, -1, SEEK_CUR); // In order to exit, the loop above read 1 byte we need, so take it back
	check = fscanf(file, "%d %d %d", &m, &n, &nonzeros);
	if(symm == 'Y' || symm == 'y') nonzeros = nonzeros * 2 - n;
	numRows[0] = n;
	// Store coordinates and values of MM format
	(*ia) = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*((n+1) + 16 - ((n+1) % 16)), 64);
	coords = (double**)malloc(sizeof(double*)*nonzeros);
	for(i=0;i<nonzeros;i++)
		coords[i] = (double*)malloc(sizeof(double)*3);	
	(*ia)[0] = 0;
	ia_index = 1;
	int last_row = -1;
	// read matrixMarket format and store matrix in CSR
	for(i=0;i<nonzeros;i++)
	{
		if(symm == 'Y' || symm == 'y')
		{
			fscanf(file, "%lf %lf %lf", &coords[i][0], &coords[i][1], &coords[i][2]);
			if(coords[i][0] != coords[i][1] && i < nonzeros - 1)
			{
				i++;
				coords[i][0] = coords[i-1][1];
				coords[i][1] = coords[i-1][0];
				coords[i][2] = coords[i-1][2];
			}
		}
		else
			fscanf(file,"%lf %lf %lf",&coords[i][0], &coords[i][1], &coords[i][2]);
	}
	qsort(coords, nonzeros, sizeof(double*), compare);
	if(uplo == 'N' || uplo == 'n')
	{
		// Store whole matrix with CSR format.
		(*a) = (double*)mkl_malloc(sizeof(double)*(nonzeros + 8 - (nonzeros % 8)), 64);
		(*ja) = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*(nonzeros + 16 - (nonzeros % 16)), 64);
		for(i=0;i<nonzeros;i++)
		{
			(*a)[i] = coords[i][2];
			(*ja)[i] = (int)coords[i][1] - 1;
			if(last_row != (int)coords[i][0] && last_row != -1)
			{
				(*ia)[ia_index] = i;
				ia_index++;
			}
			last_row = (int)coords[i][0];
		}
		(*ia)[ia_index] = nonzeros;
	}
	else if(uplo == 'U' || uplo == 'u') 
	{
		// Store only upper-triangle matrix with CSR format.
		for(i=0;i<nonzeros;i++)
			if((int)coords[i][0] <= (int)coords[i][1])
				nnz_triangular++;
		(*a) = (double*)mkl_malloc(sizeof(double)*(nnz_triangular + 8 - (nnz_triangular % 8)), 64);
		(*ja) = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*(nnz_triangular + 16 - (nnz_triangular % 16)), 64);
		i_triangular = 0;
		for(i=0;i<nonzeros;i++)
		{
			if((int)coords[i][0] <= (int)coords[i][1])
			{
				(*a)[i_triangular] = coords[i][2];
				(*ja)[i_triangular] = (int)coords[i][1] - 1;
				if(last_row != (int)coords[i][0] && last_row != -1)
				{
					(*ia)[ia_index] = i_triangular;
					ia_index++;
				}
				i_triangular++;
				last_row = (int)coords[i][0];
			}
		}
		(*ia)[ia_index] = nnz_triangular;
	}
	else if(uplo == 'L' || uplo == 'l') 
	{
		// Store only lower-triangle matrix with CSR format.
		for(i=0;i<nonzeros;i++)
			if((int)coords[i][0] >= (int)coords[i][1])
				nnz_triangular++;
		(*a) = (double*)mkl_malloc(sizeof(double)*(nnz_triangular + 8 - (nnz_triangular % 8)), 64);
		(*ja) = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*(nnz_triangular + 16 - (nnz_triangular % 16)), 64);
		i_triangular = 0;
		for(i=0;i<nonzeros;i++)
		{
			if((int)coords[i][0] >= (int)coords[i][1])
			{
				(*a)[i_triangular] = coords[i][2];
				(*ja)[i_triangular] = (int)coords[i][1] - 1;
				if(last_row != (int)coords[i][0] && last_row != -1)
				{
					(*ia)[ia_index] = i_triangular;
					ia_index++;
				}
				i_triangular++;
				last_row = (int)coords[i][0];
			}
		}
		(*ia)[ia_index] = nnz_triangular;
	}
	fclose(file);
	for(i=0;i<nonzeros;i++)
		free(coords[i]);
	free(coords);
}

void strcopy(char *dest, const char *src, int len)
{
	int i;
	for(i=0;i<len && src[i]!='\0';i++)
		dest[i] = src[i];
	dest[len] = '\0';
}

char* pathToMatrixName(const char *filepath)
{
	/*
		It returns the name of the matrix for a given file path
		Example:
			input: /home/tyrion/casterlyRock.mtx
			output: casterlyRock
	*/
	int i, start = 0, end = 1;
	char *matrixName;

	// Determine the start & end indices of the name string
	for(i=0;filepath[i]!='\0';i++)
	{
		if(filepath[i] == '/')
			start = i + 1;
		else if(filepath[i] == '.')
			end = i;
	}

	// If everything is fine, allocate & define the name string
	if(end > start)
	{
		matrixName = (char*)malloc(sizeof(char)*((end - start) + 1));
		strcopy(matrixName, &filepath[start], end - start);
		return matrixName;
	}
	else
		return NULL;
}

char* strconcat(const char *str1, const char *str2)
{
	/*
		Wrapper for string concatenation function
	*/
	int len1 = strlen(str1);
	int len2 = strlen(str2);
	char *result;

	result = (char*)malloc(sizeof(char)*(len1 + len2 + 1));
	strcopy(result, str1, len1);
	strcopy(&result[len1], str2, len2);

	return result;
}

void CSRtoMatrixMarket(const char *matrixName, int n, double *a, MKL_INT *ia, MKL_INT *ja)
{
	/*
		Converts matrices in compressed sparse row format 
		into matrix market format and writes it on a file.
		This function automatically adds ".mtx" to the end 
		of the matrixName, so provide names accordingly.
	*/
	char *prefix = "%%MatrixMarket matrix coordinate real general\n%-------------------------------------------------------------------------------\n";
	int i, row = 1, col, shift = ia[0];
	double val;
	FILE *file;
	file = fopen(strconcat(matrixName, ".mtx"), "w");
	
	if(file == NULL)
	{
		fprintf(stderr, "%s", "Error opening the file!\n");
		exit(1);
	}
	
	fprintf(file, "%s", prefix);

	// Write numRows numCols numNnzs
	fprintf(file, "%d %d %d\n", n, n, ia[n]);

	for(i=ia[0]-shift;i<ia[n]-shift;i++)
	{
		if(i == ia[row]-shift)
			row++;
		col = ja[i] + 1;
		val = a[i];
		fprintf(file, "%d %d %f\n", row, col, val);
	}

	fclose(file);
}

void COOtoMatrixMarket(const char *matrixName, int n, int nnz, double *Av, int *Ai, int *Aj)
{
	/*
		Converts matrices in coordinate format into
		matrix market format and writes it on a file.
		This function automatically adds ".mtx" to end  
		of the matrixName, so provide names accordingly.
	*/
	char *prefix = "%%MatrixMarket matrix coordinate real general\n%-------------------------------------------------------------------------------\n";
	int i, row = 1, col;
	double val;
	FILE *file;
	file = fopen(strconcat(matrixName, ".mtx"), "w");
	
	if(file == NULL)
	{
		fprintf(stderr, "%s", "Error opening the file!\n");
		exit(1);
	}
	
	fprintf(file, "%s", prefix);

	// Write numRows numCols numNnzs
	fprintf(file, "%d %d %d\n", n, n, nnz);

	for(i=0;i<nnz;i++)
	{
		row = Ai[i];
		col = Aj[i];
		val = Av[i];
		fprintf(file, "%d %d %f\n", row, col, val);
	}

	fclose(file);
}