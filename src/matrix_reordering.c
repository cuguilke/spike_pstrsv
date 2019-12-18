/*
	Title           :matrix_reordering.c
	Description     :The code to reorder the matrix & distribute its partitions to different threads
	Author          :Ilke Cugu
	Date Created    :26-06-2017
	Date Modified   :18-12-2019
	version         :1.7.2
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <metis.h>
#include "misc.h"
#include "matrix_partitioning.h"
#include "matrix_reordering.h"
#include "spike_pstrsv_globals.h"

void matrixDuplication(int numRows, double *a, MKL_INT *ia, MKL_INT *ja, double **a_dup, MKL_INT **ia_dup, MKL_INT **ja_dup) 
{
	int nnz = ia[numRows], i;
	(*a_dup) = (double*)mkl_malloc(nnz*sizeof(double), 64);
	(*ja_dup) = (MKL_INT*)mkl_malloc(nnz*sizeof(MKL_INT), 32);
	(*ia_dup) = (MKL_INT*)mkl_malloc((numRows+1)*sizeof(MKL_INT), 32);
	cblas_dcopy(nnz, a, 1, (*a_dup), 1);
	for(i=0;i<nnz;i++) (*ja_dup)[i] = ja[i];
	for(i=0;i<=numRows;i++) (*ia_dup)[i] = ia[i];
}

void matrixSymmetrization(double **a, MKL_INT **ia, MKL_INT **ja, int numRows)
{
	/*
		If matrix A is not symmetric, call this routine
		It computes A' = A + A^T
		Returns symmetric matrix A' by updating the
		input pointers a, ia and ja
	*/
	int i, request, nnz, info, sort = 0;
	double *c, *b, beta = 1;
	MKL_INT *ic, *jc, *ib, *jb;
	const char trans = 'T';

	nnz = (*ia)[numRows];
	b = (double*)mkl_malloc(nnz*sizeof(double), 64);
	jb = (MKL_INT*)mkl_malloc(nnz*sizeof(MKL_INT), 32);
	ib = (MKL_INT*)mkl_malloc((numRows+1)*sizeof(MKL_INT), 32);
	ic = (MKL_INT*)mkl_malloc((numRows+1)*sizeof(MKL_INT), 32);	

	// Convert 0-based indices to 1-based indices
	for(i=0;i<nnz;i++)
		(*ja)[i]++;
	for(i=0;i<=numRows;i++)
		(*ia)[i]++;
	cblas_dcopy(nnz, (*a), 1, b, 1);
	for(i=0;i<nnz;i++) jb[i] = (*ja)[i];
	for(i=0;i<=numRows;i++) ib[i] = (*ia)[i];

	// Compute A^T + A and store in temp
	request = 1;
	mkl_dcsradd(&trans, &request, &sort, &numRows, &numRows, (*a), (*ja), (*ia), &beta, b, jb, ib, c, jc, ic, &nnz, &info);
	nnz = ic[numRows] - 1;
	c = (double*)mkl_malloc(nnz*sizeof(double), 64);
	jc = (MKL_INT*)mkl_malloc(nnz*sizeof(MKL_INT), 32);
	request = 2;
	mkl_dcsradd(&trans, &request, &sort, &numRows, &numRows, (*a), (*ja), (*ia), &beta, b, jb, ib, c, jc, ic, &nnz, &info);

	// Convert 1-based C to 0-based, then update the pointers of A
	for(i=0;i<nnz;i++)
		jc[i]--;
	for(i=0;i<=numRows;i++)
		ic[i]--;
	mkl_free((*a));
	mkl_free((*ia));
	mkl_free((*ja));
	mkl_free(b);
	mkl_free(ib);
	mkl_free(jb);
	(*a) = c;
	(*ia) = ic;
	(*ja) = jc;
}

void matrixReordering(double **a, MKL_INT **ia, MKL_INT **ja, int numRows, int nthreads, const char uplo, const char symm, const char *matrixFileName)
{
	/*
		Reorders the symmetric matrix A via METIS
		If uplo != 'N' or 'n', this routine only 
		returns the desired triangular matrix by
		by updating the input pointers a, ia and ja
		Else, returns the full reordered matrix by
		updating the input pointers a, ia and ja
	*/
	int i, j;
	double *a_;
	MKL_INT *ia_, *ja_;

	// Check whether reordering is necessary or not
	if(matrixAnalysis(numRows, nthreads, a, ia, ja, uplo, symm, matrixFileName)) return;

	// Before reordering, produce symmetric matrix from A if not
	// since METIS only works for symmetric matrices, then apply
	// reordering to the original A matrix
	if(symm == 'N' || symm == 'n') 
	{
		// Copy the original matrix
		matrixDuplication(numRows, (*a), (*ia), (*ja), &a_, &ia_, &ja_);

		// Modify copy matrix to produce the symmetric one
		matrixSymmetrization(&a_, &ia_, &ja_, numRows);
	}
	else
	{
		a_ = (*a);
		ia_ = (*ia);
		ja_ = (*ja);
	}

	// Init METIS variables
	idx_t nWeights  = 1;
	idx_t objval;
	int *part;
	part = (int*)malloc(sizeof(int)*numRows);

	// Set METIS Options
	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_CCORDER] = 0; // Identify the connected components
	options[METIS_OPTION_CONTIG] = 0; // Enforce contiguous partitions
	options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY; // Recursive bisectioning or k-way partitioning
	options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL; // Edge-cut or Minimize Comm Vol 
	options[METIS_OPTION_NUMBERING] = 0; // Index start
	//options[METIS_OPTION_UFACTOR] = 1.03; // Allowed Load Imbalance 
	//options[METIS_OPTION_DBGLVL] = 0;// Debugging info

	// Partition the graph
	int ret = METIS_PartGraphKway(&numRows, &nWeights, ia_, ja_,
					       NULL, NULL, NULL, &nthreads, NULL,
					       NULL, options, &objval, part);
	if(ret) // If the graph is successfully partitioned, reorder the matrix
	{
		// Build permutation vector of symmetric matrix
		int *perm, cnt = 0;
		perm = (int*)malloc(sizeof(int)*numRows);
		// Fill the variable k with amount of rows per thread 
		for(i=0;i<nthreads;i++)
		{
			partition_numRows[i] = 0;
			for(j=0;j<numRows;j++)
			{
				if(part[j] == i)
				{	
					partition_numRows[i]++;
					perm[cnt] = j;
					cnt++;
				}
			}
		}

		// Build inverse of permutation vector
		int *iperm;
		iperm = (int*)malloc(sizeof(int)*numRows);
		for(i=0;i<numRows;i++)
			iperm[perm[i]] = i;

		// Apply permutation to the original matrix
		double *Av;
		int *Aj, *Ai, nnz = (*ia)[numRows];
		Av = (double*)malloc(sizeof(double)*nnz);
		Ai = (int*)malloc(sizeof(int)*nnz);
		Aj = (int*)malloc(sizeof(int)*nnz);
		for(i=0;i<numRows;i++)
		{
			for(j=(*ia)[i];j<(*ia)[i+1];j++) 
			{
				Ai[j] = iperm[i];
				Aj[j] = iperm[(*ja)[j]];
				Av[j] = (*a)[j];
			}
		}

		// Convert COO matrix Av, Ai, Aj to CSR matrix a, ia, ja
		int job[6] = {2, 0, 0, 0, nnz, 0};
		int info;
		mkl_dcsrcoo(job, &numRows, (*a), (*ja), (*ia), &nnz, Av, Ai, Aj, &info);

		if(info == 0)
		{
			// Take upper or lower triangular of the reordered full matrix
			double *tmp;
			MKL_INT *itmp, *jtmp; 
			int nnzNew = 0, i_new, row = 0;
			if(uplo == 'U' || uplo == 'u')
			{
				// Count nnzs of Upper Triangular System
				for(i=(*ia)[0];i<(*ia)[numRows];i++)
				{
					if(i >= (*ia)[row + 1])
						row++;
					if((*ja)[i] >= row)
						nnzNew++;
				}

				// Init new matrix variables
				tmp  = (double*)mkl_malloc(sizeof(double)*nnzNew, 64);
				jtmp = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*nnzNew, 32);
				itmp = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*(numRows+1), 32);

				// Fill the new matrix
				row = 0;
				i_new = (*ia)[0];
				itmp[0] = 0;
				for(i=(*ia)[0];i<(*ia)[numRows];i++)
				{	
					if(i >= (*ia)[row + 1])
					{
						row++;
						itmp[row] = i_new;
					}
					if((*ja)[i] >= row)
					{
						tmp[i_new] = (*a)[i];
						jtmp[i_new] = (*ja)[i];
						i_new++;
					}
				}
				itmp[numRows] = nnzNew;
			}
			else // uplo == 'L' || uplo == 'l'
			{
				// Count nnzs of Lower Triangular System
				for(i=(*ia)[0];i<(*ia)[numRows];i++)
				{
					if(i >= (*ia)[row + 1])
						row++;
					if((*ja)[i] <= row)
						nnzNew++;
				}
				// Init new matrix variables
				tmp  = (double*)mkl_malloc(sizeof(double)*nnzNew, 64);
				jtmp = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*nnzNew, 32);
				itmp = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*(numRows+1), 32);

				// Fill the new matrix
				row = 0;
				i_new = (*ia)[0];
				itmp[0] = 0;
				for(i=(*ia)[0];i<(*ia)[numRows];i++)
				{	
					if(i >= (*ia)[row + 1])
					{
						row++;
						itmp[row] = i_new;
					}
					if((*ja)[i] <= row)
					{
						tmp[i_new] = (*a)[i];
						jtmp[i_new] = (*ja)[i];
						i_new++;
					}
				}
				itmp[numRows] = nnzNew;
			}

			// Determine partition_starts and partition_ends of threads
			int temp;
			IS_MATRIX_PARTITIONED = 1; // indicate that the matrix is already partitioned via METIS
			for(i=0;i<nthreads;i++)
			{
				temp = 0;
				for(j=0;j<i;j++)
					temp += partition_numRows[j];
				partition_starts[i] = temp;
				partition_ends[i] = partition_starts[i] + partition_numRows[i];
			}

			// Update matrix A and free the full matrix
			free(Av);
			free(Ai);
			free(Aj);
			mkl_free((*a));
			mkl_free((*ia));
			mkl_free((*ja));
			free(part);
			free(perm);
			free(iperm);
			(*a) = tmp;
			(*ia) = itmp;
			(*ja) = jtmp;
			if(symm == 'N' || symm == 'n') 
			{
				mkl_free(a_);
				mkl_free(ia_);
				mkl_free(ja_);
			}
		}
		else
		{
			printf("Matrix Conversion from COO to CSR is NOT successfull\n");
		}
	}
	else
	{
		printf("Matrix Reordering is NOT successfull\n");
	}
}

int matrixAnalysis(int numRows, int nthreads, double **a, MKL_INT **ia, MKL_INT **ja, const char uplo, const char symm, const char *matrixFileName)
{
	/*
		Iterates through CSR formatted sparse matrix and
		looks for dependent elements across solver threads.
		If there is no dependency, no need to reorder:
			returns 1
		else, reordering is necessary:
			returns 0
	*/
	int i, t, col, partition_start, partition_end;

	// Temporary partitioning, if reordering is necessary
	matrixPartitioning(numRows, nthreads);

	for(t=0;t<nthreads;t++)
	{
		partition_start = partition_starts[t];
		partition_end = partition_ends[t];
		if(uplo == 'u' || uplo == 'U')
		{
			for(i=(*ia)[partition_start];i<(*ia)[partition_end];i++)
			{
				col = (*ja)[i];
				if(partition_end <= col)
					return 0;
			}
		}
		else // uplo == 'l' || uplo == 'L'
		{
			for(i=(*ia)[partition_start];i<(*ia)[partition_end];i++)
			{
				col = (*ja)[i];
				if(partition_start > col)
					return 0;
			}
		}
	}

	// If the execution reaches here, no need for reordering.
	// This means METIS reordering function will return immediately,
	// and since it uses the whole matrix, triangular part won't be
	// extracted. Therefore, we must extract the triangular part 
	// indicated by uplo now. The easiest way is to:
	mkl_free(*a);
	mkl_free(*ia);
	mkl_free(*ja);
	matrixMarketToCSR(matrixFileName, a, ia, ja, &numRows, uplo, symm);

	return 1;
}