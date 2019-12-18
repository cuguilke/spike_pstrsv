/*
	Title           :matrix_partitioning.c
	Description     :The code to partition matrices & contain global variables
	Author          :Ilke Cugu
	Date Created    :26-06-2017
	Date Modified   :18-12-2019
	version         :1.8.3
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "misc.h"
#include "matrix_partitioning.h"
#include "spike_pstrsv_globals.h"

void matrixPartitioning(int numRows, int nthreads)
{
	int i;
	int leftovers  = numRows % nthreads;
	int kperThread = numRows / nthreads;
	if(!IS_MATRIX_PARTITIONED)
	{
		for(i=0;i<nthreads;i++)
		{
			partition_numRows[i] = kperThread;
			partition_starts[i] = kperThread * i;
			partition_ends[i] = partition_starts[i] + partition_numRows[i];
		}
		partition_numRows[nthreads - 1] += leftovers; 
		partition_ends[nthreads - 1] += leftovers;
	}
	else
		IS_MATRIX_PARTITIONED = 0;
}

void preprocessD(double *a, MKL_INT *ia, MKL_INT *ja, int nthreads)
{
	/*
		Computes start & end indices of each D matrix partition, and
		produces D matrix. Each partition matrix Di, columns are left
		aligned. Meaning:
			- newCol[i] = oldCol[i] - partition_start[t]
		Call this function once, then the solver can be called n times.
	*/
	int t, i, row = 1, col, i_D = 0, partition_start, partition_end;

	// Produce D and set start & end indices for each thread
	ID[0] = 0;
	for(t=0;t<nthreads;t++)
	{
		starts_D[t] = row - 1;
		partition_start = partition_starts[t];
		partition_end = partition_ends[t];
		for(i=ia[partition_start];i<ia[partition_end];i++)
		{
			col = ja[i];
			if(i == ia[row])
			{
				ID[row] = i_D;
				row++;
			}
			if(partition_start <= col && col < partition_end)
			{
				D[i_D] = a[i];
				JD[i_D] = col - partition_start; // left-align to be able to act it like a separate matrix
				i_D++;
			}
		}
		ID[row] = i_D; // process the last element of ID array
		ends_D[t] = row;
		row++; // at the end, row = n + 1
	}
	starts_R[0] = ends_D[nthreads - 1];
}

void preprocessR(double *a, MKL_INT *ia, MKL_INT *ja, int nthreads, const char uplo)
{
	/*
		Computes start & end indices of each R matrix partition, and
		produces R matrix. It is stored in the leftover space at the
		end of D, ID, and JD arrays. It is guaranteed that the memory
		is enough to store R matrix iff A matrix does not require more
		rows than MAXSIZE & nonzeros than MAXNNZS.
		Call this function once, then the solver can be called n times.
	*/
	int t, i, row = 1, col, prefix, i_R, partition_start, partition_end;
	prefix = starts_R[0];
	i_R = ID[prefix];

	// Produce R and set start & end indices for each thread
	for(t=0;t<nthreads;t++)
	{
		starts_R[t] = prefix + row - 1;
		partition_start = partition_starts[t];
		partition_end = partition_ends[t];
		for(i=ia[partition_start];i<ia[partition_end];i++)
		{
			col = ja[i];
			if(i == ia[row])
			{
				ID[prefix + row] = i_R;
				row++;
			}
			if(uplo == 'u' || uplo == 'U')
			{
				if(partition_end <= col)
				{
					D[i_R] = a[i];
					JD[i_R] = col;
					i_R++;
				}
			}
			else // uplo == 'l' || uplo == 'L'
			{
				if(partition_end > col)
				{
					D[i_R] = a[i];
					JD[i_R] = col;
					i_R++;
				}
			}
		}
		ID[prefix + row] = i_R;
		ends_R[t] = prefix + row;
		row++;
	}
}

int getOwnerThread(int col, int nthreads)
{
	/*
		Returns id of the owner thread of the partition that the given 
		column belongs to.
	*/
	int t, partition_start, partition_end;
	for(t=0;t<nthreads;t++)
	{
		partition_start = partition_starts[t];
		partition_end = partition_ends[t];
		if(partition_start <= col && col < partition_end)
			return t;
	}
	return -1;
}