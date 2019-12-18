/*
	Title           :spike_pstrsv_ops.c
	Description     :Necessary functions specialized for pSTRSV3
	Author          :Ilke Cugu
	Date Created    :23-07-2017
	Date Modified   :18-12-2019
	version         :4.8.4
*/

#include "spike_pstrsv_ops.h"
#include "spike_pstrsv_globals.h"
#include "matrix_partitioning.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

double *S; 											// partial S matrix of DS Factorization
double *R_dense;									// partial compressed R matrix to compute partial S
double *compressedX;
MKL_INT *mapR; 										// will be used to store mapping of R_dense -> R & compressedX -> x
MKL_INT starts_mapR[MAXTHREADCOUNT]; 				// will be used for both R_dense & compressedX 
MKL_INT ends_mapR[MAXTHREADCOUNT]; 					// will be used for both R_dense & compressedX
MKL_INT numRows_S[MAXTHREADCOUNT]; 					// will be used to store 1st dim of S parts per thread
MKL_INT numCols_S[MAXTHREADCOUNT]; 					// will be used to store 2nd dim of S parts per thread
long long starts_S[MAXTHREADCOUNT]; 				// indices of each thread's start point of S sections
long long ends_S[MAXTHREADCOUNT]; 					// indices of each thread's end point of S sections
MKL_INT partition_numRows[MAXTHREADCOUNT];			// row count per thread
MKL_INT partition_starts[MAXTHREADCOUNT];			// start indices of A matrix for each thread
MKL_INT partition_ends[MAXTHREADCOUNT];				// end indices of A matrix for each thread
MKL_INT starts_R[MAXTHREADCOUNT];					// points to ID array
MKL_INT ends_R[MAXTHREADCOUNT];						// points to ID array
MKL_INT starts_D[MAXTHREADCOUNT];					// points to ID array
MKL_INT ends_D[MAXTHREADCOUNT];						// points to ID array
MKL_INT starts_D0[MAXTHREADCOUNT];  				// points to IT2 array
MKL_INT ends_D0[MAXTHREADCOUNT];					// points to IT2 array
MKL_INT starts_D1[MAXTHREADCOUNT];  				// points to IT2 array
MKL_INT ends_D1[MAXTHREADCOUNT];					// points to IT2 array
MKL_INT starts_D2[MAXTHREADCOUNT]; 					// points to IT2 array
MKL_INT ends_D2[MAXTHREADCOUNT];					// points to IT2 array
MKL_INT dependents[MAXSIZE]; 						// will be used to indicate the elements of the reduced system
double D[MAXNNZS*2]; 								// will be used to store D & R matrices of DS Factorization
MKL_INT ID[MAXSIZE*2];
MKL_INT JD[MAXNNZS*2];
double T[MAXNNZS]; 									// will be used to store the reduced system
MKL_INT IT[MAXSIZE]; 
MKL_INT JT[MAXNNZS];
double T2[3*MAXNNZS];								// will be used to store D0, D1, D2 and D3 of pSTRSV3
MKL_INT IT2[3*MAXSIZE];
MKL_INT JT2[3*MAXNNZS];
MKL_INT mapReducedX[MAXSIZE];						// will be used to store mapping of reducedX -> x
double reducedX[MAXSIZE];
MKL_INT reducedSystemSolverStart;					// will be used to prevent unnecessary memory ops for the rows consist of the diagonal only in the reduced system
MKL_INT numRowsReducedSystem;
MKL_INT numNnzsReducedSystem;
char isReducedSystemNecessary = 0;
char hasDependences[MAXTHREADCOUNT];				// indicates whether a particular thread has R matrix or not
char hasReflections[MAXTHREADCOUNT];				// indicates whether partial S calculation is necessary or not for a particular thread
char hasToCalculateSs[MAXTHREADCOUNT];				// indicates whether a thread needs to calculate the partial S matrix
MKL_INT start_intervals[MAXTHREADCOUNT];			// holds minDependentRows
MKL_INT end_intervals[MAXTHREADCOUNT];				// holds maxIntersectionRows
char isOptimizeds[MAXTHREADCOUNT]; 					// indicates if the thread has D1, D2 and D3 partitions despite its hasReflection = false
char FORCE_LOAD_BALANCE	= 1;						// environment variable to enable further partitioning of D matrix to improve the load-balance
char IS_MATRIX_PARTITIONED = 0;						// if > 0 -> matrixPartitioning function creates partitions
char PSTRSV_VERBOSE = 0;							// environment variable to enable PSTRSV3 operations log
int epsilon_1 = 0;
int epsilon_2 = 0;

void initGlobals_PSTRSV3(int n, int nnz, int nthreads)
{
	pmemset_d(D,0,sizeof(double)*2*nnz,nthreads);
	pmemset_i(ID,0,sizeof(MKL_INT)*2*(n+1),nthreads);
	pmemset_i(JD,0,sizeof(MKL_INT)*2*nnz,nthreads);
	pmemset_i(dependents,0,sizeof(MKL_INT)*n,nthreads);
	pmemset_d(T,0,sizeof(double)*nnz,nthreads);
	pmemset_i(JT,0,sizeof(MKL_INT)*nnz,nthreads);
	pmemset_i(IT,0,sizeof(MKL_INT)*(n+1),nthreads);
	pmemset_d(T2,0,sizeof(double)*3*nnz,nthreads);
	pmemset_i(JT2,0,sizeof(MKL_INT)*3*nnz,nthreads);
	pmemset_i(IT2,0,sizeof(MKL_INT)*3*(n+1),nthreads);
	pmemset_i(mapReducedX,0,sizeof(MKL_INT)*n,nthreads);
	pmemset_d(reducedX,0,sizeof(double)*n,nthreads);
	memset(starts_mapR,0,sizeof(MKL_INT)*nthreads);
	memset(ends_mapR,0,sizeof(MKL_INT)*nthreads);
	memset(starts_R,0,sizeof(MKL_INT)*nthreads);
	memset(ends_R,0,sizeof(MKL_INT)*nthreads);
	memset(starts_D,0,sizeof(MKL_INT)*nthreads);
	memset(ends_D,0,sizeof(MKL_INT)*nthreads);
	memset(starts_D0,0,sizeof(MKL_INT)*nthreads);
	memset(ends_D0,0,sizeof(MKL_INT)*nthreads);
	memset(starts_D1,0,sizeof(MKL_INT)*nthreads);
	memset(ends_D1,0,sizeof(MKL_INT)*nthreads);
	memset(starts_D2,0,sizeof(MKL_INT)*nthreads);
	memset(ends_D2,0,sizeof(MKL_INT)*nthreads);
	memset(starts_S,0,sizeof(long long)*nthreads);
	memset(ends_S,0,sizeof(long long)*nthreads);
	memset(numRows_S,0,sizeof(MKL_INT)*nthreads);
	memset(numCols_S,0,sizeof(MKL_INT)*nthreads);
	memset(start_intervals,0,sizeof(MKL_INT)*nthreads);
	memset(end_intervals,0,sizeof(MKL_INT)*nthreads);
	memset(hasDependences,0,sizeof(char)*nthreads);
	memset(hasReflections,0,sizeof(char)*nthreads);
	memset(hasToCalculateSs,0,sizeof(char)*nthreads);
	memset(isOptimizeds,0,sizeof(char)*nthreads);
	isReducedSystemNecessary = 0;
}

void initTimers_PSTRSV3(int nthreads)
{
	memset(temp_timer,0,sizeof(double)*nthreads);
	memset(waitBefore_timer_pstrsv3,0,sizeof(double)*nthreads);
	memset(computeY_timer,0,sizeof(double)*nthreads);
	memset(solveIndependent_timer,0,sizeof(double)*nthreads);
	memset(subtract_Rx_timer,0,sizeof(double)*nthreads);
	memset(waitAfter_timer_pstrsv3, 0, sizeof(double)*nthreads);
	memset(solveD2_timer,0,sizeof(double)*nthreads);
	memset(solveD_timer,0,sizeof(double)*nthreads);
	reduced_solver_timer_pstrsv3 = 0;
	init_vars_pstrsv3 = 0;
}

void free_PSTRSV3()
{
	mkl_free(R_dense);
	mkl_free(compressedX);
	mkl_free(mapR);
}

void setDependents_PSTRSV3(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo)
{
	/*
		Provides a map of all dependents elements in A matrix
		by marking the nonzeros outside of the partition area 
		for each thread. Also indicates general dependency in
		the A matrix.
	*/
	int t, i, col, start, end, counter = 0;

	// Tag all dependents according to R matrix
	numRowsReducedSystem = 0;
	memset(dependents,0,sizeof(MKL_INT)*n);
	if(uplo == 'u' || uplo == 'U')
	{
		for(t=0;t<nthreads-1;t++)
		{
			start = partition_starts[t];
			end = partition_ends[t];
			for(i=ia[start];i<ia[end];i++)
			{
				col = ja[i];
				if(end <= col)
				{
					isReducedSystemNecessary = 1;
					hasDependences[t] = 1;
					numRowsReducedSystem = dependents[col] ? numRowsReducedSystem : numRowsReducedSystem + 1;
					dependents[col] = 1;
					counter++;
				}
			}
		}
	}
	else
	{
		for(t=0;t<nthreads;t++)
		{
			start = partition_starts[t];
			end = partition_ends[t];
			for(i=ia[start];i<ia[end];i++)
			{
				col = ja[i];
				if(col < start)
				{
					isReducedSystemNecessary = 1;
					hasDependences[t] = 1;
					numRowsReducedSystem = dependents[col] ? numRowsReducedSystem : numRowsReducedSystem + 1;
					dependents[col] = 1;
					counter++;
				}
			}
		}
	}
}

int getIntervalEnd(double *a, MKL_INT *ia, MKL_INT *ja, int start, int end, const char uplo)
{
	/*
		Returns the row index of the nnz of R matrix at the bottom
		for the upper and at the top for the lower triangular case. 
	*/
	int i, col, result = 0;
	if(uplo == 'u' || uplo == 'U')
	{
		result = end - 1;
		for(i=ia[end]-1;i>=ia[start];i--)
		{
			col = ja[i];
			if(end <= col)
				break;
			else
			{
				if(i == ia[result])
					result--;
			}
		}
	}
	else // uplo == 'l' || uplo == 'L'
	{
		//TODO: Implement lower triangular support
	}

	return result;
}

int preprocessS_PSTRSV3(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo)
{
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
	int i, t, nextRow, col, start_D, end_D, partition_start, partition_end, gap, minDependentRow, maxIntersectionRow, i_D0 = 0, row = 0;
	long long colCount, rowCount, totalArea = 0;
	
	if(isReducedSystemNecessary)
	{
		for(t=0;t<nthreads;t++)
		{
			colCount = 0; // use it to count numCols_S[t]
			rowCount = 0;
			partition_start = partition_starts[t];
			partition_end = partition_ends[t];
			minDependentRow = partition_end;
			maxIntersectionRow = 0;
			if(uplo == 'u' || uplo == 'U')
			{
				// The first thread does not need to compute S
				if(t == 0)
				{
					maxIntersectionRow = getIntervalEnd(a, ia, ja, partition_start, partition_end, uplo);
					end_intervals[t] = maxIntersectionRow - partition_start + 1;
					start_intervals[t] = end_intervals[t];
					starts_S[t] = 0;
					ends_S[t] = 0;
					continue; 
				}

				// Find the dependency with the smallest column index, and
				// use it as the stopping point of the smaller D matrix
				for(i=partition_start;i<partition_end;i++)
				{
					if(dependents[i])
					{
						minDependentRow = i;
						hasReflections[t] = 1;
						break;
					}
				}
				if(hasReflections[t])
				{
					gap = minDependentRow - partition_start;

					// Draw a line through the min dependent row to find intersection
					// points which are then used to determine borders of the partial
					// S matrix to be computed
					for(i=ia[minDependentRow];i<ia[partition_end];i++)
					{
						col = ja[i];
						if(partition_end <= col && dependents[col] != 2 + t)
						{
							colCount++; // count nonzeros at intersections to determine numCols_S
							dependents[col] = 2 + t; // mark the necessary nonzeros
						}
					}
					if(colCount > 0)
						hasToCalculateSs[t] = 1;

					// Set column count of partial S matrix of thread t
					numCols_S[t] = (MKL_INT)colCount;

					// Find out upto which point the S matrix contains only 0s.
					// Then, use that point to end the smaller D matrix which is
					// then used to compute the partial S matrix.
					maxIntersectionRow = getIntervalEnd(a, ia, ja, partition_start, partition_end, uplo);

					// Set row count of partial S matrix of thread t
					numRows_S[t] = max(1 + maxIntersectionRow - minDependentRow, 0);
					rowCount = (long long)numRows_S[t];

					// Set intervals indicating the crucial parts of R and S matrices to produce the small system
					start_intervals[t] = minDependentRow - partition_start;
					end_intervals[t] = (maxIntersectionRow < minDependentRow) ? start_intervals[t] + 1 : maxIntersectionRow - partition_start + 1;

					// Produce the smaller D matrix (a.k.a D0 matrix of pSTRSV3)
					start_D = starts_D[t] + start_intervals[t];
					end_D = starts_D[t] + end_intervals[t];
					nextRow = start_D + 1;
					starts_D0[t] = row;
					for(i=ID[start_D];i<ID[end_D];i++)
					{
						col = JD[i];
						if(i == ID[nextRow])
						{
							nextRow++;
							row++;
							IT2[row] = i_D0;
						}
						if(start_intervals[t] <= col && col < end_intervals[t])
						{
							T2[i_D0] = D[i];
							JT2[i_D0] = col - gap; // left-align D0 matrices
							i_D0++;
						}
					}
					row++;
					IT2[row] = i_D0;
					ends_D0[t] = row;
				}
				else
				{
					// We can still partition the D_i matrix if it provides better load-balance
					// Therefore, determine the partition line where the solution of the lower
					// subsystem calculates the lower part of the solution vector x_i
					maxIntersectionRow = getIntervalEnd(a, ia, ja, partition_start, partition_end, uplo);
					end_intervals[t] = maxIntersectionRow - partition_start + 1;
					start_intervals[t] = end_intervals[t];
				}
			}
			else // uplo == 'l' || uplo == 'L'
			{
				//TODO: Implement lower triangular support
			}
			
			// Determine memory need of S matric partitions and their starting & ending points
			starts_S[t] = ends_S[t-1];
			ends_S[t] = starts_S[t] + (MKL_INT)(colCount * rowCount);
			totalArea += colCount * rowCount;
		}

		// For 64 byte alignment & parallel memset
		totalArea += 40 - (totalArea % 40);
		
		// Check for a better load-balance
		if(FORCE_LOAD_BALANCE) optimizeLoadBalance(a, ia, ja, n, nthreads, uplo);

		// Check for memory boundaries
		if((totalArea * (long long)sizeof(double) * 3) > MEMSIZE)
		{
			// If logs are enabled print detailed information before exit
			if(PSTRSV_VERBOSE) logPSTRSV3(nthreads);

			fprintf(stderr, "%s", "Error: The estimated size of Matrix S is beyond boundaries!\n");	
			exit(1);
		}
		
		// Split D partitions in three (D1, D2, D3) at maxIntersectionRow if the thread hasReflection or isOptimized
		separateDsections_PSTRSV3(nthreads, uplo, row);
		
		// If memory bounds are not exceeded, allocate the necessary space
		temp_timer[0] = omp_get_wtime();
		R_dense = (double*)mkl_malloc(sizeof(double)*totalArea, 64);
		pmemset_d(R_dense, 0, totalArea, nthreads);
		memoryAlloc_timer_pstrsv3 += omp_get_wtime() - temp_timer[0];
	}
	
	return 0;
}

void optimizeLoadBalance(double *a, MKL_INT *ia, MKL_INT *ja, int n, int nthreads, const char uplo)
{
	/*
		Checks whether we can achieve better load-balance by also
		partitioning D_i matrices with no reflections that are waiting
		for the full D_i * x_i = b_i solution while other threads are
		solving the D1_i * x_i = b_i at the beginning of pSTRSV3.
	*/
	int i, j, temp, row, col, start, end, start_interval, end_interval, gain, loss, totalLoss, minLoss = INF, bestCut;
	int nnz_R, nnz_D, nnz_D1, nnz_D2, nnz_P, cost1 = 0, cost2 = 0, curCost1 = 0, curCost2 = 0, newCost1 = 0, newCost2 = 0;
	char hasReflection, betterLoadBalance = 0;

	if(uplo == 'u' || uplo == 'U')
	{
		// Calculate the current costs
		for(i=0;i<nthreads;i++)
		{
			hasReflection = hasReflections[i];
			start = partition_starts[i];
			end = partition_ends[i];
			nnz_R = 0;
			if(hasReflection)
			{
				// We need to calculate costs for threads with reflections only once since they are static
				start_interval = start + start_intervals[i];
				end_interval = start + end_intervals[i];
				nnz_D1 = 0;
				nnz_D2 = 0;
				nnz_P = 0;
				row = start;
				for(j=ia[start];j<ia[end];j++)
				{
					col = ja[j];
					if(j == ia[row + 1])
						row++;

					if(end <= col)
						nnz_R++;
					else
					{
						if(start_interval > row)
						{
							if(end_interval > col)
								nnz_D2++;
							else
								nnz_P++;
						}
						else if(end_interval > row)
						{
							if(end_interval > col)
								nnz_D2++;
							else
								nnz_P++;
							nnz_D1++;
						}
						else
							nnz_D1++;
					}
				}
				cost1 = max(cost1, nnz_D1);
				cost2 = max(cost2, nnz_R + nnz_P + nnz_D2);
			}
			else
			{
				// Load-balance operations will modify only these threads' contents
				nnz_D = ia[end] - ia[start];
				for(j=ia[start];j<ia[end];j++)
				{
					col = ja[j];
					if(end < col)
						nnz_R++;
				}
				curCost2 = max(curCost2, nnz_R + nnz_D);

				// Mark threads with R_old[i] is empty since we can set {start | end}_intervals[i]
				// to any value we want to maximize the load-balance (1: regular, 2: this condition)
				isOptimizeds[i] = nnz_R == 0 ? 2 : 0; 
			}
		}
		curCost1 = cost1;
		curCost2 = max(curCost2, cost2); 

		// Calculate the probable costs
		for(i=0;i<nthreads;i++)
		{
			hasReflection = hasReflections[i];
			start = partition_starts[i];
			end = partition_ends[i];
			nnz_R = 0;
			if(!hasReflection && isOptimizeds[i] != 2)
			{
				end_interval = start + end_intervals[i];
				start_interval = end_interval - 1;
				nnz_D1 = 0;
				nnz_D2 = 0;
				nnz_R = 0;
				row = start;
				for(j=ia[start];j<ia[end];j++)
				{
					col = ja[j];
					if(j == ia[row + 1])
						row++;

					if(end <= col)
						nnz_R++;
					else
					{
						if(start_interval > row)
						{
							if(end_interval > col)
								nnz_D2++;
							else
								nnz_P++;
						}
						else if(end_interval > row)
						{
							if(end_interval > col)
								nnz_D2++;
							else
								nnz_P++;
							nnz_D1++;
						}
						else
							nnz_D1++;
					}
				}
				newCost1 = max(newCost1, nnz_D1);
				newCost2 = max(newCost2, nnz_R + nnz_P + nnz_D2);
				isOptimizeds[i] = 1;
			}
		}
		newCost2 = max(newCost2, cost2);

		// Check if a better load-balance is feasible
		loss = max(0, newCost1 - curCost1);
		gain = max(0, curCost2 - newCost2);
		betterLoadBalance = (gain > loss + epsilon_1) ? 1 : 0;
		//printf("gain: %d, loss: %d\n", gain, loss);

		// If applicable, determine ideal cuts for the threads where R_old[i] is empty
		if(betterLoadBalance)
		{
			totalLoss = 0;
			for(i=0;i<nthreads;i++)
			{
				if(isOptimizeds[i] == 2)
				{
					start = partition_starts[i];
					end = partition_ends[i];
					nnz_D2 = 0;
					nnz_P = 0;
					nnz_D = ia[end] - ia[start];
					row = 0;
					for(j=ia[start];j<ia[end];j++)
					{
						col = ja[j];
						if(j == ia[row + 1])
						{
							nnz_D1 = nnz_D - (nnz_D2 + nnz_P);
							cost1 = max(0, nnz_D1 - newCost1);
							cost2 = max(0, nnz_D2 + nnz_P - newCost2);
							loss = cost1 + cost2;
							if(loss <= minLoss)
								minLoss = loss;
							else
								bestCut = row - 1;
								break;
							row++;
						}

						if(col <= row)
							nnz_D2++;
						else
							nnz_P++;
					}
					start_intervals[i] = bestCut;
					end_intervals[i] = bestCut + 1;
					totalLoss += minLoss;
				}
			}
		}
	}
	else // uplo == 'l' || uplo == 'L'
	{
		//TODO: Implement lower triangular support
	}

	// Final decision on the load-balance
	if(totalLoss > epsilon_2)
		memset(isOptimizeds,0,sizeof(char)*nthreads);
}

void separateDsections_PSTRSV3(int nthreads, const char uplo, int rowD1)
{
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
	char hasReflection, isOptimized;
	int t, i, col, partition_start, partition_end, stop, newN, k, row, nextRow, i_D1, i_D2;

	// Produce D1 sections
	row = rowD1;  
	i_D1 = IT2[row];
	if(uplo == 'u' || uplo == 'U')
	{
		for(t=0;t<nthreads;t++)
		{
			hasReflection = hasReflections[t];
			isOptimized = isOptimizeds[t];
			if(hasReflection || isOptimized)
			{
				stop = start_intervals[t];
				starts_D1[t] = row;
				partition_start = starts_D[t] + stop;
				partition_end = ends_D[t];
				nextRow = partition_start + 1;
				for(i=ID[partition_start];i<ID[partition_end];i++)
				{
					if(i == ID[nextRow])
					{
						nextRow++;
						row++;
						IT2[row] = i_D1;
					}
					T2[i_D1] = D[i];
					JT2[i_D1] = JD[i] - stop; // left-align D1 matrices
					i_D1++;
				}
				row++;
				IT2[row] = i_D1;
				ends_D1[t] = row;
			}
			else 
			{
				starts_D1[t] = 0;
				ends_D1[t] = 0;
			}
		}
	}
	else // uplo == 'l' || uplo == 'L'
	{
		for(t=0;t<nthreads;t++)
		{
			hasReflection = hasReflections[t];
			isOptimized = isOptimizeds[t];
			if(hasReflection || isOptimized)
			{
				stop = end_intervals[i];
				starts_D1[t] = row;
				partition_start = starts_D[t];
				partition_end = partition_start + stop;
				nextRow = partition_start + 1;
				for(i=ID[partition_start];i<ID[partition_end];i++)
				{
					if(i == ID[nextRow])
					{
						nextRow++;
						row++;
						IT2[row] = i_D1;
					}
					T2[i_D1] = D[i];
					JT2[i_D1] = JD[i];
					i_D1++;
				}
				row++;
				IT2[row] = i_D1;
				ends_D1[t] = row;
			}
			else 
			{
				starts_D1[t] = 0;
				ends_D1[t] = 0;
			}
		}
	}	
	
	// Produce D2 sections
	i_D2 = i_D1;
	if(uplo == 'u' || uplo == 'U')
	{
		for(t=0;t<nthreads;t++)
		{
			hasReflection = hasReflections[t];
			isOptimized = isOptimizeds[t];
			if(hasReflection || isOptimized)
			{
				stop = end_intervals[t];
				starts_D2[t] = row;
				partition_start = starts_D[t];
				partition_end = partition_start + stop;
				nextRow = partition_start + 1;
				for(i=ID[partition_start];i<ID[partition_end];i++)
				{
					col = JD[i]; // reminder: JD is left aligned for each thread
					if(i == ID[nextRow])
					{
						nextRow++;
						row++;
						IT2[row] = i_D2;
					}
					if(col < stop)
					{
						T2[i_D2] = D[i];
						JT2[i_D2] = col; // no need to left align the top of a upper triangular matrix
						i_D2++;
					}
				}
				row++;
				IT2[row] = i_D2;
				ends_D2[t] = row;
			}
			else 
			{
				starts_D2[t] = 0;
				ends_D2[t] = 0;
			}
		}
	}
	else // uplo == 'l' || uplo == 'L'
	{
		for(t=0;t<nthreads;t++)
		{
			hasReflection = hasReflections[t];
			isOptimized = isOptimizeds[t];
			if(hasReflection || isOptimized)
			{
				stop = end_intervals[t];
				starts_D2[t] = row;
				partition_start = starts_D[t] + stop;
				partition_end = ends_D[t];
				nextRow = partition_start + 1;
				for(i=ID[partition_start];i<ID[partition_end];i++)
				{
					col = JD[i]; // reminder: JD is left aligned for each thread
					if(i == ID[nextRow])
					{
						nextRow++;
						row++;
						IT2[row] = i_D2;
					}
					if(col >= stop)
					{
						T2[i_D2] = D[i];
						JT2[i_D2] = col - stop; // left-align to make it a (k-stop)x(k-stop) matrix
						i_D2++;
					}
				}
				row++;
				IT2[row] = i_D2;
				ends_D2[t] = row;
			}
			else 
			{
				starts_D2[t] = 0;
				ends_D2[t] = 0;
			}
		}
	}

	// Check for memory availability
	if(i_D2 > MAXNNZS*3)
		printf("Memory overflow (nonzeros) while separating D sections!\n");
	if(row > MAXSIZE*3)
		printf("Memory overflow (row indices) while separating D sections!\n");
}

void preprocessR_PSTRSV3(double *a, MKL_INT *ia, MKL_INT *ja, int nthreads, const char uplo)
{
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
	char withinD3, hasReflection, isOptimized;
	int t, i, row = 1, col, prefix, border, end_interval, i_R, partition_start, partition_end;
	prefix = starts_R[0];
	i_R = ID[prefix];

	// Produce R and set start & end indices for each thread
	if(uplo == 'u' || uplo == 'U')
	{
		for(t=0;t<nthreads;t++)
		{
			hasReflection = hasReflections[t];
			isOptimized = isOptimizeds[t];
			starts_R[t] = prefix + row - 1;
			partition_start = partition_starts[t];
			partition_end = partition_ends[t];
			end_interval = end_intervals[t];
			border = end_intervals[t] + partition_start;
			for(i=ia[partition_start];i<ia[partition_end];i++)
			{
				col = ja[i];
				if(i == ia[row])
				{
					ID[prefix + row] = i_R;
					row++;
				}
				withinD3 = ((hasReflection || isOptimized) && border <= col && border > (row - 1)) ? 1 : 0;
				if(withinD3 || partition_end <= col)
				{
					hasDependences[t] = 1;
					D[i_R] = a[i];
					JD[i_R] = col;
					i_R++;
				}
			}
			ID[prefix + row] = i_R;
			ends_R[t] = prefix + row;
			row++;
		}
	}
	else // uplo == 'l' || uplo == 'L'
	{
		//TODO: Rewrite this part to produce R_new = R_old + D3
		for(t=0;t<nthreads;t++)
		{
			hasReflection = hasReflections[t];
			isOptimized = isOptimizeds[t];
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
				if(partition_end > col)
				{
					D[i_R] = a[i];
					JD[i_R] = col;
					i_R++;
				}
			}
			ID[prefix + row] = i_R;
			ends_R[t] = prefix + row;
			row++;
		}
	}
}

void preprocessMappings_PSTRSV3(int nthreads) 
{
	/*
		Determine the number of nonzeros of R matrix partition
		and allocate the necessary memory for mapR & compressedX
		Note:
			mapR represents both R_dense -> R & compressedX -> x
			Therefore, len(mapR) = len(compressedX) 
	*/
	int t, colCount, mappingsSize = 0;

	// Determine start & end points
	for(t=0;t<nthreads;t++)
	{
		colCount = numCols_S[t]; // since numCols_R_dense[t] = numCols_S[t];
		starts_mapR[t] = mappingsSize;
		mappingsSize += colCount;
		ends_mapR[t] = mappingsSize;
	}

	// Allocate necessary space for mapR & compressedX
	mappingsSize += 16 - (mappingsSize % 16);
	mapR = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*mappingsSize, 64);
	compressedX = (double*)mkl_malloc(sizeof(double)*mappingsSize, 64);
	memset(mapR,0,sizeof(MKL_INT)*mappingsSize);
	memset(compressedX,0,sizeof(double)*mappingsSize);
}

void produceCompressedR_PSTRSV3(int n, int nthreads, const char uplo)
{
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
	char hasToCalculateS;
	int i, t, col, counter, partition_start, partition_end, start_R, end_R, start_map, row, colCount, start_Rdense;
	MKL_INT *tempMap;

	tempMap = (MKL_INT*)mkl_malloc(sizeof(MKL_INT)*(n + 16 - (n % 16)), 64);
	memset(tempMap,0,sizeof(MKL_INT)*n);

	// Produce dense partitions of R matrix for each thread 
	for(t=0;t<nthreads;t++)
	{
		hasToCalculateS = hasToCalculateSs[t];
		if(hasToCalculateS)
		{
			partition_start = partition_starts[t];
			partition_end = partition_ends[t];
			start_R = starts_R[t] + start_intervals[t];
			end_R = starts_R[t] + end_intervals[t];
			memset(tempMap,0,sizeof(MKL_INT)*n);

			// Determine columns with nonzeros
			for(i=ID[start_R];i<ID[end_R];i++)
			{
				col = JD[i];
				if(!(partition_start <= col && col < partition_end))
					tempMap[col] = 1; // mapping for each thread regardless of uplo value
			}

			// Store the original sparse indices in corresponding dense indices of mapR
			// and temporarly store the vice versa by overwriting tempMap
			counter = 0;
			start_map = starts_mapR[t];
			for(i=0;i<n;i++)
			{
				if(tempMap[i])
				{
					mapR[start_map + counter] = i;
					tempMap[i] += counter; // index = tempMap[i] - 1;
					counter++;
				}
			}

			// Produce compressed R matrix partition
			row = 0;
			colCount = numCols_S[t];
			start_Rdense = starts_S[t];
			for(i=ID[start_R];i<ID[end_R];i++)
			{
				col = JD[i];
				while(i == ID[start_R + row + 1])
					row++;
				if(!(partition_start <= col && col < partition_end))
					R_dense[start_Rdense + ((row * colCount) + (tempMap[col] - 1))] = D[i];
			}
		}
	}
}

void produceS_PSTRSV3(int nthreads, const char uplo)
{
	/*
		Produces partial S of DS Factorization sequentially.
		(S = D'A -> DS = A)
		Since D partitions are triangular systems, 
		there is no need to calculate the inverse 
		of D sections. Instead, triangular solvers
		calculate dense S parts.
	*/
	int t, start_D0, start_R, start_S, rowCount, colCount;
	double alpha = 1;
	const char transa = 'N';
	const char diag = 'N';		
	char matdescra[6] = {'T', uplo, 'N', 'C', 0, 0};
	char hasToCalculateS;

	for(t=0;t<nthreads;t++)
	{
		start_D0 = starts_D0[t];
		start_R = starts_S[t];
		start_S = starts_S[t];
		rowCount = numRows_S[t];
		colCount = numCols_S[t];
		hasToCalculateS = hasToCalculateSs[t];
		if(hasToCalculateS && rowCount > 0)
			mkl_dcsrsm(&transa, &rowCount, &colCount, &alpha, matdescra, &T2[IT2[start_D0]], &JT2[IT2[start_D0]], &IT2[start_D0], &IT2[start_D0 + 1], &R_dense[start_R], &colCount, &S[start_S], &colCount);
	}
}

void produceS_PSTRSV3_Parallel_OLD(int nthreads, const char uplo)
{
	/*
		Produces partial S of DS Factorization in parallel.
		(S = D'A -> DS = A)
		Since D partitions are triangular systems, 
		there is no need to calculate the inverse 
		of D sections. Instead, triangular solvers
		calculate dense S parts.
	*/
	double alpha = 1;
	const char transa = 'N';
	const char diag = 'N';		
	char matdescra[6] = {'T', uplo, 'N', 'C', 0, 0};
	omp_set_num_threads(nthreads);

	#pragma omp parallel
	{
		int t = omp_get_thread_num();
		int start_D0 = starts_D0[t];
		int start_R = starts_S[t];
		int start_S = starts_S[t];
		int rowCount = numRows_S[t];
		int colCount = numCols_S[t];
		int hasToCalculateS = hasToCalculateSs[t];
		if(hasToCalculateS && rowCount > 0)
			mkl_dcsrsm(&transa, &rowCount, &colCount, &alpha, matdescra, &T2[IT2[start_D0]], &JT2[IT2[start_D0]], &IT2[start_D0], &IT2[start_D0 + 1], &R_dense[start_R], &colCount, &S[start_S], &colCount);
	}
}

void produceS_PSTRSV3_Parallel(int nthreads, const char uplo)
{
	/*
		Produces partial S of DS Factorization in parallel.
		(S = D'A -> DS = A)
		Since D partitions are triangular systems, 
		there is no need to calculate the inverse 
		of D sections. Instead, triangular solvers
		calculate dense S parts.
	*/
	int t, i, temp, rowCount, colCount, hasToCalculateS, start_D0, start_R, start_S;
	int newStarts[MAXTHREADCOUNT], newColCounts[MAXTHREADCOUNT];
	omp_set_num_threads(nthreads);
	mkl_set_num_threads(1);

	for(t=0;t<nthreads;t++)
	{
		rowCount = numRows_S[t];
		colCount = numCols_S[t];
		hasToCalculateS = hasToCalculateSs[t];
		start_D0 = starts_D0[t];
		start_R = starts_S[t];
		start_S = starts_S[t];

		if(hasToCalculateS && rowCount > 0)
		{
			temp = numCols_S[t] / nthreads;
			for(i=0;i<nthreads;i++)
				newColCounts[i] = temp; //(numCols_S[i] % nthreads > i) ? temp + 1 : temp;
			for(i=0;i<numCols_S[t] % nthreads;i++)
				newColCounts[i]++;
			newStarts[0] = 0;
			for(i=1;i<nthreads;i++)
				newStarts[i] = newStarts[i-1] + newColCounts[i-1];

			#pragma omp parallel
			{
				int th_id = omp_get_thread_num();
				int newStart = newStarts[th_id];
				int newColCount = newColCounts[th_id];

				__udsol2(rowCount, newColCount, &R_dense[start_R + newStart], T2, &IT2[start_D0], JT2, colCount);
			}
		}
	}

	// Pointer assignment for memory efficiency
	S = R_dense;
}

void produceReducedSystem_PSTRSV3(int n, int nthreads, char const uplo) 
{
	/*
		Produces the reduced system of dependent elements
		and stores it in T, IT, and JT with CSR format.
	*/
	int i, j, t, i_S, temp, row = 1, index = 0, col, row_S, partition_start, partition_end, rowCount, colCount, start_mapR, start_S;
	char hasReflection, hasToCalculateS, firstEntrance;
	double val;

	// Modify the dependents array to use it as the mapping for the reduced system
	for(i=0;i<n;i++)
	{
		if(dependents[i])
		{
			mapReducedX[index] = i; // reducedX -> x
			index++;
			dependents[i] = index;
		}
	}
	numRowsReducedSystem = index;

	// Build the reduced system thread by thread
	IT[0] = 0;
	index = 0;
	if(uplo == 'u' || uplo == 'U')
	{
		for(t=1;t<nthreads-1;t++)
		{
			// Assign global values to local ones for the effective usage of memory chunks during iterations
			partition_start = partition_starts[t];
			partition_end = partition_ends[t];
			rowCount = numRows_S[t];
			colCount = numCols_S[t];
			start_mapR = starts_mapR[t];
			start_S = starts_S[t];	
			hasReflection = hasReflections[t];
			hasToCalculateS = hasToCalculateSs[t];	
			firstEntrance = 1;
	
			for(i=partition_start;i<partition_end;i++)
			{	
				if(dependents[i])
				{
					if(firstEntrance)
					{	
						temp = i;
						firstEntrance = 0;
					}
					row_S = i - temp;
					if(hasReflection && hasToCalculateS && row_S < rowCount)
					{
						i_S = start_S + row_S * colCount; 
						for(j=0;j<colCount;j++)
						{
							val = S[i_S + j];
							if(val != 0)
							{
								T[index] = val;
								JT[index] = dependents[mapR[start_mapR + j]] - 1;
								index++;
							}
						}
					}
					IT[row] = index;
					row++; 
				}
			}
		}

		// Last thread does not have intersection, so only need to indicate the starting point
		reducedSystemSolverStart = row - 1;
	}
	else // uplo == 'l' || uplo == 'L'
	{
		//TODO: Implement lower triangular support
	}

	numNnzsReducedSystem = index + numRowsReducedSystem;
}

inline void produceReducedY_PSTRSV3(double *reducedRHS, double *originalRHS)
{
	/*
		Produces reduced right-hand side vector containing only the
		corresponding elements of the original one with respect to
		mapReducedX.
	*/
	int i;

	#pragma omp for simd
	for(i=0;i<numRowsReducedSystem;i++)
		reducedRHS[i] = originalRHS[mapReducedX[i]];
}

inline void updateX(double *reducedX, double *originalX) 
{
	/*
		Updates the original solution vector using the reducedX.
		mapReducedX is used to map reducedX values into the original x.
	*/
	int i;

	#pragma omp for simd
	for(i=0;i<reducedSystemSolverStart;i++)
		originalX[mapReducedX[i]] = reducedX[i];
}

inline void __udsol(int n, double *x, double *b, double *A, MKL_INT *IA, MKL_INT *JA)
{
	/*
		Upper triangular solver: U*x = b
	*/
	register double t;
	int k, j;
	x[n-1] = b[n-1] / A[IA[n-1]];
	for(k=n-2;k>=0;k--)
	{
		t = b[k];
		for(j=IA[k]+1;j<IA[k+1];j++)
			t -= A[j]*x[JA[j]];
		x[k] = t / A[IA[k]];
	}
}

inline void __udsol2(int rowCount, int colCount, double *X, double *A, MKL_INT *IA, MKL_INT *JA, int stepsize)
{
	/*
		Upper triangular solver: U*X = B where X will be directly stored in B
	*/
	register double t, *tempX, *tempX2;
	register int k, j, l, col;
	tempX = &X[(rowCount-1) * stepsize];

	#pragma omp simd
	for(l=0;l<colCount;l++)
		tempX[l] /= A[IA[rowCount-1]];

	for(k=rowCount-2;k>=0;k--)
	{
		tempX = &X[k * stepsize];

		for(j=IA[k]+1;j<IA[k+1];j++)
		{
			t = A[j];
			col = JA[j];
			tempX2 = &X[col * stepsize];

			#pragma omp simd
			for(l=0;l<colCount;l++)
				tempX[l] -= t * tempX2[l];
		}
		
		#pragma omp simd
		for(l=0;l<colCount;l++) 
			tempX[l] /= A[IA[k]];
	}
}

inline void __usol(int n, double *x, double *b, double *A, MKL_INT *IA, MKL_INT *JA)
{
	/*
		Upper triangular solver: U*x = b for the unit diagonal matrix U
	*/
	register double t;
	register int k, j;
	x[n-1] = b[n-1];
	for(k=n-2;k>=0;k--)
	{
		t = b[k];
		for(j=IA[k]+1;j<IA[k+1];j++)
			t -= A[j]*x[JA[j]];
		x[k] = t;
	}
}

inline void __usol2(int n, double *x, double *A, MKL_INT *IA, MKL_INT *JA)
{
	/*
		Upper triangular solver: U*x = b for the unit diagonal matrix U
		with single vector for both the right-hand side & the solution
	*/
	register double t;
	register int k, j;
	for(k=reducedSystemSolverStart;k>=0;k--)
	{
		t = 0;
		for(j=IA[k];j<IA[k+1];j++)
			t += A[j]*x[JA[j]];
		x[k] -= t;
	}
}

inline void __ldsol(int n, double *x, double *b, double *A, MKL_INT *IA, MKL_INT *JA)
{
	/*
		Lower triangular solver: L*x = b
	*/
	double t;
	int k, j;
	x[0] = b[0] / A[IA[0]];
	for(k=1;k<n;k++)
	{
		t = b[k];
		for(j=IA[k];j<IA[k+1]-1;j++)
			t -= A[j]*x[JA[j]];
		x[k] = t / A[IA[k+1]-1];
	}
}

void pmemset_d(double *array, int x, int n, int nthreads)
{
	/*
		Parallel memory initializer for double pointers
	*/
	omp_set_num_threads(nthreads);
	#pragma omp parallel
	{
		int th_id = omp_get_thread_num();
		int step = n / nthreads;
		int leftover = (th_id == nthreads-1) ? (step % nthreads) : 0;
		memset(&array[th_id*step],x,sizeof(double)*(step+leftover));
	}
}

void pmemset_i(MKL_INT *array, int x, int n, int nthreads)
{
	/*
		Parallel memory initializer for MKL_INT pointers
	*/
	omp_set_num_threads(nthreads);
	#pragma omp parallel
	{
		int th_id = omp_get_thread_num();
		int step = n / nthreads;
		int leftover = (th_id == nthreads-1) ? (step % nthreads) : 0;
		memset(&array[th_id*step],x,sizeof(MKL_INT)*(step+leftover));
	}
}

void logPSTRSV3(int nthreads)
{
	/*
		spike_pstrsv logger
	*/
	int t;
	printf("* * * * * * * * * * * * * * * * * * * * * * \n");
	printf("* Matrix Information						\n");
	printf("* ------------------						\n");
	printf("* Reduced system size: %d x %d				\n", numRowsReducedSystem, numRowsReducedSystem);
	printf("* Reduced system nnzs: %d 					\n", numNnzsReducedSystem);
	for(t=0;t<nthreads;t++)
	{
		printf("* S[%d] size: %d x %d					\n", t, numRows_S[t], numCols_S[t]);
		printf("* isOptimized[%d]: %d					\n", t, isOptimizeds[t]);
		printf("* hasReflection[%d]: %d					\n", t, hasReflections[t]);
		printf("* hasDependences[%d]: %d				\n", t, hasDependences[t]);
		printf("* hasToCalculateS[%d]: %d				\n", t, hasToCalculateSs[t]);
		printf("* Intervals start[%d]: %d, end[%d]: %d	\n", t, start_intervals[t], t, end_intervals[t]);
		printf("* Partition start[%d]: %d, end[%d]: %d	\n", t, partition_starts[t], t, partition_ends[t]);
	}
	printf("* * * * * * * * * * * * * * * * * * * * * * \n\n");
}