#ifndef SPIKE_PSTRSV_GLOBALS_H   /* Include guard */
#define SPIKE_PSTRSV_GLOBALS_H

#define MAXSIZE 4194304 							// defines max acceptable number of rows and columns 
#define MAXNNZS 25165824 							// defines max acceptable number of nonzeros 
#define MAXTHREADCOUNT 128
#define INF 2147483640
#define MEMSIZE 17179869184
#define max(a, b) (((a) > (b)) ? (a) : (b))

// 1.9 GB of Stack is required for the settings below.
//----------------------------------------------------------------------------------------------//
extern MKL_INT partition_numRows[MAXTHREADCOUNT];	// row count per thread
extern MKL_INT partition_starts[MAXTHREADCOUNT];	// start indices of A matrix for each thread
extern MKL_INT partition_ends[MAXTHREADCOUNT];		// end indices of A matrix for each thread
extern double D[MAXNNZS*2]; 						// will be used to store D of DS Factorization
extern MKL_INT ID[MAXSIZE*2];
extern MKL_INT JD[MAXNNZS*2];
extern MKL_INT dependents[MAXSIZE]; 				// will be used to indicate the elements of the reduced system
extern double T[MAXNNZS]; 							// will be used to store reduced system of spike_pstrsv
extern MKL_INT IT[MAXSIZE]; 
extern MKL_INT JT[MAXNNZS];
extern double T2[3*MAXNNZS];						// will be used to store D0, D1, D2 and D3 of spike_pstrsv
extern MKL_INT IT2[3*MAXSIZE];
extern MKL_INT JT2[3*MAXNNZS];
extern MKL_INT starts_D[MAXTHREADCOUNT];			// points to ID array
extern MKL_INT ends_D[MAXTHREADCOUNT];				// points to ID array
extern MKL_INT starts_R[MAXTHREADCOUNT];			// points to ID array
extern MKL_INT ends_R[MAXTHREADCOUNT];				// points to ID array
extern char isReducedSystemNecessary;
extern char hasDependences[MAXTHREADCOUNT];	
extern char hasReflections[MAXTHREADCOUNT];
extern double *S; 									// matrix S of DS Factorization
extern MKL_INT numRows_S[MAXTHREADCOUNT]; 			// will be used to store 1st dim of S parts per thread
extern MKL_INT numCols_S[MAXTHREADCOUNT]; 			// will be used to store 2nd dim of S parts per thread
extern long long starts_S[MAXTHREADCOUNT]; 			// indices of each thread's start point of S sections
extern long long ends_S[MAXTHREADCOUNT]; 			// indices of each thread's end point of S sections
extern MKL_INT starts_D0[MAXTHREADCOUNT];  			// points to IT2 array
extern MKL_INT ends_D0[MAXTHREADCOUNT];				// points to IT2 array
extern MKL_INT starts_D1[MAXTHREADCOUNT];  			// points to IT2 array
extern MKL_INT ends_D1[MAXTHREADCOUNT];				// points to IT2 array
extern MKL_INT starts_D2[MAXTHREADCOUNT]; 			// points to IT2 array
extern MKL_INT ends_D2[MAXTHREADCOUNT];				// points to IT2 array
extern double *compressedX;
extern MKL_INT *mapR;								// will be used to store mapping of R_dense -> R & compressedX -> x
extern MKL_INT starts_mapR[MAXTHREADCOUNT]; 		// will be used for both R_dense & compressedX 
extern MKL_INT ends_mapR[MAXTHREADCOUNT]; 			// will be used for both R_dense & compressedX
extern double reducedX[MAXSIZE];
extern MKL_INT numRowsReducedSystem;
extern MKL_INT start_intervals[MAXTHREADCOUNT];		// holds minDependentRows
extern MKL_INT end_intervals[MAXTHREADCOUNT];		// holds maxIntersectionRows
extern char isOptimizeds[MAXTHREADCOUNT]; 			// indicates if the thread has D1, D2 and D3 partitions despite its hasReflection = false
extern char PSTRSV_VERBOSE;							// environment variable to enable PSTRSV3 operations log
extern char FORCE_LOAD_BALANCE;						// environment variable to enable further partitioning of D matrix to improve the load-balance
extern char IS_MATRIX_PARTITIONED;					// if > 0 -> matrixPartitioning function creates partitions
//----------------------------------------------------------------------------------------------//

/* 								Runtime analysis variables 										*/
//----------------------------------------------------------------------------------------------//
extern double memoryAlloc_timer_pstrsv3;
extern double preprocessD_timer_pstrsv3;
extern double preprocessR_timer_pstrsv3;
extern double setDependents_timer_pstrsv3;
extern double preprocessS_timer_pstrsv3;
extern double preprocessMappings_timer_pstrsv3;
extern double produceCompressedR_timer_pstrsv3;
extern double produceS_timer_pstrsv3;
extern double produceReducedSystem_timer_pstrsv3;
extern double init_vars_pstrsv3; 
extern double reduced_solver_timer_pstrsv3;
extern double temp_timer[MAXTHREADCOUNT];
extern double init_vars_pstrsv3;
extern double waitBefore_timer_pstrsv3[MAXTHREADCOUNT];
extern double computeY_timer[MAXTHREADCOUNT];
extern double solveIndependent_timer[MAXTHREADCOUNT];
extern double subtract_Rx_timer[MAXTHREADCOUNT];
extern double waitAfter_timer_pstrsv3[MAXTHREADCOUNT];
extern double solveD2_timer[MAXTHREADCOUNT];
extern double solveD_timer[MAXTHREADCOUNT];
//----------------------------------------------------------------------------------------------//
#endif