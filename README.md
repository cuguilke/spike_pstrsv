# spike_pstrsv

By [Ilke Cugu](http://user.ceng.metu.edu.tr/~e1881739/) and [Murat Manguoglu](http://user.ceng.metu.edu.tr/~manguoglu/).

## Table of Contents

1. [Introduction](#introduction)
2. [Citation](#citation)
3. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [API](#api)
7. [Tools](#tools)

## Introduction

**spike_pstrsv** is a parallel sparse triangular linear system solver based on the Spike algorithm. This repository contains the codes described in the paper "A parallel multithreaded sparse triangular linear system solver" (https://www.sciencedirect.com/science/article/pii/S0898122119304602).

**Full list of items**
  * src: The original source codes of spike_pstrsv
  * test: Performance profiling codes
  * tools: Python/Bash scripts to generate performance plots and result tables (in LaTeX format)
  * compile.sh: Bash script to compile the codes (--help: for detailed explanations)
  * example_usage.c: Sample code to read a test matrix in MatrixMarket format and to run spike_pstrsv
  * Makefile.profiler: Makefile for the profiler
  * Makefile.sample: Makefile for a quick test drive 
  
## Citation

If you use this solver in your research, please cite:

```
@article{CUGU2019,
  title = "A parallel multithreaded sparse triangular linear system solver",
  journal = "Computers & Mathematics with Applications",
  year = "2019",
  issn = "0898-1221",
  doi = "https://doi.org/10.1016/j.camwa.2019.09.012",
  url = "http://www.sciencedirect.com/science/article/pii/S0898122119304602",
  author = "İlke Çuğu and Murat Manguoğlu",
  keywords = "Sparse triangular linear systems, Direct methods, Parallel computing",
  abstract = "We propose a parallel sparse triangular linear system solver based on the Spike algorithm. Sparse triangular systems are required to be solved in many applications. Often, they are a bottleneck due to their inherently sequential nature. Furthermore, typically many successive systems with the same coefficient matrix and with different right hand side vectors are required to be solved. The proposed solver decouples the problem at the cost of extra arithmetic operations as in the banded case. Compared to the banded case, there are extra savings due to the sparsity of the triangular coefficient matrix. We show the parallel performance of the proposed solver against the state-of-the-art parallel sparse triangular solver in Intel’s Math Kernel Library (MKL) on a multicore architecture. We also show the effect of various sparse matrix reordering schemes. Numerical results show that the proposed solver outperforms MKL’s solver in ∼80% of cases by a factor of 2.47, on average."
}
```

## Prerequisites
For spike_pstrsv:
```
OpenMP
Intel MKL (>=2018)
```
For performance profiling using several reordering algorithms:
```
METIS
```
For the tools:
```
Python 2.7
matplotlib
numpy
```

## Installation
We currently have 2 executable creation modes for quick installation of spike_pstrsv:

**runtimeAnalysis**
 - To compile:
 ```
 ./compile.sh --runtime_analysis
 ```
 - To run:
 ```
 runtimeAnalysis <{matrix_filename}.mtx> <thread count> <uplo{U,u:L,l}> <symmetric{Y,y:S,s:N,n}> <METIS usage{Y,y:N,n}>
 ```
 
**exampleUsage**
- To compile:
 ```
 ./compile.sh --example_usage
 ```
 - To run:
 ```
 exampleUsage <{matrix_filename}.mtx> <thread count> <uplo{U,u:L,l}> <symmetric{Y,y:S,s:N,n}>
 ```

## API
**spike_pstrsv(const char uplo, int m, double \*a, MKL_INT \*ia, MKL_INT \*ja, double \*b, double \*x, int nthreads)**

This routine first calls the preprocessor, then solves the given sparse triangular system of linear equations which is stored in the compressed sparse row (CSR) format (3 array variation):

```
A*x = b
```

**Parameters**
  - uplo: Specifies whether the upper or low triangle of the matrix A is used. 
    - 'U' or 'u' for upper triangle of the matrix A
    - 'L' or 'l' for lower triangle of the matrix A
  - m: Number of rows of the matrix A.
  - a: Array containing non-zero elements of the matrix A. (Array, size is ia[m])
  - ia: Array containing indices of the elements in the array a. (Array, size is m + 1)
  - ja: Array containing the column indices of the non-zero element of the matrix A. (Array, size is ia[m])
  - b: Right-hand side vector. (Array, size is m)
  - x: Solution vector. (Array, size is m)
  - nthreads: Number of OpenMP threads to be used by the solver.
***
**spike_pstrsv_preproc(double \*a, MKL_INT \*ia, MKL_INT \*ja, int n, int nthreads, const char uplo)**

Preprocessing routine for spike_pstrsv_solve. In this routine, we handle operations that are independent from the right hand side vector.  This splitting is useful when it is used in an iterative scheme, preprocessing is done only once and the solver is often called multiple times. 

**Parameters**
  - a: Array containing non-zero elements of the matrix A. (Array, size is ia[n])
  - ia: Array containing indices of the elements in the array a. (Array, size is n + 1)
  - ja: Array containing the column indices of the non-zero element of the matrix A. (Array, size is ia[n])
  - n: Number of rows of the matrix A.
  - nthreads: Number of OpenMP threads to be used by the solver.
  - uplo: Specifies whether the upper or low triangle of the matrix A is used. 
   - 'U' or 'u' for upper triangle of the matrix A
   - 'L' or 'l' for lower triangle of the matrix A
***
**spike_pstrsv_solve(const char uplo, int m, double \*b, double \*x, int nthreads)**

The proposed parallel sparse triangular system solver. 

**Parameters**
  - uplo: Specifies whether the upper or low triangle of the matrix A is used. 
   - 'U' or 'u' for upper triangle of the matrix A
   - 'L' or 'l' for lower triangle of the matrix A
  - m: Number of rows of the matrix A.
  - b: Right-hand side vector. (Array, size is m)
  - x: Solution vector. (Array, size is m)
  - nthreads: Number of OpenMP threads to be used by the solver.

## Tools
This folder is reserved for tools to analyze the spike_pstrsv logs mainly for profiling.

**Content**
 - plotSpeedUp.py: Python script that generates performance plots & result tables of runtime analysis logs of spike_pstrsv
 - auto_experiments.sh: Bash script to run detailed performance profiling experiments using different matrix reordering algorithms.
