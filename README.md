# spike_pstrsv

By [Ilke Cugu](http://user.ceng.metu.edu.tr/~e1881739/) and [Murat Manguoglu](http://user.ceng.metu.edu.tr/~manguoglu/).

## Table of Contents

1. [Introduction](#introduction)
2. [Citation](#citation)
3. [API](#api)

## Introduction

spike_pstrsv is a parallel sparse triangular linear system solver based on the Spike algorithm. This repository contains the codes described in the paper "A parallel multithreaded sparse triangular linear system solver" (https://www.sciencedirect.com/science/article/pii/S0898122119304602).

**Full list of items**
  * MicroExpNet.py: The original source code of the proposed FER model
  * exampleUsage.py: A script to get prediction from a pre-trained MicroExpNet for an unlabeled image
  * Models: Pre-trained MicroExpNet models for CK+ and Oulu-CASIA datasets.
  * Candidates: Candidate networks build in search of a better FER model
  
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

## API
**spike_pstrsv(const char uplo, int m, double \*a, MKL_INT \*ia, MKL_INT \*ja, double \*b, double \*x, int nthreads)**

This routine solves a triangular system of linear equations with matrix-vector operations for a sparse matrix stored in the compressed sparse row (CSR) format (3 array variation):

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
  
  
