#!/bin/sh
MATRIX_FOLDER=/home/ilke/Matrices/Literature
echo "******************* CASE-1: ORIGINAL ******************"
for matrix in ${MATRIX_FOLDER}/Original/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U Y N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/Original/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U S N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/Original/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U N N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

echo "******************* CASE-2: METIS ******************"
for matrix in ${MATRIX_FOLDER}/Original/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U Y Y
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/Original/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U S Y
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/Original/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U N Y
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

echo "******************* CASE-3: RCM ******************"
for matrix in ${MATRIX_FOLDER}/RCM/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U Y N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/RCM/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U S N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done
-----------------------
for matrix in ${MATRIX_FOLDER}/RCM/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U N N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

echo "******************* CASE-4: AMD ******************"
for matrix in ${MATRIX_FOLDER}/AMD/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U Y N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/AMD/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U S N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/AMD/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U N N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

echo "******************* CASE-5: ColPerm ******************"
for matrix in ${MATRIX_FOLDER}/ColPerm/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U Y N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/ColPerm/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U S N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/ColPerm/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U N N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

echo "******************* CASE-6: NDP ******************"
for matrix in ${MATRIX_FOLDER}/NDP/Symmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U Y N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/NDP/PatternSymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U S N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done

for matrix in ${MATRIX_FOLDER}/NDP/Unsymmetric/*.mtx; do
	echo "######-START: Tests on Upper Triangular "$matrix"-######"
	for nthreads in 2 4 8 10 16 20; do
		echo "NTHREAD: "$nthreads
		numactl --cpunodebind=1 ./runtimeAnalysis $matrix $nthreads U N N
	done
	echo "######-END: Tests on "$matrix"-######"
	echo ""
	echo ""
	echo ""
done
echo "******************* BENCHMARK END ******************"
