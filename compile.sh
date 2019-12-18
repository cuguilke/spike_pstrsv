#!/bin/bash
echo ""
echo "                  ------------------------------                   "
echo "                 |                              |                  "
echo "-----------------          spike_pstrsv          ------------------"
echo "                 |                              |                  "
echo "                  ------------------------------                   " 
echo ""
echo "-------------------------------------------------------------------"

for arg in "$@"
do
	if [ "$arg" == "--help" ]
	then
		echo "Available modes: "
		echo "   --runtime_analysis : creates the executable for performance profiling"
		echo "   --example_usage    : created the executable for example usage"
	fi
	if [ "$arg" == "--runtime_analysis" ]
	then
		echo "Creating... runtimeAnalysis"
		cp Makefile.profiler Makefile
		make distclean
		make
		rm Makefile
		echo "Done."
	fi
	if [ "$arg" == "--example_usage" ]
	then
		echo "Creating... exampleUsage"
		cp Makefile.sample Makefile
		make distclean
		make
		rm Makefile
		echo "Done."
	fi
done