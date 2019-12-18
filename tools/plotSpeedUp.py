'''
	Title           :plotSpeedUp.py
	Description     :This script generates performance plots & result tables of runtime analysis logs of spike_pstrsv
	Author          :Ilke Cugu
	Date Created    :12-02-2017
	Date Modified   :15-11-2019
	version         :5.3
	python_version  :2.7.11
'''

import sys
import math
import argparse
import itertools
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pylab as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Default configurations
DEBUG = False
VERBOSE = False
USEAVG = True
CHARTSIZE = 5
NTHREADS = [2, 4, 8, 10, 16, 20]
SOLVERS = ["MKL_SERIAL", "MKL_PARALLEL", "SPARSKIT", "PSTRSV3"]
REORDERINGS = ["METIS", "RCM", "AMD", "ColPerm", "NDP"]
POSTFIXES = ["", "_RCM", "_AMD", "_ColPerm", "_NDP"]
OVERVIEW_MODE = False
CASE_STUDY_MODE = True
ONE_CHART_FOR_ALL = True
FULL_APPENDIX_MODE = False
MEMORY_ERROR_CHECK_MODE = True

tableau20 = [(0, 70, 105), (0, 95, 135), (255, 127, 14), (255, 187, 120), (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148), (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 

def errorLog():
	global singulars, epicFails

	if len(singulars) > 0:
		print
		print "Singular Matrices:"
		print "================="
		singulars.sort()
		for matrix in singulars:
			print matrix
		print

	if len(epicFails) > 0:
		print
		print "Epic Fails of pSTRSV:"
		print "===================="
		epicFails.sort()
		for matrix in epicFails:
			print matrix
		print

def processPerformanceParams(parent, perfParamContent):
	global gPerfParams
	temp = 0
	case = parent[0]
	postfix = "_" + case
	matrix = parent[1] if postfix in parent[1] else parent[1] + postfix
	nThreads = int(parent[2])

	# Parse the content string
	perThread = {}
	perfParams = []
	while "Matrix Information" not in perfParamContent[temp]:
		temp += 1
	perfParamContent = perfParamContent[temp + 2:]
	k = int(perfParamContent[0][23:].split(" x ")[0])
	nnzReducedSystem = int(perfParamContent[1].split(": ")[1])
	for thID in range(nThreads):
		startIndex = 1 + (thID * 7)
		sizeMatrixS = perfParamContent[startIndex + 1].split("\t")[0].split(": ")[1]
		isOptimized = True if perfParamContent[startIndex + 2].split("\t")[0].split(": ")[1] == "1" else False
		hasReflection = True if perfParamContent[startIndex + 3].split("\t")[0].split(": ")[1] == "1" else False
		hasDependence = True if perfParamContent[startIndex + 4].split("\t")[0].split(": ")[1] == "1" else False
		hasToCalculateS = True if perfParamContent[startIndex + 5].split("\t")[0].split(": ")[1] == "1" else False 
		interval = perfParamContent[startIndex + 6].split(", ")
		intervalStart = int(interval[0].split(": ")[-1])
		intervalEnd = int(interval[1].split("\t")[0].split(": ")[1])
		perThread = {"sizeMatrixS": sizeMatrixS, "isOptimized": isOptimized, "hasReflection": hasReflection, "hasDependence": hasDependence, "hasToCalculateS": hasToCalculateS, "intervalStart": intervalStart, "intervalEnd": intervalEnd} 
		perfParams.append(perThread)
	info = {"reducedSystemSize": k, "reducedSystemNnzs": nnzReducedSystem, "infoPerThread": perfParams}

	# Add structured info also into the global dictionary of performance constraints
	i = matrix.rfind("_")
	firstKey = matrix[:i]
	secondKey = matrix[i+1:]
	sizeMatrixS = 0
	sizeString = "0 x 0"
	for infoPerThread in info["infoPerThread"]:
		vals = infoPerThread["sizeMatrixS"].split(" x ")
		newSize = int(vals[0]) * int(vals[1])
		if newSize > sizeMatrixS:
			sizeMatrixS = newSize
			sizeString = infoPerThread["sizeMatrixS"]
	if firstKey in gPerfParams:
		if secondKey in gPerfParams[firstKey]:
			gPerfParams[firstKey][secondKey].update({nThreads: sizeString})
		else:
			gPerfParams[firstKey].update({secondKey: {nThreads: sizeString}})
	else:
		gPerfParams.update({firstKey: {secondKey: {nThreads: sizeString}}})

	return info

def processProfile(parent, profileContent):
	global runtimeResults
	isSolving= False
	isPreprocessing = False
	totalPreprocessingTime = 0.0
	totalPreprocessingTimeMKL = 0.0
	solvingDict = {}
	preprocessingDict = {}
	for line in profileContent:
		if "ms" not in line:
			isSolving = False
			isPreprocessing = False
		elif "Total preprocessing" in line: 
			isPreprocessing = True
		elif "Total runtime" in line:
			isSolving = True

		if isPreprocessing:
			tmp = line.split(": ")
			key = tmp[0].split(" runtime")[0][2:]
			val = float(tmp[1].split(" ms")[0])
			val = .0 if math.isnan(val) else val
			preprocessingDict.update({key: val})
			
			if "Total" not in key: # Total time includes memory init with 0s
				if "memoryAlloc" in key:
					totalPreprocessingTime -= val
				else:
					totalPreprocessingTime += val
			elif "MKL" in key:
				totalPreprocessingTimeMKL = val
		elif isSolving:
			tmp = line.split(": ")
			key = tmp[0]
			val = float(tmp[1].split(" ms")[0])
			val = .0 if math.isnan(val) else val
			solvingDict.update({key: val})

	# Preprocessing times of PSTRSV for runtime results table
	reordering = parent[0]
	matrix = parent[1]
	nThreads = parent[2]
	if reordering in matrix:
		matrix = matrix[:matrix.index(reordering)-1]

	if matrix in runtimeResults:
		if reordering in runtimeResults[matrix]:
			runtimeResults[matrix][reordering].update({nThreads: {"preprocessPSTRSV": totalPreprocessingTime, "preprocessMKL": totalPreprocessingTimeMKL}})
		else:
			runtimeResults[matrix].update({reordering: {nThreads: {"preprocessPSTRSV": totalPreprocessingTime, "preprocessMKL": totalPreprocessingTimeMKL}}})
	else:
		runtimeResults.update({matrix: {reordering: {nThreads: {"preprocessPSTRSV": totalPreprocessingTime, "preprocessMKL": totalPreprocessingTimeMKL}}}})
	
	return {"preprocessing": preprocessingDict, "solution": solvingDict}

def processComparison(parent, comparisonContent):
	global runtimeResults
	structuredContent = {}
	for i in range(len(comparisonContent)):
		temp = comparisonContent[i].split(" \t| ")
		bestRuntime = float(temp[-2].split(" ")[-2])
		avgRuntime = float(temp[-1].split(" ")[-2])
		runtime = avgRuntime if USEAVG else bestRuntime
		structuredContent.update({SOLVERS[i] : runtime})

	# Runtimes of solvers for runtime results table
	reordering = parent[0]
	matrix = parent[1]
	nThreads = parent[2]
	if reordering in matrix:
		matrix = matrix[:matrix.index(reordering)-1]
	runtimeResults[matrix][reordering][nThreads].update({"PSTRSV3": structuredContent["PSTRSV3"], "MKL_PARALLEL": structuredContent["MKL_PARALLEL"], "BEST_SERIAL": min(structuredContent["MKL_SERIAL"], structuredContent["SPARSKIT"])})

	return structuredContent

def processRun(parent, runContent):
	global singulars, epicFails, memoryErrors
	wrongSolutionError = False
	structuredContent = {}
	tempContent = []
	startInfo = True
	startProfile = 2
	for line in runContent:
		if "* * * * * * * * * * * * * * * * * * * * * * " in line:
			if startInfo:
				startInfo = False
			else:
				structuredContent.update({"info" : processPerformanceParams(parent, tempContent)})
				tempContent = []
		elif "--------------------------------------------------------" in line:
			if startProfile > 0:
				startProfile -= 1
			else:
				structuredContent.update({"profile" : tempContent})
				tempContent = []
		else:
			if " != 0" in line:
				if "Intel MKL dcsrtrsv" in line and parent[1] not in singulars:
					singulars.append(parent[1])
				elif "Parallel Solver3" in line and parent[1] not in epicFails and parent[1] not in singulars:
					epicFails.append(parent[1])
				wrongSolutionError = True
			tempContent.append(line)
	
	# Error check
	if len(tempContent) != len(SOLVERS) or wrongSolutionError:
		if DEBUG:
			print "Missing runtime analysis for CASE: " + parent[0] + " Matrix: " + parent[1] + " where nthreads: " + parent[2]
		structuredContent = "ERROR"

		# Record failiures of different reordering algorithms
		memoryErrors[parent[0]][parent[2]] += 1 
	else:
		structuredContent.update({"profile" : processProfile(parent, structuredContent["profile"])})
		structuredContent.update({"performance" : processComparison(parent, tempContent)})

	return structuredContent

def processMatrix(parent, matrixContent):
	structuredContent = {}
	tempContent = []
	nThreads = 0
	for line in matrixContent:
		if "NTHREAD" in line:
			if tempContent:
				result = processRun(parent + [str(nThreads)], tempContent)
				if result != "ERROR":
					structuredContent.update({nThreads : result})
				tempContent = []
			nThreads = int(line.split("NTHREAD: ")[-1])
		else:
			tempContent.append(line)
	if tempContent:
		result = processRun(parent + [str(nThreads)], tempContent)
		if result != "ERROR":
			structuredContent.update({nThreads : result})
	
	# Error check
	if len(structuredContent) != len(NTHREADS):
		if DEBUG:
			print "Missing benchmark result for CASE: " + parent[0] + " Matrix: " + parent[1]
		structuredContent = "ERROR"
	
	return structuredContent

def processCase(parent, caseContent):
	structuredContent = {}
	tempContent = []
	for line in caseContent:
		if "START" in line:
			matrix = line[:-11].split("/")[-1]
		elif "END" in line:
			structuredContent.update({matrix : processMatrix([parent, matrix], tempContent)})
			tempContent = []
		else:
			tempContent.append(line)
	return structuredContent

def clearErroneousData(cases):
	erroneousMatrices = {}
	newCases = {}
	for case in cases:
		temp = {}
		for matrix in cases[case]:
			info = cases[case][matrix]
			if info == "ERROR":
				matrixname = matrix
				if case in matrixname:
					matrixname = matrixname.split("_" + case)[0]
				if matrixname not in erroneousMatrices:
					erroneousMatrices.update({matrixname: []})	
				erroneousMatrices[matrixname].append(case) 
			else:
				temp.update({matrix : info})
		newCases.update({case : temp})

	# Print erroneous matrices
	for matrix in erroneousMatrices:
		matrixname = matrix
		while len(matrixname) < 17:
			matrixname += " "
		print "ERROR: " + matrixname + " [ " + ", ".join(erroneousMatrices[matrix]) + " ]"

	return newCases

def produceReorderingEffectData(cases, reorderings, postfixes):
	isShared = True
	sharedMatrices = []
	originalCase = cases["ORIGINAL"]
	for matrix in originalCase:
		isShared = True
		for i in range(len(reorderings)):
			key = matrix + postfixes[i]
			if reorderings[i] in cases and key in cases[reorderings[i]]:
				pass
			else:
				isShared = False
		if isShared:
			sharedMatrices.append(matrix)
	
	result_PSTRSV3 = {}
	result_MKL = {}
	for matrix in sharedMatrices:
		tempCase_PSTRSV3 = {}
		tempCase_MKL = {}
		for case in reorderings + ["ORIGINAL"]:
			temp_PSTRSV3 = []
			temp_MKL = []
			key = matrix
			if case in reorderings:
				key += postfixes[reorderings.index(case)]
			for nthread in NTHREADS:
				info = cases[case][key][nthread]["performance"]

				# ------------------------------------------------------------------ #
				if DEBUG:
					if info["MKL_SERIAL"] < info["SPARSKIT"]:
						print "Best sequential solver for " + matrix + ": MKL_SERIAL" 
					else:
						print "Best sequential solver for " + matrix + ": SPARSKIT"
				# ------------------------------------------------------------------ #

				bestSequentialPerformance = info["MKL_SERIAL"] if info["MKL_SERIAL"] < info["SPARSKIT"] else info["SPARSKIT"]
				speedup_PSTRSV3 = round(bestSequentialPerformance / info["PSTRSV3"], 3)
				speedup_MKL = round(bestSequentialPerformance / info["MKL_PARALLEL"], 3)
				temp_PSTRSV3.append(speedup_PSTRSV3)
				temp_MKL.append(speedup_MKL)
			tempCase_PSTRSV3.update({case : temp_PSTRSV3})
			tempCase_MKL.update({case : temp_MKL}) 
		result_PSTRSV3.update({matrix : tempCase_PSTRSV3})
		result_MKL.update({matrix : tempCase_MKL})

	return result_PSTRSV3, result_MKL

def produceReorderingEffectChart(solverName, data, solver):
	figCount = 1
	matrixCounter = 0
	matrices = []
	dataPerChart = []
	dataPerChartList = []
	matrixCount = len(data)
	print "matrixCount: " + str(matrixCount)
	print "Building reordering effect chart for " + solver
	if VERBOSE:
		print "----------------------------------------------------------------------------"
	fig = plt.figure(figsize=(16,9))
	ax1 = fig.add_subplot(111)
	for matrix in data:
		temp = data[matrix]
		for case in temp:
			matrices.append(matrix + "_" + case)
			speedup, = ax1.plot([1] + NTHREADS, [1] + temp[case], style.next(), linewidth=2)
			dataPerChart.append(speedup)
			dataPerChartList.append([1] + temp[case])
		matrixCounter += 1
		if matrixCounter > 0 and matrixCounter % CHARTSIZE == 0 or matrixCounter == matrixCount:
			ax1.set_ylim(ymin=0, ymax=8)
			ax1.set_xlim(xmin=0, xmax=24)
			ax1.set_xlabel('nThreads', fontsize=15)
			ax1.set_ylabel('Speed-up', fontsize=15)
			ax1.tick_params(labelsize=15)
			# Adding legend
			plt.legend(dataPerChart, matrices, bbox_to_anchor=(1, 1))
			plt.title('mkl_dstrsv / ' + solver, fontsize=18)
			# Print out the information
			if VERBOSE:
				for i in range(len(matrices)):
					print matrices[i] + ": ", dataPerChartList[i]
			# Saving 
			plt.savefig("./speedup_" + solver + "_" + str(figCount) + ".png")
			figCount += 1
			matrices = []
			dataPerChart = []
			dataPerChartList = []
			fig = plt.figure(figsize=(16,9))
			ax1 = fig.add_subplot(111)
	if VERBOSE:
		print "----------------------------------------------------------------------------"

def producePreprocessingComparisonChart(perfParams):
	# caseStudies list is produced by producePerformanceComparisonChart() function
	global caseStudies, runtimeResults, NTHREADS
	opacity = 0.8
	bar_width = 0.35
	style_PSTRSV = itertools.cycle(["-v"])
	style_MKL = itertools.cycle(["-s"])
	print "Building preprocessing comparison chart"
	for matrix in caseStudies:
		pos = 1
		fig = plt.figure(figsize=(9,7))
		reorderingCount = len(perfParams[matrix])
		col = reorderingCount / 3
		for t in NTHREADS:
			# Produce result lists
			selectedThreadCount = str(t)
			temp_reorderings = []
			temp_PSTRSV = []
			temp_MKL = []
			for reordering in perfParams[matrix]:
				preprocessPSTRSV = runtimeResults[matrix][reordering][selectedThreadCount]["preprocessPSTRSV"]
				preprocessMKL = runtimeResults[matrix][reordering][selectedThreadCount]["preprocessMKL"]
				sizeMatrixS = perfParams[matrix][reordering][t]
				temp_PSTRSV.append(preprocessPSTRSV)
				temp_MKL.append(preprocessMKL)
				temp_reorderings.append(reordering)
		
			# Start producing the charts
			ax1 = fig.add_subplot(3, col, pos)

			# Add data
			y_pos = np.arange(6)
			ax1.bar(y_pos, temp_PSTRSV, bar_width, alpha=opacity, color=tableau20[6], label='PSTRSV', edgecolor=tableau20[6], hatch="")
			ax1.bar(y_pos + bar_width, temp_MKL, bar_width, alpha=opacity, color=tableau20[1], label='MKL', edgecolor=tableau20[0], hatch="//")

			# Prepare the subplot
			ax1.set_yscale('log', basey=10)
			ax1.set_ylabel('Elapsed time (ms)', fontsize=10)
			ax1.grid(linestyle=':', linewidth='1')

			# Adding legend
			plt.legend(loc=9, prop={'size': 11})
			plt.xticks(y_pos + bar_width, temp_reorderings)
			plt.title("t = " + selectedThreadCount, fontsize=12)
			plt.tight_layout()
			plt.grid(True)

			pos += 1

		# Saving 
		plt.savefig("./" + matrix + "_preprocessing.png")

def producePerformanceComparisonChart(data_PSTRSV3, data_MKL):
	global caseStudies, gPerfParams
	caseStudies = []
	style_PSTRSV3 = itertools.cycle(["-v"])
	style_MKL = itertools.cycle(["-s"])
	print "matrixCount:", len(data_PSTRSV3)
	print "Building performance comparison charts"
	if ONE_CHART_FOR_ALL:
		for matrix in data_PSTRSV3:
			pos = 1
			fig = plt.figure(figsize=(9,5))
			temp_PSTRSV3 = data_PSTRSV3[matrix]
			temp_MKL = data_MKL[matrix]
			reorderingCount = len(temp_PSTRSV3)
			col = reorderingCount / 2
			caseStudies.append(matrix)
			matrices = []
			dataPerChart = []
			for case in temp_PSTRSV3:
				ax1 = fig.add_subplot(2, col, pos)

				# Add data
				matrices.append("PSTRSV")
				speedup, = ax1.plot([1] + NTHREADS, [1] + temp_PSTRSV3[case], style_PSTRSV3.next(), color="crimson", linewidth=2)
				dataPerChart.append(speedup)
				matrices.append("MKL")
				speedup, = ax1.plot([1] + NTHREADS, [1] + temp_MKL[case], style_MKL.next(), color="#005f87", linewidth=2)
				dataPerChart.append(speedup)

				# Prepare the subplot
				ax1.set_ylim(ymin=0, ymax=6)
				ax1.set_xlim(xmin=0, xmax=21)
				ax1.set_xlabel('nThreads', fontsize=10)
				ax1.set_ylabel('Speed-up', fontsize=10)
				ax1.grid(linestyle=':', linewidth='1')
				#ax1.tick_params(labelsize=15)

				# Adding legend
				plt.legend(dataPerChart, matrices, bbox_to_anchor=(0.52, 1), fontsize=11)
				plt.title(case, fontsize=12)
				plt.tight_layout()
				plt.grid(True)

				matrices = []
				dataPerChart = []
				pos += 1	

			# Saving 
			plt.savefig("./" + matrix + ".png")
	else:
		for matrix in data_PSTRSV3:
			temp_PSTRSV3 = data_PSTRSV3[matrix]
			temp_MKL = data_MKL[matrix]
			reorderingCount = len(temp_PSTRSV3)
			caseStudies.append(matrix)
			for case in temp_PSTRSV3:
				matrices = []
				dataPerChart = []
				fig = plt.figure(figsize=(7,6))
				ax1 = fig.add_subplot(1, 1, 1)

				# Add data
				matrices.append("PSTRSV")
				speedup, = ax1.plot([1] + NTHREADS, [1] + temp_PSTRSV3[case], style_PSTRSV3.next(), color="crimson", linewidth=2)
				dataPerChart.append(speedup)
				matrices.append("MKL")
				speedup, = ax1.plot([1] + NTHREADS, [1] + temp_MKL[case], style_MKL.next(), color="#005f87", linewidth=2)
				dataPerChart.append(speedup)

				# Prepare the subplot
				ax1.set_ylim(ymin=0, ymax=6)
				ax1.set_xlim(xmin=0, xmax=21)
				ax1.set_xlabel('nThreads', fontsize=13)
				ax1.set_ylabel('Speed-up', fontsize=13)
				ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
				ax1.grid(linestyle=':', linewidth='1')
				#ax1.tick_params(labelsize=15)

				# Adding legend
				plt.legend(dataPerChart, matrices, loc=1, fontsize=13)
				plt.title("%s for %s Reordering" % (matrix, case), fontsize=15)
				plt.tight_layout()
				plt.grid(True)

				# Saving 
				plt.savefig("./" + matrix + "_" + case + ".png")

	# Produce preprocessing curves as well
	producePreprocessingComparisonChart(gPerfParams)

def produceMemErrorComparisonChart():
	global memoryErrors, tableau20
	print "Building memory error comparison charts"
	plt.figure(figsize=(9, 4))

	# Chart data
	reorderings = memoryErrors.keys()
	cases = {r: [] for r in reorderings}
	for r in cases:
		for n in NTHREADS:
			cases[r].append(memoryErrors[r][str(n)])
	y_pos = np.arange(len(cases))

	# Chart configuration
	bar_width = 0.11
	opacity = 0.8

	rects1 = plt.bar(y_pos, cases["METIS"], bar_width, alpha=opacity, color=tableau20[6], label='METIS', edgecolor=tableau20[6], hatch="")
	rects2 = plt.bar(y_pos + bar_width, cases["NDP"], bar_width, alpha=opacity, color=tableau20[1], label='NDP', edgecolor=tableau20[0], hatch="//")
	rects3 = plt.bar(y_pos + 2*bar_width, cases["ORIGINAL"], bar_width, alpha=opacity, color=tableau20[5], label='ORIGINAL', edgecolor=tableau20[4], hatch="--")
	rects4 = plt.bar(y_pos + 3*bar_width, cases["RCM"], bar_width, alpha=opacity, color="#FFBE7D", label='RCM', edgecolor=tableau20[7], hatch="++")
	rects5 = plt.bar(y_pos + 4*bar_width, cases["AMD"], bar_width, alpha=opacity, color="#A0CBE8", label='AMD', edgecolor=tableau20[1], hatch="..")
	rects6 = plt.bar(y_pos + 5*bar_width, cases["ColPerm"], bar_width, alpha=opacity, color="#BAB0AC", label='ColPerm', edgecolor=(0, 0, 0), hatch="\\")

	plt.xticks(y_pos + 3*bar_width, [str(n) for n in NTHREADS])
	plt.xlabel('# of threads')
	plt.ylabel('# of cases')
	plt.legend(loc=2, fontsize=11)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("./memErrComparison.png")

def getIterCountForAmortization(prepTime, parSolveTime, seqSolveTime):
	return float(prepTime) / (seqSolveTime - parSolveTime) if seqSolveTime != parSolveTime else prepTime / 1e-5

def reportAllSpeedups():
	global speedupResults, matrixList
	reorderingList = ["RCM", "ColPerm", "NDP", "METIS", "AMD", "ORIGINAL"]

	print
	print "====================================| Speedup Results |===================================="
	matrixList.sort()
	for matrix in matrixList:
		if matrix in speedupResults:
			matrixname = matrix.replace("_", "\\_")
			print "\\subsection{%s}" % matrixname
			print "\\begin{table}[H]"
			print "\\begin{center}"
			print "\\begin{tabular}{l|rrrrrr}"
			print "\\toprule"
			print "\\multirow{2}{*}{t} & \\multicolumn{6}{c}{PSTRSV}"
			print "\\\\\\cmidrule{2-7}"
			print "& RCM & ColPerm & NDP & METIS & AMD & ORIG \\\\"
			print "\\midrule"
			for nthread in NTHREADS:
				t = str(nthread)
				line = "$%s$ &" % t
				temp = []
				maxSpeedup = -1
				bestReordering = None
				for r in reorderingList:
					if r in speedupResults[matrix] and t in speedupResults[matrix][r] and "PSTRSV3" in speedupResults[matrix][r][t]:
						speedup = speedupResults[matrix][r][t]["PSTRSV3"]
						if maxSpeedup < speedup:
							maxSpeedup = speedup
							bestReordering = r
						speedup = "{0:.2f}".format(speedup)
						temp.append("$" + speedup + "$")
					else:
						temp.append("-")
				if bestReordering != None:
					for i in range(len(temp)):
						if i == reorderingList.index(bestReordering):
							line += " $\\textbf{" + temp[i][1:-1] + "}$ &"
						else:
							line += " " + temp[i] + " &"
					line = line[:-1] + " \\\\"
				else:
					line += " - & - & - & - & - & - \\\\"
				print line
			print "\\bottomrule"
			print "\\end{tabular}"
			print "\\end{center}"
			print "\\caption{Speedup results of PSTRSV using different reoderings for \\textit{%s}}" % matrixname
			print "\\label{tab:speedupPSTRSV_" + matrix + "}"
			print "\\end{table}"

			print 
			print "\\begin{table}[H]"
			print "\\begin{center}"
			print "\\begin{tabular}{l|rrrrrr}"
			print "\\toprule"
			print "\\multirow{2}{*}{t} & \\multicolumn{6}{c}{MKL}"
			print "\\\\\\cmidrule{2-7}"
			print "& RCM & ColPerm & NDP & METIS & AMD & ORIG \\\\"
			print "\\midrule"
			for nthread in NTHREADS:
				t = str(nthread)
				line = "$%s$ &" % t
				temp = []
				maxSpeedup = -1
				bestReordering = None
				for r in reorderingList:
					if r in speedupResults[matrix] and t in speedupResults[matrix][r] and "MKL_PARALLEL" in speedupResults[matrix][r][t]:
						speedup = speedupResults[matrix][r][t]["MKL_PARALLEL"]
						if maxSpeedup < speedup:
							maxSpeedup = speedup
							bestReordering = r
						speedup = "{0:.2f}".format(speedup)
						temp.append("$" + speedup + "$")
					else:
						temp.append("-")
				if bestReordering != None:
					for i in range(len(temp)):
						if i == reorderingList.index(bestReordering):
							line += " $\\textbf{" + temp[i][1:-1] + "}$ &"
						else:
							line += " " + temp[i] + " &"
					line = line[:-1] + " \\\\"
				else:
					line += " - & - & - & - & - & - \\\\"
				print line
			print "\\bottomrule"
			print "\\end{tabular}"
			print "\\end{center}"
			print "\\caption{Speedup results of MKL using different reoderings for \\textit{%s}}" % matrixname
			print "\\label{tab:speedupMKL_" + matrix + "}"
			print "\\end{table}"
			print

	print "==========================================================================================="
	print

def reportAllRuntimes():
	global speedupResults, runtimeResults, matrixList
	speedupResults = runtimeResults
	amortizationList = []
	preprocessingList = []
	mklNonamortizableCases = {}
	pstrsvNonamortizableCases = {}
	mklNonamortizableCounter = 0
	pstrsvNonamortizableCounter = 0
	bestPossibleSpeedUpsPSTRSV = {}
	bestPossibleSpeedUpsMKL = {}

	for t in NTHREADS:
		selectedThreadCount = str(t)
		counter = 0
		mklCounter = 0
		strsvCounter = 0
		pstrsvCounter = 0
		matrixList = []
		mklAmortization = []
		mklPreprocessing = []
		pstrsvAmortization = []
		pstrsvPreprocessing = []

		print
		print "====================================| Runtime Results |===================================="
		print "\\subsection{t = %s}" % t
		print "\\begin{center}"
		print "\\begin{longtable}{l|rr|rr|r}"
		print "\\toprule"
		print "\\multirow{2}{*}{Matrix} & \\multicolumn{2}{c|}{PSTRSV} & \\multicolumn{2}{c|}{MKL} &"
		print "\\multirow{2}{*}{"
		print "\\begin{tabular}{r}STRSV\\end{tabular}"
		print "} \\\\\\cmidrule{2-5}"
		print "		& Prep. & Sol. & Prep. & Sol. \\\\"
		print "\\midrule"
		for matrix in runtimeResults:
			matrixname = matrix.replace("_","\\_")
			matrixList.append(matrix)
			for reordering in runtimeResults[matrix]: 
				name = matrixname + "\\_" + reordering
				while len(name) < 26:
					name += " "
				if selectedThreadCount in runtimeResults[matrix][reordering]:
					preprocessPSTRSV = "-"
					preprocessMKL = "-"
					runtimePSTRSV = "-"
					runtimeMKL = "-"
					runtimeSTRSV = "{0:.2f}".format(runtimeResults[matrix][reordering][selectedThreadCount]["BEST_SERIAL"])

					if "PSTRSV3" in runtimeResults[matrix][reordering][selectedThreadCount]:
						preprocessPSTRSV = "{0:.2f}".format(runtimeResults[matrix][reordering][selectedThreadCount]["preprocessPSTRSV"])
						runtimePSTRSV = "{0:.2f}".format(runtimeResults[matrix][reordering][selectedThreadCount]["PSTRSV3"])
						speedupPSTRSV = float(runtimeSTRSV) / float(runtimePSTRSV)	
						speedupResults[matrix][reordering][selectedThreadCount]["PSTRSV3"] = speedupPSTRSV
						requiredIterPSTRSV = getIterCountForAmortization(float(preprocessPSTRSV), float(runtimePSTRSV), float(runtimeSTRSV))
						if matrix in bestPossibleSpeedUpsPSTRSV:
							if speedupPSTRSV > bestPossibleSpeedUpsPSTRSV[matrix][1]:
								bestPossibleSpeedUpsPSTRSV[matrix] = [reordering, speedupPSTRSV, selectedThreadCount, requiredIterPSTRSV]
						else:
							bestPossibleSpeedUpsPSTRSV.update({matrix: [reordering, speedupPSTRSV, selectedThreadCount, requiredIterPSTRSV]})
						if float(runtimeSTRSV) <= float(runtimePSTRSV):
							if name in pstrsvNonamortizableCases:
								pstrsvNonamortizableCases[name] += 1
							else:
								pstrsvNonamortizableCases.update({name: 1})
						else:
							pstrsvAmortization.append(requiredIterPSTRSV)
									
					if "MKL_PARALLEL" in runtimeResults[matrix][reordering][selectedThreadCount]:
						preprocessMKL = "{0:.2f}".format(runtimeResults[matrix][reordering][selectedThreadCount]["preprocessMKL"])
						runtimeMKL = "{0:.2f}".format(runtimeResults[matrix][reordering][selectedThreadCount]["MKL_PARALLEL"])
						speedupMKL = float(runtimeSTRSV) / float(runtimeMKL)	
						speedupResults[matrix][reordering][selectedThreadCount]["MKL_PARALLEL"] = speedupMKL
						requiredIterMKL = getIterCountForAmortization(float(preprocessMKL), float(runtimeMKL), float(runtimeSTRSV))
						if matrix in bestPossibleSpeedUpsMKL:
							if speedupMKL > bestPossibleSpeedUpsMKL[matrix][1]:
								bestPossibleSpeedUpsMKL[matrix] = [reordering, speedupMKL, selectedThreadCount, requiredIterMKL]
						else:
							bestPossibleSpeedUpsMKL.update({matrix: [reordering, speedupMKL, selectedThreadCount, requiredIterMKL]})
						if float(runtimeSTRSV) <= float(runtimeMKL):
							if name in mklNonamortizableCases:
								mklNonamortizableCases[name] += 1
							else:
								mklNonamortizableCases.update({name: 1})
						else:
							mklAmortization.append(requiredIterMKL)
						
					# Store preprocessing times of each solver whether s > 1 or not
					pstrsvPreprocessing.append(float(preprocessPSTRSV))
					mklPreprocessing.append(float(preprocessMKL))

					if float(runtimePSTRSV) > float(runtimeSTRSV) and float(runtimeMKL) > float(runtimeSTRSV):
						runtimeSTRSV = "$\\textbf{" + runtimeSTRSV + "}$"
						runtimePSTRSV = "$" + runtimePSTRSV + "$"
						runtimeMKL = "$" + runtimeMKL + "$"
						strsvCounter += 1
					else:
						runtimeSTRSV = "$" + runtimeSTRSV + "$"
						if float(runtimePSTRSV) <= float(runtimeMKL):
							runtimePSTRSV = "$\\textbf{" + runtimePSTRSV + "}$"
							runtimeMKL = "$" + runtimeMKL + "$"
							pstrsvCounter += 1
						else:
							runtimePSTRSV = "$" + runtimePSTRSV + "$"
							runtimeMKL = "$\\textbf{" + runtimeMKL + "}$"
							mklCounter += 1
					counter += 1

					print name, "& $" + preprocessPSTRSV + "$ &", runtimePSTRSV, "\t& $" + preprocessMKL + "$ &", runtimeMKL, "\t&", runtimeSTRSV, "\\\\"
			print "\\midrule"	
		print "\\bottomrule"
		print "\\caption{The elapsed times of preprocessing and solution parts of the proposed algorithm and Intel MKL against the best sequential algorithm for different matrix reorderings. Measured in milliseconds. The number of threads is $" + selectedThreadCount + "$ for parallel solvers.}"
		print "\\label{tab:runtimeResults1_" + selectedThreadCount + "}"
		print "\\end{longtable}"
		print "\\end{center}"
		print "==========================================================================================="
		print
		print "pSTRSV: " + str(pstrsvCounter) + "/" + str(counter),
		print "MKL: " + str(mklCounter) + "/" + str(counter),
		print "STRSV: " + str(strsvCounter) + "/" + str(counter) 

		# ----------------------------------------------------------------------------- #
		# Process amortization statistics
		minAmortization = 2000000
		maxAmortization = 0
		avgAmortization = 0
		stdAmortization = 0
		if len(pstrsvAmortization) > 0:
			for i in pstrsvAmortization:
				minAmortization = min(i, minAmortization)
				maxAmortization = max(i, maxAmortization)
				avgAmortization += i
			avgAmortization /= len(pstrsvAmortization)
			for i in pstrsvAmortization:
				stdAmortization += (i - avgAmortization)**2
			stdAmortization /= len(pstrsvAmortization)
			stdAmortization = stdAmortization**0.5
			minAmortization = int(math.ceil(minAmortization))
			maxAmortization = int(math.ceil(maxAmortization))
		amortizationStrPSTRSV = "$" + selectedThreadCount + "$ & $" + str(minAmortization) + "$ & $" + str(maxAmortization) + "$ & $" + "%.2f" % avgAmortization + "$ & $" + "%.2f" % stdAmortization + "$"
		
		minAmortization = 2000000
		maxAmortization = 0
		avgAmortization = 0
		stdAmortization = 0
		if len(mklAmortization) > 0:
			for i in mklAmortization:
				minAmortization = min(i, minAmortization)
				maxAmortization = max(i, maxAmortization)
				avgAmortization += i
			avgAmortization /= len(mklAmortization)
			for i in mklAmortization:
				stdAmortization += (i - avgAmortization)**2
			stdAmortization /= len(mklAmortization)
			stdAmortization = stdAmortization**0.5
			minAmortization = int(math.ceil(minAmortization))
			maxAmortization = int(math.ceil(maxAmortization))
		amortizationStrMKL = " & $" + str(minAmortization) + "$ & $" + str(maxAmortization) + "$ & $" + "%.2f" % avgAmortization + "$ & $" + "%.2f" % stdAmortization + "$ \\\\"
		
		amortizationList.append(amortizationStrPSTRSV + amortizationStrMKL)
		# ----------------------------------------------------------------------------- #

		# ----------------------------------------------------------------------------- #
		# Process preprocessing statistics
		minPreproc = 2000000
		maxPreproc = 0
		avgPreproc = 0
		stdPreproc = 0
		for i in pstrsvPreprocessing:
			minPreproc = min(i, minPreproc)
			maxPreproc = max(i, maxPreproc)
			avgPreproc += i
		avgPreproc /= len(pstrsvPreprocessing)
		for i in pstrsvPreprocessing:
			stdPreproc += (i - avgPreproc)**2
		stdPreproc /= len(pstrsvPreprocessing)
		avgPreproc = "%.2f" % avgPreproc
		stdPreproc = "%.2f" % stdPreproc**0.5
		minPreproc = "%.2f" % minPreproc
		maxPreproc = "%.2f" % maxPreproc
		preprocStrPSTRSV = "$" + selectedThreadCount + "$ & $" + minPreproc + "$ & $" + maxPreproc + "$ & $" + avgPreproc + "$ & $" + stdPreproc + "$"
		
		minPreproc = 2000000
		maxPreproc = 0
		avgPreproc = 0
		stdPreproc = 0
		for i in mklPreprocessing:
			minPreproc = min(i, minPreproc)
			maxPreproc = max(i, maxPreproc)
			avgPreproc += i
		avgPreproc /= len(mklPreprocessing)
		for i in mklPreprocessing:
			stdPreproc += (i - avgPreproc)**2
		stdPreproc /= len(mklPreprocessing)
		avgPreproc = "%.2f" % avgPreproc
		stdPreproc = "%.2f" % stdPreproc**0.5
		minPreproc = "%.2f" % minPreproc
		maxPreproc = "%.2f" % maxPreproc
		preprocStrMKL = " & $" + minPreproc + "$ & $" + maxPreproc + "$ & $" + avgPreproc + "$ & $" + stdPreproc + "$ \\\\"
		
		preprocessingList.append(preprocStrPSTRSV + preprocStrMKL)
		# ----------------------------------------------------------------------------- #

	# ----------------------------------------------------------------------------- #
	reportAllSpeedups()
	# ----------------------------------------------------------------------------- #

	# ----------------------------------------------------------------------------- #
	print
	print "Best possible speedups & the required iter count for amortization:"
	print "================================================================="
	matrixList.sort()
	for matrix in matrixList:
		if matrix in bestPossibleSpeedUpsPSTRSV and matrix in bestPossibleSpeedUpsMKL:
			requiredIterPSTRSV = "%.1f" % bestPossibleSpeedUpsPSTRSV[matrix][3] if bestPossibleSpeedUpsPSTRSV[matrix][3] > 0 else "FAIL"
			speedupPSTRSV = "%.2f" % bestPossibleSpeedUpsPSTRSV[matrix][1]
			reorderingPSTRSV = bestPossibleSpeedUpsPSTRSV[matrix][0] + " " * (8 - len(bestPossibleSpeedUpsPSTRSV[matrix][0]))
			threadsPSTRSV = "- t = " + bestPossibleSpeedUpsPSTRSV[matrix][2] + " " * (2 - len(bestPossibleSpeedUpsPSTRSV[matrix][2]))
			requiredIterPSTRSV += " " * (5 - len(requiredIterPSTRSV))

			requiredIterMKL = "%.1f" % bestPossibleSpeedUpsMKL[matrix][3] if bestPossibleSpeedUpsMKL[matrix][3] > 0 else "FAIL"
			speedupMKL = "%.2f" % bestPossibleSpeedUpsMKL[matrix][1]
			reorderingMKL = bestPossibleSpeedUpsMKL[matrix][0] + " " * (8 - len(bestPossibleSpeedUpsMKL[matrix][0]))
			threadsMKL = "- t = " + bestPossibleSpeedUpsMKL[matrix][2] + " " * (2 - len(bestPossibleSpeedUpsMKL[matrix][2]))
			name = matrix + " " * (15 - len(matrix))

			print name, " & PSTRSV:", speedupPSTRSV, "-", reorderingPSTRSV, threadsPSTRSV, "- i =", requiredIterPSTRSV, " & MKL:", speedupMKL, "-", reorderingMKL, threadsMKL, "- i =", requiredIterMKL
	# ----------------------------------------------------------------------------- #

	# ----------------------------------------------------------------------------- #
	print
	print "Amortization table:"
	print "=================="
	print "\\begin{table}"
	print "\\begin{center}"
	print "\\begin{tabular}{l|rrrr|rrrr}"
	print "\\toprule"
	print "\\multirow{2}{*}{t} & \\multicolumn{4}{c|}{PSTRSV} & \\multicolumn{4}{c}{MKL}"
	print "\\\\\\cmidrule{2-9}"
	print "& min & max & avg & std & min & max & avg & std \\\\"
	print "\\midrule"
	for line in amortizationList:
		print line
	print "\\bottomrule"
	print "\\end{tabular}"
	print "\\end{center}"
	print "\\caption{Statistics regarding the required iteration count for amortization.}"
	print "\\label{tab:amortizationStat}"
	print "\\end{table}"
	# ----------------------------------------------------------------------------- #

	# ----------------------------------------------------------------------------- #
	print
	print "Preprocessing table:"
	print "==================="
	print "\\begin{table}"
	print "\\begin{center}"
	print "\\begin{tabular}{l|rrrr|rrrr}"
	print "\\toprule"
	print "\\multirow{2}{*}{t} & \\multicolumn{4}{c|}{PSTRSV} & \\multicolumn{4}{c}{MKL}"
	print "\\\\\\cmidrule{2-9}"
	print "& min & max & avg & std & min & max & avg & std \\\\"
	print "\\midrule"
	for line in preprocessingList:
		print line
	print "\\bottomrule"
	print "\\end{tabular}"
	print "\\end{center}"
	print "\\caption{Statistics regarding the preprocessing times of PSTRSV and MKL.}"
	print "\\label{tab:preprocStat}"
	print "\\end{table}"
	# ----------------------------------------------------------------------------- #

	# ----------------------------------------------------------------------------- #
	print 
	print "Impossible to amortize:"
	print "======================"
	for i in pstrsvNonamortizableCases:
		if pstrsvNonamortizableCases[i] == len(NTHREADS):
			pstrsvNonamortizableCounter += 1
			print "PSTRSV cannot amortize", i
	for i in mklNonamortizableCases:
		if mklNonamortizableCases[i] == len(NTHREADS):
			mklNonamortizableCounter += 1
			print "MKL cannot amortize", i
	print "In total pSTRSV:", pstrsvNonamortizableCounter, "MKL:", mklNonamortizableCounter
	# ----------------------------------------------------------------------------- #

	if len(matrixList) > 0:
		print
		print "Matrices in the paper:"
		print "======================"
		for matrix in matrixList:
			print matrix
		print

if __name__ == '__main__':
	# Configuration parameters
	# Example: plotSpeedUp.py --log_file <log_file> --use_avg --case_study_mode --one_chart_for_all --memory_error_check_mode
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--log_file", help="File path to the spike_pstrsv log file", type=str, required=True)
	parser.add_argument("--debug", help="Enables debug mode", action="store_true")
	parser.add_argument("--verbose", help="Enables detailed logs", action="store_true")
	parser.add_argument("--use_avg", help="If set True, uses average runtime measurement. Else, uses best runtime measurement.", action="store_true")
	parser.add_argument("--chart_size", help="Max. # of data types in a chart", type=int)
	parser.add_argument("--n_threads", help="List of selected thread counts for the runtime analysis", nargs="+")
	parser.add_argument("--solvers", help="List of selected sparse triangular systems solvers for the runtime analysis", nargs="+")
	parser.add_argument("--reorderings", help="List of selected matrix reordering algorithms for the runtime analysis", nargs="+")
	parser.add_argument("--postfixes", help="List of postfixes for the given matrix reorderings. (List size MUST match with --reorderings option)", nargs="+")
	parser.add_argument("--overview_mode", help="Enable debug mode", action="store_true")
	parser.add_argument("--case_study_mode", help="Enables chart creation per case where all reoderings & thread counts have valid measurements.", action="store_true")
	parser.add_argument("--one_chart_for_all", help="Enables creation of line charts summarizing the runtime measurements.", action="store_true")
	parser.add_argument("--full_appendix_mode", help="Enables logging all runtime & speedup measurements as LaTeX tables.", action="store_true")
	parser.add_argument("--memory_error_check_mode", help="Enables chart creation for memory failiures per each given reodering algorithm.", action="store_true")
	args = vars(parser.parse_args())

	# Dynamic adjustments to reflect user configurations
	pSTRSVlog 				= args["log_file"]
	DEBUG 					= args["debug"] if "debug" in args else DEBUG
	VERBOSE 				= args["verbose"] if "verbose" in args else VERBOSE
	USEAVG 					= args["use_avg"] if "use_avg" in args else USEAVG
	CHARTSIZE 				= args["chart_size"] if "chart_size" in args else CHARTSIZE
	NTHREADS 				= [int(i) for i in args["n_threads"]] if args["n_threads"] is not None else NTHREADS
	SOLVERS 				= args["solvers"] if args["solvers"] is not None else SOLVERS
	REORDERINGS 			= args["reorderings"] if args["reorderings"] is not None else REORDERINGS
	POSTFIXES 				= args["postfixes"] if args["postfixes"] is not None else POSTFIXES
	OVERVIEW_MODE 			= args["overview_mode"] if "overview_mode" in args else OVERVIEW_MODE
	CASE_STUDY_MODE 		= args["case_study_mode"] if "case_study_mode" in args else CASE_STUDY_MODE
	ONE_CHART_FOR_ALL 		= args["one_chart_for_all"] if "one_chart_for_all" in args else ONE_CHART_FOR_ALL
	FULL_APPENDIX_MODE 		= args["full_appendix_mode"] if "full_appendix_mode" in args else FULL_APPENDIX_MODE
	MEMORY_ERROR_CHECK_MODE = args["memory_error_check_mode"] if "memory_error_check_mode" in args else MEMORY_ERROR_CHECK_MODE

	# Global parameters
	gPerfParams 	= {}
	cases 			= {}
	content			= []
	singulars		= []
	epicFails		= []
	matrixList		= []
	caseStudies 	= []
	runtimeResults	= {}
	speedupResults  = {}
	memoryErrors 	= {r: {str(n): 0 for n in NTHREADS} for r in REORDERINGS + ["ORIGINAL"]}
	style 			= itertools.cycle(["--v","--^","-s","-p","--o","-.s","-^","-o","-.p","-v"])

	#----------------------------------------------------------#

	with open(pSTRSVlog, 'r') as logfile:
		for line in logfile:
			if "*** CASE-" in line:
				if content:
					cases.update({mode : content})
					content = []
				mode = line.split(": ")[1].split(" ")[0]
			elif "BENCHMARK END" in line:
				if content:
					cases.update({mode : content})
			elif line != "\n":
				content.append(line[:-1])

	#----------------------------------------------------------#

	for case in cases:
		cases.update({case : processCase(case, cases[case])})

	#----------------------------------------------------------#

	for i in range(len(tableau20)):
		r, g, b = tableau20[i]
		tableau20[i] = (r / 255., g / 255., b / 255.)

	#----------------------------------------------------------#

	cases = clearErroneousData(cases)
	data_PSTRSV3, data_MKL = produceReorderingEffectData(cases, REORDERINGS, POSTFIXES)

	if OVERVIEW_MODE:
		produceReorderingEffectChart("PSTRSV3", data_PSTRSV3, SOLVERS[3])
		produceReorderingEffectChart("MKL", data_MKL, SOLVERS[1])
	if CASE_STUDY_MODE:
		producePerformanceComparisonChart(data_PSTRSV3, data_MKL)
	if MEMORY_ERROR_CHECK_MODE:
		produceMemErrorComparisonChart()
	if FULL_APPENDIX_MODE:
		reportAllRuntimes()

	errorLog()

	print "Done."

	#----------------------------------------------------------#