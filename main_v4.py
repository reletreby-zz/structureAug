from __future__ import division
import argparse, sys, pdb, site, collections, itertools, datetime, random, copy, sklearn, csv
import networkx as nx
from utils import Clusters
import numpy as np
from dimReducUtils import PCAm
from scipy.spatial import distance
import matplotlib.pyplot as plt


# List of egonodes: [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]

def createTestMat(width, length, std=10):
	x = np.array([0]*width)

	for ii in range(length):
		tmp = [np.random.normal(scale=std*random.random()) for jj in range(width)]
		x = np.vstack((x, tmp))

	return x[1:]


def getMatrix(nodeFeaturesDict):
	outMat = np.array([nodeFeaturesDict[key] for key in nodeFeaturesDict.keys()])
	outIDs = np.array(list(nodeFeaturesDict.keys()))
	error = 0
	for ii in range(len(outIDs)):
		error += sum(outMat[ii] - nodeFeaturesDict[outIDs[ii]])
	if error>0:
		raise ValueError('Something Wrong Here')
	return outMat, outIDs


def getGroundTruthCircles(egoCircles):
	outDict = collections.defaultdict(list)
	for line in open(egoCircles, mode='r'):
		line = line.strip().split()
		outDict[line[0]] = [int(item) for item in line[1:]]
	return outDict

def readFeatures(egoFeatures, scale=0):
	nodeFeatures = collections.defaultdict(list)
	for line in open(egoFeatures, mode='r'):
		line = line.strip().split()
		if len(line)<=2:
			continue
		currFeatVals = [float(item) for item in line[1:]]
		if scale==0:
			nodeFeatures[int(line[0])] = currFeatVals
		else:
			currFeatValsScaled = [2*item-1 for item in currFeatVals]
			nodeFeatures[int(line[0])] = currFeatValsScaled
	return nodeFeatures

def readFeatureNames(egoFeaturesNames):
	featureNames = []
	for line in open(egoFeaturesNames, mode='r'):
		line = line.strip().split()
		featureNames.append(line[1])
	return featureNames

def parse_args(args):
	parser = argparse.ArgumentParser(description = 'Parameters')
	parser.add_argument('-egoNode', type = int, default = 0, help='The index of the egoNode')
	parser.add_argument('-scale', type = int, default = 0, help='Whether or not to scale Features/Structure')
	parser.add_argument('-numRuns', type = int, default = 1, help='Number of runs')
	parser.add_argument('-numPCA', type = int, default = 20, help='Number of PCA Components')
	parser.add_argument('-overlappingFlag', type = int, default = 0, help='Whether Overlapping clusters are allowed')
	parser.add_argument('-alphaVal', type = float, default = 0.2, help='Edge addition probability')
	parser.add_argument('-gmmCutOff', type = float, default = 0.8, help='Prob. Cutoff for GMM')
	parser.add_argument('-weight1', type = float, default = 0.2, help='Edge Weight = 1+weight1*avgHamming')
	parser.add_argument('-weight0', type = float, default = 0.2, help='Edge Weight = 0+weight0*avgHamming')
	parser.add_argument('-structAug', type = int, default = 0, help='Whether to augment structure with clustered features or not')
	parser.add_argument('-subset', type = int, default = 0, help='Whether or not to only consider a subset of nodes which actually apppeared in the ground truth circles')
	parser.add_argument('-pcaEnabled', type = int, default = 1, help='Whether or not to use PCA when augmenting with feature clusters')
	parser.add_argument('-specClustMethod', default='gmm', help = 'Clustering Method for graph structure')
	parser.add_argument('-csvFilePath', default='./out2/outCSV', help = 'Path to save CSV file')
	parser.add_argument('-featClustMethod', default='gmm', help = 'Clustering Method for features')
	parser.add_argument('-maxNumComponents', default=50, type=int, help = 'Max number of components to try if we are to optimize')
	parser.add_argument('-numClusters', default=20, type=int, help = 'Number of clusters')
	parser.add_argument('-covarType', default='full', help = 'Covariance Type for GMM')
	parser.add_argument('-getBestNumComp', default=0, type=int, help = 'Whether to optimize the number of clusters for GMM')
	parser.add_argument('-costType', default='BER', help = 'BER or fScore')
	return parser.parse_args(args)

def countNodeOverlap(groundTruthCircles, nodeList):
	countDict = collections.defaultdict(int)
	for node in nodeList:
		count = 0
		for circle in groundTruthCircles:
			if node in groundTruthCircles[circle]:
				count += 1
		countDict[node] = count
	return countDict

def cleanTuples(allPossibleTuples):
	retList = []
	for element in allPossibleTuples:
		if element[0]==element[1]:
			continue
		retList.append(element)
	return retList

def getCircleStats(G, groundTruthCircles):
	edgeRatioDict = collections.defaultdict()
	for circle in groundTruthCircles:
		circleMembers = groundTruthCircles[circle]
		subGraphMembers = [str(item) for item in circleMembers]
		n = len(subGraphMembers)
		Gsub = G.subgraph(subGraphMembers)
		totNumPossibleEdges = n*(n-1)/2
		if len(Gsub.edges())>0:
			edgeRatioDict[circle] = (n, len(Gsub.edges())/totNumPossibleEdges, nx.average_clustering(Gsub))
		else:
			# It is a circle of single component
			edgeRatioDict[circle] = (n, -1, -1)
	return edgeRatioDict


def writeLog(circleStat, logFile, egoNode, numOverlap, percOverlap, numIsolated, percIsolated, numNodes, groundTruthCircles):
	with open(logFile, 'a+') as logFile:
		logFile.write('Time: {0}\n'.format(str(datetime.datetime.today())))
		logFile.write('egoNode: {0}\n'.format(egoNode))
		logFile.write('numNodes: {0}\n'.format(numNodes))
		logFile.write('numOverlap: {0} PercentageOverlap: {1}\n'.format(numOverlap, percOverlap*100))
		logFile.write('numIsolated: {0} PercentageIsolated: {1}\n'.format(numIsolated , percIsolated*100))
		logFile.write('------------\n')
		for key in circleStat.keys():
			logFile.write('{0} numMembers: {1} edgeRatio: {2} clustering: {3}\n'.\
			format(key, circleStat[key][0], circleStat[key][1], circleStat[key][2]))

		logFile.write('\n')
		logFile.write('------------\n')
		logFile.write('\n')
		allMembers = []
		for key in groundTruthCircles.keys():
			circleMembers = groundTruthCircles[key]
			allMembers.extend(circleMembers)
			circleMembers = list(map(str, circleMembers))
			circleMembersString = ', '.join(circleMembers)
			logFile.write('{0} Members: {1} \n'.\
			format(key, circleMembersString))

		logFile.write('\n')
		logFile.write('------------\n')
		logFile.write('\n')
		allMembers = list(set(allMembers))
		allMembers = list(map(str, allMembers))
		allMembersString = ', '.join(allMembers)
		logFile.write('AllMembers: {0} \n'.format(allMembersString))
		logFile.write('\n')
		logFile.write('########################################################\n')
		logFile.write('\n')

def getMissingEdges(G, inDict, alphaVal):
	missingEdges = []
	for key, group in inDict.items():
		for ii in range(len(group)):
			for jj in range(ii+1, len(group)):
				flag = G.has_edge(str(group[ii]), str(group[jj]))
				if (flag==False) and (random.random()<=alphaVal):
					missingEdges.append((group[ii], group[jj]))
	Gcopy = copy.deepcopy(G)
	Gcopy.add_edges_from(missingEdges)

	return 	missingEdges, Gcopy

def getCategoryIdx(featureNames):
	compressedFeatureList = list(set(featureNames))
	startIdx = []
	endIdx = []
	for item in compressedFeatureList:
		startIdx.append(featureNames.index(item))
		endIdx.append(len(featureNames) - featureNames[::-1].index(item) - 1)

	return startIdx, endIdx, len(compressedFeatureList)

def cleanFeatureNames(featureNames):
	featureNamesOut = copy.deepcopy(featureNames)
	for ii in range(len(featureNames)):
		featureNamesOut[ii] = featureNamesOut[ii].replace(';anonymized','')
	return featureNamesOut

def hammingDistance(v1, v2, mode='flip'):
	if mode=='flip':
		return 1-distance.hamming(v1, v2)
	else:
		# This is better for Agg. clustering distance matrix
		return distance.hamming(v1, v2)


def getPairWiseDiff(inDict, u, v, featCategStIdx, featCategEndIdx, mode='flip'):
	hammingScore = 0
	count = 0
	for startIdx, endIdx in zip(featCategStIdx, featCategEndIdx):
		count += 1
		score = hammingDistance(inDict[int(u)][startIdx:endIdx+1], inDict[int(v)][startIdx:endIdx+1], mode)
		hammingScore += score
	return hammingScore*1.0/count

def getWeightedGraph(G, nodeFeaturesDict, featCategStIdx, featCategEndIdx, weight0, weight1):
	Gnew = nx.Graph()
	edgeList = []
	nodeSet = list(nodeFeaturesDict.keys())
	nodeSet = [str(item) for item in nodeSet]
	for ii in range(len(nodeSet)):
		for jj in range(ii+1, len(nodeSet)):
			flag = G.has_edge(str(nodeSet[ii]), str(nodeSet[jj]))
			pairWiseDiff = getPairWiseDiff(nodeFeaturesDict, nodeSet[ii], nodeSet[jj], featCategStIdx, featCategEndIdx)
			if flag==True:
				edgeList.append((nodeSet[ii], nodeSet[jj], 1 + weight1*pairWiseDiff ))
			else:
				edgeList.append((nodeSet[ii], nodeSet[jj], weight0*pairWiseDiff ))

	Gnew.add_weighted_edges_from(edgeList)
	return Gnew

def computeAllPairwiseDiffs(nodeSet, nodeFeaturesDict, featCategStIdx, featCategEndIdx):
	nodeSet = [int(item) for item in nodeSet]
	# Iterate over all possible edges:
	outMat = np.zeros(shape=(len(nodeSet), len(nodeSet)))
	for i in range(len(nodeSet)):
		n1 = nodeSet[i]
		outMat[i, i] = 0
		for j in range(i+1, len(nodeSet)):
			n2 = nodeSet[j]
			outMat[i, j] = getPairWiseDiff(nodeFeaturesDict, n1, n2, featCategStIdx, featCategEndIdx, mode='notFlip')
			outMat[j, i] = outMat[i, j]

	return outMat, nodeSet



paras = parse_args(sys.argv[1:])
egoNode = paras.egoNode
scale = paras.scale
specClustMethod = paras.specClustMethod
featClustMethod = paras.featClustMethod
maxNumComponents = paras.maxNumComponents
numClusters = paras.numClusters
covarType = paras.covarType
getBestNumComp = paras.getBestNumComp
numClusters = paras.numClusters
costType = paras.costType
subsetFlag = paras.subset
numPCA = paras.numPCA
alphaVal = paras.alphaVal
structAug = paras.structAug
pcaEnabled = paras.pcaEnabled
weight0 = paras.weight0
weight1 = paras.weight1
overlappingFlag = paras.overlappingFlag
gmmCutOff = paras.gmmCutOff
numRuns = paras.numRuns
csvFilePath = paras.csvFilePath

# Files
edgeListPath = './dataset/{0}.edges'.format(egoNode)
featureListPath = './dataset/{0}.feat'.format(egoNode)
egoFeatures = './dataset/{0}.egofeat'.format(egoNode)
egoCircles = './dataset/{0}.circles'.format(egoNode)
egoFeaturesNames = './dataset/{0}.featnames'.format(egoNode)
outFile = './out2/{0}.out'.format(egoNode)
logFile = './out2/{0}.log'.format(egoNode)
clusterMembers = './dataset/{0}.clustMembers'.format(egoNode)

# Read the Graph
fh = open(edgeListPath, 'rb')
G = nx.read_edgelist(fh)
nodeList = list(G.nodes())
nodeList = [int(item) for item in nodeList]
numNodes = len(nodeList)
print('Procesing Node {0}'.format(egoNode))
print('Graph Loaded. Number of nodes is {0}'.format(numNodes))
fh.close()

# Use only a subset of nodes that were reported in the ground truth circles
if subsetFlag==1:
	fNodes = open(clusterMembers, 'r')
	nodeSet = [line.strip().split() for line in fNodes]
	nodeSet = nodeSet[0]
	nodeSet = [item for item in nodeSet]
	G = G.subgraph(nodeSet)

# Load Circles
groundTruthCircles = getGroundTruthCircles(egoCircles)

# Load Features
egoFeatures = readFeatures(egoFeatures)

# Load Feature Names
featureNames = readFeatureNames(egoFeaturesNames)
featureNames = cleanFeatureNames(featureNames)
featCategStIdx, featCategEndIdx, numCategories = getCategoryIdx(featureNames)

# Load Features (One-hot)
nodeFeaturesDict = readFeatures(featureListPath)
nodeFeatures, nodeIDs = getMatrix(nodeFeaturesDict)

# Pairwise Features
pairWiseDiff, pairWiseIDs = computeAllPairwiseDiffs(nodeSet, nodeFeaturesDict, featCategStIdx, featCategEndIdx)
featClustObjAgglo = Clusters(groundTruthCircles, None, pairWiseIDs, pairWiseDiff, \
outFile, costType = costType, method='agglo', numClusters=numClusters, distanceMat='yes')
outString = '----------------Detour---------------\nClustering Based Only On Features Using Hierarchical \
Clustering\n costTye: {0} numClusters: {1} SuccessRate: {2}\n ----------------------------\n'.format(costType,numClusters, 1-featClustObjAgglo.finalError)
print(outString)

successRateVec = []
for runId in range(numRuns):

	# PCA Transformation on Features
	if structAug==0:
		inGraph = G
	elif structAug==1:
		if pcaEnabled==1:
			pca_model = PCAm(n_components=numPCA)
			Xcomps = pca_model.fit_and_decompose(nodeFeatures)
			featClustObj = Clusters(None, None, nodeIDs, Xcomps, None, method=featClustMethod, \
			numClusters=numClusters, maxNumComponents=maxNumComponents, getBestNumComp = getBestNumComp, \
			covarianceType=covarType, mode='features')
		else:
			featClustObj = Clusters(None, None, nodeIDs, nodeFeatures, None, method=featClustMethod, \
			numClusters=numClusters, maxNumComponents=maxNumComponents, getBestNumComp = getBestNumComp, \
			covarianceType=covarType, mode='features')
		# CHANGE: I replaced featClustObj with featClustObjAgglo for the following function
		missingEdges, Gnew = getMissingEdges(G, featClustObjAgglo.clusterDict, alphaVal)
		inGraph = Gnew
	elif structAug==2:
		# Perform Louvain's algorithm on the weighted adjacency matrix
		inGraph = getWeightedGraph(G, nodeFeaturesDict, featCategStIdx, featCategEndIdx, weight0, weight1)

	# Count Node Overlap in groundTruthCircles
	nodeOverlap = countNodeOverlap(groundTruthCircles, nodeList)
	numOverlap = sum([1 for item in nodeOverlap.values() if item>=2])
	percOverlap = numOverlap/numNodes
	#print('numOverlap: {0} PercentageOverlap: {1}'.format(numOverlap, percOverlap*100))

	# Reporting Isolated Nodes
	isolatedNodes = [key for key in nodeOverlap.keys() if nodeOverlap[key]==0]
	numIsolated = len(isolatedNodes)
	percIsolated = numIsolated/numNodes
	#print('numIsolated: {0} PercentageIsolated: {1}'.format(numIsolated , percIsolated*100))

	## Insights into circle structure
	#circleStat = getCircleStats(inGraph, groundTruthCircles)
	# circleStat is a dictionary with:
	# - key: circle id
	# - Value: (number of nodes in the circle, ratio of actual edges to possible number \
	#of edges, clustering coefficient)
	# The value (-1) represents a circle with a single member
	#writeLog(circleStat, logFile, egoNode, numOverlap, percOverlap, numIsolated, percIsolated, numNodes, groundTruthCircles)

	specClusterObj = Clusters(groundTruthCircles, inGraph, nodeIDs, nodeFeatures, outFile, method=specClustMethod, \
	numClusters=numClusters, maxNumComponents=maxNumComponents, getBestNumComp = getBestNumComp, \
	covarianceType=covarType, costType=costType, overlapping=overlappingFlag, gmmCutOff=gmmCutOff)
	successRateVec.append(1-specClusterObj.finalError)

outString = "Ego: {0} ClustMethod: {1} Scale: {2} getBestNumComp: {3} bestNumComp: {4} \
numClusters: {5} covarianceType: {6} maxNumComponents:{7} Metric: {8} SuccessRateAvg: {9} SuccessRateStd: {10}\n"\
.format(egoNode, specClustMethod, scale, getBestNumComp, specClusterObj.bestNumComp,\
numClusters, covarType, maxNumComponents, costType, np.average(successRateVec), np.std(successRateVec))
print(outString)

myData = [[structAug, egoNode, costType, np.average(successRateVec), len(specClusterObj.clusterDict), len(groundTruthCircles), 1-featClustObjAgglo.finalError]]
myFile = open(csvFilePath, 'a+')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(myData)
