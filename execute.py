import os, time, sys

pyCommand       = 'python'
codeVer         = 4
#egoNode        = int(sys.argv[1])
specClustMethod = 'lova'
subset          = 1
structAug       = 2
alphaVal        = 0.1
numPCA          = 5
numClusters     = 10
pcaEnabled      = 0
costType        = 'fScore'
weight0         = 0.5
weight1         = 0.8
overlappingFlag = 1
csvFilePath     = './out2/{0}_{1}_{2}.csv'.format(costType, int(weight0*100), int(weight1*100))
egoNodeList     = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]

for egoNode in egoNodeList:
    os.system("{0} main_v{1}.py -specClustMethod {2} -subset {3} \
    -structAug {4} -alphaVal {5} -numPCA {6} -numClusters {7} \
    -pcaEnabled {8} -costType {9} -egoNode {10} -weight0 {11} -weight1 {12} -overlappingFlag {13} -csvFilePath {14}".\
    format(pyCommand, codeVer, specClustMethod, subset, \
    structAug, alphaVal, numPCA, numClusters, pcaEnabled, costType, egoNode, weight0, weight1, overlappingFlag, csvFilePath))
