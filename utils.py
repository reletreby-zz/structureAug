from __future__ import division
import sklearn.cluster
import sklearn.mixture
import community, pdb, datetime, collections
import numpy as np
from scipy.optimize import linear_sum_assignment

class Clusters():
	def __init__(self, groundTruthCircles, G, nodesID, dataIn, logFile, method='gmm', \
	numClusters=5, maxNumComponents=100, getBestNumComp = 0, \
	covarianceType='full', costType='BER', mode='wGroundTruth', overlapping=False, gmmCutOff=0.2, distanceMat='no'):
		self.bestNumComp = numClusters
		if method=='kmeans':
			self.model = sklearn.cluster.KMeans(n_clusters=numClusters)
			self.model.fit(dataIn)
			self.clusters = self.model.labels_
			self.clusterDict = self.constructClusterDict(self.clusters, nodesID)
		elif method=='gmm':
			if getBestNumComp==0:
				self.model = sklearn.mixture.GaussianMixture(n_components=numClusters, covariance_type=covarianceType)
				self.model.fit(dataIn)
				self.clusters = self.model.predict(dataIn)
				self.clusterDict = self.constructClusterDict(self.clusters, nodesID)
			else:
				self.bestNumComp = self.getBestNumComp(maxNumComponents, dataIn, covarianceType)
				print("Best number of components: {0}".format(self.bestNumComp))
				self.model = sklearn.mixture.GaussianMixture(n_components=self.bestNumComp, covariance_type=covarianceType)
				self.model.fit(dataIn)
				self.clusters = self.model.predict(dataIn)
				self.clusterDict = self.constructClusterDict(self.clusters, nodesID)
			if overlapping==True:
				classPostProb = self.model.predict_proba(dataIn)
				#classPostProb = self.normalize(classPostProb)
				numPossibleClusters = self.getNumPossibleClusters(classPostProb)
				pdb.set_trace()
		elif method=='spectral':
			self.model = sklearn.cluster.SpectralClustering(n_clusters=numClusters)
			self.model.fit(dataIn)
			self.clusters = self.model.labels_
			self.clusterDict = self.constructClusterDict(self.clusters, nodesID)
		elif method=='agglo':
			if distanceMat=='no':
				self.model = sklearn.cluster.AgglomerativeClustering(n_clusters=numClusters)
				self.model.fit(dataIn)
				self.clusters = self.model.labels_
				self.clusterDict = self.constructClusterDict(self.clusters, nodesID)
			else:
				self.model = sklearn.cluster.AgglomerativeClustering(n_clusters=numClusters, \
				affinity='precomputed', linkage='complete')
				self.model.fit(dataIn)
				self.clusters = self.model.labels_
				self.clusterDict = self.constructClusterDict(self.clusters, nodesID)

		elif method=='lova':
			self.clusterDict = community.best_partition(G)
			self.dendo = community.generate_dendrogram(G)
			self.clusterDict = self.adjustPartitionDict(self.clusterDict)
			self.bestNumComp = len(self.clusterDict)
			if (overlapping==True) and (len(self.dendo)>2):
				secondBestCluster = self.adjustPartitionDict(community.partition_at_level(self.dendo, len(self.dendo)-2))
				additionalClusters = self.findMergedClusters(self.clusterDict, secondBestCluster)
				self.appendClusters(additionalClusters)
		else:
			raise ValueError('Incorrect clustering method')

		if mode=='wGroundTruth':
			self.costMat, self.matchingDic, self.matchingRowIdx, self.matchingColumnIdx = \
			self.constructCostMat(self.clusterDict, groundTruthCircles, nodesID, costType)
			self.finalError = self.computeFinalError(costType, groundTruthCircles, nodesID)
			self.writeLog(logFile, nodesID, costType, groundTruthCircles)

	def getNumPossibleClusters(self, classPostProb):
		outList = []
		for ii in range(classPostProb.shape[0]):
			sumVals = sum([1 for item in classPostProb[ii] if item>0])
			outList.append(sumVals)
		return outList

	def normalize(self, classPostProb):
		# Input: classPostProb Un-normalized, i.e., column sum is larger than 1
		# Output: Normalized version of classPostProb
		sumVec = sum(classPostProb)
		outMat = np.zeros(shape=(classPostProb.shape[0], classPostProb.shape[1]))
		for ii in range(classPostProb.shape[1]):
			for jj in range(classPostProb.shape[0]):
				outMat[jj, ii] = classPostProb[jj, ii]*1.0/sumVec[ii]
		return 	outMat

	def appendClusters(self, additionalClusters):
		currMaxKey = max(list(self.clusterDict.keys()))
		for ii in range(len(additionalClusters)):
			self.clusterDict[currMaxKey+ii+1] = additionalClusters[ii]

	def findMergedClusters(self, mergedDict, originalDict):
		additionalClusters = []
		for key, value in originalDict.items():
			# Check if Value appears as is in any of mergedDict
			Flag = 0
			for keyMerged, valueMerged in mergedDict.items():
				if sorted(value)==sorted(valueMerged):
					# Then this cluster wasn't merged in the merging process
					Flag = 1
			if Flag==0:
				# The cluster under consideration was later merged in the final stage with another cluster
				additionalClusters.append(value)
		return additionalClusters



	def computeFinalError(self, costType, groundTruthCircles, nodesID):
		# The objective is to compute the final error of this particular
		# cluster assignment
		# Given: self.matchingDic, costType
		# Output: Error
		count = 0
		error = 0
		for key in self.matchingDic:
			myCircle = self.clusterDict[key]
			trueCircle = groundTruthCircles[self.matchingDic[key]]
			error += self.getError(myCircle, trueCircle, nodesID, costType)
			count += 1

		if len(self.clusterDict)>len(self.matchingDic):
			for ii in range(len(self.clusterDict) - len(self.matchingDic)):
				count += 1
				if costType=='BER':
					error += 0.5
				elif costType=='fScore':
					error += 1
				else:
					raise ValueError('Incorrect costType')
		return error/count


	def constructCostMat(self, myClusters, trueClusters, nodesID, costType):
		myClustersKeys = list(myClusters.keys())
		trueClustersKeys = list(trueClusters.keys())
		costMat = np.zeros(shape=(len(myClustersKeys),len(trueClustersKeys)))
		for ii in range(len(myClustersKeys)):
			for jj in range(len(trueClustersKeys)):
				myCircle = myClusters[myClustersKeys[ii]]
				trueCircle = trueClusters[trueClustersKeys[jj]]
				costMat[ii,jj] = self.getError(myCircle, trueCircle, nodesID, costType)

		rowIdx, columnIdx = linear_sum_assignment(costMat)
		columnIdx = [trueClustersKeys[ii] for ii in columnIdx]
		rowIdx = [myClustersKeys[ii] for ii in rowIdx]
		if len(rowIdx)!=len(columnIdx): raise ValueError('Something Wrong Here')
		matchingDic = {ii:jj for ii, jj in zip(rowIdx, columnIdx)}

		return costMat, matchingDic, rowIdx, columnIdx



	def getError(self, myCircle, trueCircle, nodesID, costType):
		if costType=='BER':
			return self.BER(myCircle, trueCircle, nodesID)
		elif costType=='fScore':
			return self.fScore(myCircle, trueCircle)
		else:
			raise ValueError('Something Wrong Here')

	def divFun(self, a,b):
		if b==0:
			return 0
		else:
			return a/b

	def BER(self, myCircle, trueCircle, nodesID):
		myCircleSet = set(myCircle)
		myCircleConjSet = set(nodesID).difference(myCircleSet)
		trueCircleSet = set(trueCircle)
		trueCircleSetConjSet = set(nodesID).difference(trueCircleSet)
		error = self.divFun(0.5*len(myCircleSet.difference(trueCircleSet)), len(myCircleSet)) +\
		self.divFun(0.5*len(myCircleConjSet.difference(trueCircleSetConjSet)), len(myCircleConjSet))
		return error

	def fScore(self, myCircle, trueCircle):
		myCircleSet = set(myCircle)
		trueCircleSet = set(trueCircle)
		precision = len(myCircleSet.intersection(trueCircleSet))*1.0/len(myCircleSet)
		recall = len(myCircleSet.intersection(trueCircleSet))*1.0/len(trueCircleSet)

		# return 1-fScore
		return 1-self.divFun(2*precision*recall, precision+recall)

	def constructClusterDict(self, clusters, Ids):
		numClusters = len(set(self.clusters))
		outDict = collections.defaultdict(int)
		for k in range(numClusters):
			outDict[k] = [Ids[ii] for ii in range(len(Ids)) if clusters[ii]==k]
		return outDict

	def adjustPartitionDict(self, partitionDict):
		# Given a dictionary node:cluster, transform into
		# cluster:node
		outDict = collections.defaultdict(list)
		clusterList = set(partitionDict.values())
		for cluster in clusterList:
			outDict[cluster] = [int(key) for key in partitionDict.keys() if partitionDict[key]==cluster]
		return outDict


	def getBestNumComp(self, maxNumComponents, dataIn, covarianceType):
		errorDic = collections.defaultdict(float)
		for numComp in range(2, maxNumComponents+1):
			model = sklearn.mixture.GaussianMixture(n_components=numComp, covariance_type=covarianceType)
			model.fit(dataIn)
			errorDic[numComp] = model.aic(dataIn)

		return sorted(errorDic, key = lambda x:errorDic[x])[0]

	def writeLog(self, logFile, nodesID, costType, groundTruthCircles):
		with open(logFile, mode='a+') as logFile:
			logFile.write('Time: {0}\n'.format(str(datetime.datetime.today())))
			logFile.write('-.-.-.-.\n')
			for key in self.matchingDic.keys():
				logFile.write('PredictedCircleIdx: {0}\n'.format(key))
				predMembers = sorted(self.clusterDict[key])
				predictedCircleMembers = list(map(str,predMembers))
				predictedCircleMembers = ' '.join(predictedCircleMembers)
				logFile.write('PredictedCircleMembers: {0}\n'.format(predictedCircleMembers))
				logFile.write('ActualCircleIdx: {0}\n'.format(self.matchingDic[key]))
				actualMembers = sorted(groundTruthCircles[self.matchingDic[key]])
				actualCircleMembers = list(map(str,actualMembers))
				actualCircleMembers = ' '.join(actualCircleMembers)
				logFile.write('ActualCircleMembers: {0}\n'.format(actualCircleMembers))
				errorVal = self.getError(predMembers, actualMembers, nodesID, costType)
				predMembers = set(predMembers)
				actualMembers = set(actualMembers)
				logFile.write('NumIntersection: {0} NumDiff: {1} Metric: {2} Accuracy: {3}\n'\
				.format(len(predMembers.intersection(actualMembers)), len(predMembers.difference(actualMembers)), costType, 1-errorVal ))
				logFile.write('-----------\n')
			logFile.write('\n')
			logFile.write('########################################################\n')
			logFile.write('\n')
