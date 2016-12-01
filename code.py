#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA

def readCSV():
	#train = pd.read_csv('data/netflix_train.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False)
	train = pd.read_csv('data/test.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False)
	test = pd.read_csv('data/netflix_train.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False)
	uid = pd.read_csv('data/users.txt',header=None,names=['uid'],index_col=False)
	fid = pd.read_csv('data/movie_titles.txt',header=None,names=['fid','year','name'],index_col=False)
	return (train,test,uid,fid)

def showData(data):
	data.info()
	print data.head()

def procData(data):
	uids = data.drop_duplicates(['uid'])['uid']
	fids = data.drop_duplicates(['fid'])['fid']
	df = pd.DataFrame(index= uids,columns=fids)
	df = df.fillna(0)
	for index, d in data.iterrows():
                df.loc[d['uid'],d['fid']]=d['score']
	return df

def procData_np(data):
	uids = data.uid.unique().shape[0]
	fids = data.fid.unique().shape[0]
	df = np.zeros((uids,fids))
	for line in data.itertuples():
	        df[line[1]-1, line[2]-1] = line[3]
	return df

def RMSE(X_test,X_pred):
	x,y = X_test.shape
	return np.sqrt((np.square(X_test-X_pred).sum().sum())/(x*y))

def RMSE_np(X_test,X_pred):
	return np.sqrt(mean_squared_error(X_pred, X_test))

def Task1(X_train,X_test):
	#为了减少运算量，先计算向量和
	tmp = pd.DataFrame(index=X_train.index)
	tmp['sum'] = np.sqrt(np.square(X_train).sum(axis=1))
	#计算用户相似度
	sim = pd.DataFrame(index=X_train.index,columns=X_train.index)
	sim = sim.fillna(0)
	_counted = []
	for x in X_train.index:
		#不重复计算
		_counted.append(x)
		sim.loc[x,x] = 1
		for k in X_train.index:
			if k in _counted:
				sim.loc[x,k] = sim.loc[k,x]
				continue
			for y in X_train.columns:
				sim.loc[x,k] += X_train.loc[x,y]*X_train.loc[k,y]
			sim.loc[x,k] = sim.loc[x,k]/(tmp.loc[x,'sum']*tmp.loc[k,'sum'])
	#预测测试集中的值
	tmp = pd.DataFrame(index=X_test.index,columns=X_test.columns)
	for x in X_test.index:
		for y in X_test.columns:
			sum1 = 0
			sum2 = 0
			for k in X_train.index:
				sum1 += sim.loc[k,x]*X_train.loc[k,y]
				sum2 += sim.loc[k,x]
			tmp.loc[x,y]=sum1/sum2
	#评估准确率
	print RMSE(X_test,tmp)

def Task1_np(X_train,X_test):
	#计算相似度
	similarity = cosine_similarity(X_train)
	#计算测试集值
	pred = similarity.dot(X_train) / np.array([np.abs(similarity).sum(axis=1)]).T
	#评估准确率
	print RMSE_np(X_test,pred)

def random(k,X_train):
	m,n = X_train.shape
	u = np.random.rand(m,k)
	v = np.random.rand(n,k)
	a = np.where(X_train>0,1,X_train)
	return (a,u,v)

def fro(a):
	return np.sqrt(np.square(a).sum())

def calJ(a,x,u,v,_lambda):
	j = LA.norm(a*(x-u.dot(v.T)),'fro')**2/2+_lambda*LA.norm(u,'fro')**2+_lambda*LA.norm(v,'fro')**2
	#j = fro(a*(x-u.dot(v.T)))**2/2+_lambda*fro(u)**2+_lambda*fro(v)**2
	return j

def calJU(a,x,u,v,_lambda):
	ju = (a*(u.dot(v.T)-x)).dot(v)+2*_lambda*u
	return ju

def calJV(a,x,u,v,_lambda):
	jv = (a*(u.dot(v.T)-x)).T.dot(u)+2*_lambda*v
	return jv

def Task2(X_train,X_test):
	ks = [50]
	_lambdas = [0.01]
	for k in ks :
		for _lambda in _lambdas:
			(a,u,v) = random(k,X_train)
			count = 0
			small = 0.0001
			alphas = [2]
			res = []
			for alpha in alphas :
				while (count<10 and calJ(a,X_train,u,v,_lambda)>small) :
					count += 1
					u = u - alpha*calJU(a,X_train,u,v,_lambda)
					v = v - alpha*calJV(a,X_train,u,v,_lambda)
			x = u.dot(v.T)
			print RMSE_np(X_train,x)
			#print X_test
			#print x
			#tmp = pd.DataFrame(index=X_test.index,columns=X_test.columns)
			#nx,ny =  X_test.shape
			#for x in xrange(nx):
			#	for y in xrange(ny):
			#		tmp.loc[x[x,1],x[0,y]] = x[x,y] 
			#res.append(RMSE(X_test,tmp))
			#print res
	pass

def plotData(x,y,tstr):
	plt.plot(x,y)
	plt.title(tstr)
	plt.xlabel(tstr)
	plt.ylabel('RMSE')
	plt.show()

if __name__ == '__main__':
	(train,test,uid,fid) = readCSV()
	'''
	X_train = procData(train)
	X_test = procData(test)

	import time
	starttime = time.time()
	start = time.clock()

	Task1(X_train,X_test)

	end = time.clock()
	endtime = time.time()
	print 'time : ',str(end-start)
	print 'time : ',str(endtime - starttime)
	'''

	X_train = procData_np(train)
	#X_test = procData_np(test)
	
	#starttime = time.time()
	#start = time.clock()

	#Task1_np(X_train,X_train)	
	Task2(X_train,X_train)

	#end = time.clock()
        #endtime = time.time()
        #print 'time : ',str(end-start)
        #print 'time : ',str(endtime - starttime)
