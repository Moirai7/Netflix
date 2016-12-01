#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import mean_squared_error


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
	X_test = X_test[X_test.nonzero()].flatten()
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
	print RMSE(X_test,pred)

def calJ(a,x,u,v,_lambda):
	j = 0
	return j

def calJU(a,x,u,v,_lambda):
	ju = 0
	return ju

def calJV(a,x,u,v,_lambda):
	jv = 0
	return jv

def random(k,X_train):
	u = []
	v = []
	a = []
	return (a,u,v)

def Task2(X_train,X_test):
	k = [0]
	_lambdas = [1]
	for k in ks :
		for _lambda in _lambdas:
			(a,u,v) = random(k,X_train)
			count = 0
			small = 0.0001
			alphas = [0.0001]
			res = []
			for alpha in alphas :
				while (count<1000 or calJ(a,x,u,v,_lambda)>small) :
					u = u - alpha*calJU()
					v = v - alpha*calJV()
			tmp = pd.DataFrame(index=X_test.index,columns=X_test.columns)
			for x in X_test.index:
				for y in X_test.columns:
					tmp.loc[x,y] = X_train.loc[x,y]
			res.append(RMSE(X_test,tmp))
	pass

if __name__ == '__main__':
	(train,test,uid,fid) = readCSV()
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

	X_train = procData_np(train)
	X_test = procData_np(test)
	
	starttime = time.time()
	start = time.clock()

	Task1_np(X_train,X_test)	

	end = time.clock()
        endtime = time.time()
        print 'time : ',str(end-start)
        print 'time : ',str(endtime - starttime)
