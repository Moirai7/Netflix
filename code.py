#!/usr/b	in/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA

def readCSV():
	train = pd.read_csv('data/netflix_train.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False)
	test = pd.read_csv('data/netflix_train.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False)
	#train = pd.read_csv('data/test.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False)
	#test = pd.read_csv('data/test.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False)
	#uid = pd.read_csv('data/users.txt',header=None,names=['uid'],index_col=False)
	#fid = pd.read_csv('data/movie_titles.txt',header=None,names=['fid','year','name'],index_col=False)
	uid = []
	fid = []
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
	x = X_train.values
	#ks = [10,20,30,40,50,60,70,80,90]
	ks = np.arange(10,90,5)
	maxs = {'k':0,'lambda':0,'alpha':0,'max':1000}
	for k in ks :
		#_lambdas = [0.001,0.003,0.006,0.009,0.01,0.03,0.06,0.09,0.1]
		_lambdas = np.arange(0.001,0.1,0.004)
		for _lambda in _lambdas:
			#alphas = [0.0001,0.0003,0.0006,0.0009,0.001,0.003,0.006,0.009,0.01,0.03,0.06,0.09,0.1]
			alphas = np.arange(0.0001,0.1,0.0004)
			res = []
			resj = []
			for alpha in alphas :
				(a,u,v) = random(k,x)
				count = 0
				j = calJ(a,x,u,v,_lambda)
				while (count<10000 and j>0.001) :
					count += 1
					u = u - alpha*calJU(a,x,u,v,_lambda)
					v = v - alpha*calJV(a,x,u,v,_lambda)
					j = calJ(a,x,u,v,_lambda)
				resj.append(j)
				pred = u.dot(v.T)
				pred = pd.DataFrame(pred,index=X_train.index,columns=X_train.columns)
				tmp = pd.DataFrame(index=X_test.index,columns=X_test.columns)
				for i in X_test.index:
					for j in X_test.columns:
						tmp.loc[i,j] = pred.loc[i,j]
				print "#################"
				print 'k ',str(k)
				print 'lambda ',str(_lambda)
				print 'alpha ',str(alpha)
				#print '\n'
				#print tmp
				rmse = RMSE_np(X_test,tmp)
				if maxs['max']>rmse:
					maxs['k']=k
					maxs['lambda']=_lambda
					maxs['alpha']=alpha
					maxs['max']=rmse
				res.append(rmse)
				print 'rmse ',str(rmse)
			print 'result: lambda=',str(_lambda),'k=',str(k)
			print res
			print resj
			#if _lambda==0.01 and k==50:
			#	plotData(alphas,res,'alpha','RMSE')
			#	plotData(alphas,resj,'alpha','J')
	print maxs
	pass

def plotData(x,y,xstr,ystr):
	plt.plot(x,y)
	plt.title(xstr)
	plt.xlabel(xstr)
	plt.ylabel(ystr)
	plt.xticks(x)
	plt.yticks(y)
	plt.show()

if __name__ == '__main__':
	print 'start read train'
	print 'start read test'
	(train,test,uid,fid) = readCSV()
	print 'start proc train'
	X_train = procData(train)
	print 'start proc test'
	X_test = procData(test)

	'''
	starttime = time.time()
	start = time.clock()

	Task1(X_train,X_test)

	end = time.clock()
	endtime = time.time()
	print 'time : ',str(end-start)
	print 'time : ',str(endtime - starttime)

	#X_train = procData_np(train)
	#X_test = procData_np(test)
	'''

	import time
	print 'start clock'
	starttime = time.time()
	start = time.clock()

	#Task1_np(X_train.values,X_test.values)	
	Task2(X_train,X_test)

	end = time.clock()
        endtime = time.time()
        print 'time : ',str(end-start)
        print 'time : ',str(endtime - starttime)
