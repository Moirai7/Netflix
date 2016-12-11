#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
import gc

def readCSV():
	train = pd.read_csv('/Users/emma/Work/Netflix/data/netflix_train.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False,parse_dates=['time'])
	test = pd.read_csv('/Users/emma/Work/Netflix/data/netflix_test.txt',sep=' ',header=None,names=['uid','fid','score','time'],index_col=False)
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

def procData(train,test):
	uids = train.drop_duplicates(['uid'])['uid'].sort_values()
	fids = train.drop_duplicates(['fid'])['fid'].sort_values()
	tfids = test.drop_duplicates(['fid'])['fid'].sort_values()
	
	diff = list(set(fids.tolist())-set(tfids.tolist()))
	diff = list(map(lambda x: x-1,diff))
	#fids = pd.concat([diff,fids]).drop_duplicates(keep=False)

	df1 = pd.DataFrame(index= uids,columns=fids)
	df2 = pd.DataFrame(index= uids,columns=tfids)
	for index, d in train.iterrows():
               	df1.loc[d['uid'],d['fid']]=d['score']
	for index, d in test.iterrows():
             	df2.loc[d['uid'],d['fid']]=d['score']
	#del train
	#del test
	gc.collect()
	df1 = df1.fillna(0)
	df2 = df2.fillna(0)
	return (df1,df2,diff)

def procData_sim(train):
	uids = train.drop_duplicates(['uid'])['uid']
	fids = train.drop_duplicates(['fid'])['fid'].sort_values()
	df = pd.DataFrame(index= fids,columns=fids)
	for uid in uids:
		x = train[train['uid']==uid]
		x = x.sort_values(['time','fid'])
		for i in xrange(0,len(x)-1):
			timedelta = float((x.iloc[i+1,3]-x.iloc[i,3]).total_seconds())/86400+1
			if timedelta>3:
				continue
			df.loc[x.iloc[i,1],x.iloc[i+1,1]]=0.2**(timedelta)
			df.loc[x.iloc[i+1,1],x.iloc[i,1]]=0.2**(timedelta)
	print df.values.shape
	df = df.fillna(0)
	return df
'''
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
'''

def RMSE_np(X_test,X_pred):
	return np.sqrt(mean_squared_error(X_pred, X_test))

def Task1_np(X_train,X_test,diff):
	#计算相似度
	similarity = cosine_similarity(X_train)
	#计算测试集值
	pred = similarity.dot(X_train) / np.array([np.abs(similarity).sum(axis=1)]).T
	#评估准确率
	print X_test
	print pred
	pred = np.delete(pred,diff,axis=1)
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

def Task2(x,X_test,diff):
	#ks = [10,20,30,40,50,60,70,80,90]
	#ks = np.arange(10,40,10)
	ks=[30]
	mins = {'k':0,'lambda':0,'alpha':0,'min':1000}
	for k in ks :
		#_lambdas = [0.001,0.003,0.006,0.009,0.01,0.03,0.06,0.09,0.1]
		#_lambdas = np.arange(0.001,0.1,0.04)
		#_lambdas = [0.001,0.009,0.002]
		_lambdas = [0.009]
		for _lambda in _lambdas:
			#alphas = np.arange(0.000001,0.00001,0.000001)
			#alphas =   [0.00006]
			#alphas = np.arange(0.00001,0.0001,0.00004)
			alphas = [0.00009]
			res = []
			resj = []
			for alpha in alphas :
				(a,u,v) = random(k,x)
				count = 0
				j = calJ(a,x,u,v,_lambda)
				iters = []
				while (count<10 and j>0.6) :
					count += 1
					u = u - alpha*calJU(a,x,u,v,_lambda)
					v = v - alpha*calJV(a,x,u,v,_lambda)
					j = calJ(a,x,u,v,_lambda)
					'''
					if count%3 == 0:
						iters.append(count)
						resj.append(j)
						pred = u.dot(v.T)
						pred = np.delete(pred,diff,axis=1)
						rmse = RMSE_np(X_test,pred)
						res.append(rmse)
						if count/3 == 15:
							plotData(iters,res,'alpha','RMSE')
							plotData(iters,resj,'alpha','J')
					'''
				print "#################"
				print 'k ',str(k)
				print 'lambda ',str(_lambda)
				print 'alpha ',str(alpha)
				print 'j',str(j)
				pred = u.dot(v.T)
				pred = np.delete(pred,diff,axis=1)
				rmse = RMSE_np(X_test,pred)
				print 'rmse ',str(rmse)
				if mins['min']>rmse:
					mins['k']=k
					mins['lambda']=_lambda
					mins['alpha']=alpha
					mins['min']=rmse
	print mins
	pass

def plotData(x,y,xstr,ystr):
	plt.plot(x,y)
	plt.title(xstr)
	plt.xlabel(xstr)
	plt.ylabel(ystr)
	plt.xticks(x)
	plt.yticks(y)
	plt.show()

def Task3(sim,X_train,X_test,diff):
	similarity = pairwise_distances(X_train.T, metric='cosine')+sim
	pred = X_train.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)]) 
	print X_test
        print pred
        pred = np.delete(pred,diff,axis=1)
        print RMSE_np(X_test,pred)

if __name__ == '__main__':
	#print 'start read'
	#(train,test,uid,fid) = readCSV()
	#print 'start proc'
	#sim = procData_sim(train)
	#sim.to_csv('data/sim.pkl')
	#X_train,X_test,diff = procData(train,test)
	#X_train.to_csv('data/train.pkl')
        #X_test.to_csv('data/test.pkl')
	
	X_train = pd.read_csv('data/train.pkl',index_col='uid')
	X_test = pd.read_csv('data/test.pkl',index_col='uid')
	sim = pd.read_csv("data/sim.pkl",index_col='fid')
	sim = sim.fillna(0)
	diff = [448, 3233, 8213, 4293, 2536, 6902, 6635, 5036, 5485, 7374, 8145, 4372, 7029, 5192, 9978, 4859, 9118]
        import time
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

	#Task1_np(X_train.values,X_test.values,diff)	
	#Task2(X_train.values,X_test.values,diff)
	Task3(sim.values,X_train.values,X_test.values,diff)
	end = time.clock()
        endtime = time.time()
        print 'time : ',str(end-start)
        print 'time : ',str(endtime - starttime)
