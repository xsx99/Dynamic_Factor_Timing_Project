#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:14:28 2017

@author: kunmingwu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_cleaning as DC
import imp
from cvxpy import *
    
imp.reload(DC)
plt.style.use('ggplot')

# notice that PE seems very out of place in terms of return
data = pd.read_csv('./data/asset_return.csv')
#data.drop(['PE'], axis=1, inplace=True)
col = data.columns
data['Date']=pd.to_datetime(data['Date'])
data.set_index(data['Date'],inplace=True)
del data['Date']


def summary_stats(p_ret, p_cumret, b_ret, rf_ret):
    # max drawdown
    x = p_cumret
    i = np.argmax(np.maximum.accumulate(x) - x) # end of the period
    j = np.argmax(x[:i]) # start of period
    plt.plot(x)
    plt.plot([i, j], [x[i], x[j]], 'o', color='Red', markersize=10)
    max_dd = x[j] - x[i]
    max_dd_period = i-j
    # SR
    SR = np.mean(p_ret - rf_ret)/np.std(p_ret) * np.sqrt(12)
    # IR
    IR = np.mean(p_ret - b_ret)/np.std(p_ret - b_ret) * np.sqrt(12)
    # cumulative return
    total_ret = (p_cumret[len(p_cumret)-1] - p_cumret[0])/p_cumret[0]
    # mean, std of return
    mean_ret = np.mean(p_ret)
    std_ret = np.std(p_ret)
    output = pd.DataFrame([mean_ret, std_ret, total_ret, SR, IR, max_dd, max_dd_period]).T
    output.columns = ['mean_ret', 'std_ret', 'total_ret', 'SR','IR', 'max_dd', 'max_dd_period']
    return output

###inputs: weights is a matrix, row: end of period date, col: asset weight
###inputs: data is a dataframe, row: end of period date, col: asset return 
def portfolio(data,weights,legend): 
    p0 = 1.00 * weights[0]/float(np.sum(weights[0])) # initial dollar value 
    p_sum = [1]
    for i in range(1,len(data)):
        # monthly return less holding costs
        weight = weights[i] * 1.00/float(np.sum(weights[i]))
        p1 = p0 * np.exp(data.ix[i] - holding_costs)
        diff = np.abs(p1 - weight * np.sum(p1))
        p1 = weight * (np.sum(p1) - np.dot(diff, trading_costs))
        p_sum.append(np.sum(p1))
        p0=p1
    #p_ret = (pd.Series(p_sum).diff()/pd.Series(p_sum).shift(1))[1:]  
    p_sum=pd.Series(p_sum)
    #p_ret =  np.log(p_sum)-np.log(p_sum).shift(1)
    p_ret = p_sum.diff()/p_sum.shift(1)
    p_ret=p_ret[1:]    
    plt.plot(pd.to_datetime(data.index), p_sum, label=legend)
    plt.legend(bbox_to_anchor=(0.5,-0.05),loc=0)
    #plt.plot(pd.to_datetime(data.index)[1:], p_ret, label='Monthly Return')
    #plt.legend(loc='best')
    return np.array(p_ret), p_sum




def risk_parity(freq=12):
    weights=[list(data.iloc[range(0,12),:].apply(lambda x: 1/np.std(x),axis=0))]
    for i in range(1,len(data)-12):
        weights=np.vstack((weights,list(data.iloc[range(i,i+12),:].apply(lambda x: 1/np.std(x),axis=0))))
    return weights





def exp_weight(alpha,n):
    exp_w=np.array([(1-alpha)*alpha**i for i in range(n,0,-1)])
    exp_w=exp_w/sum(exp_w)
    return exp_w




def ew_mean(ret,alpha):
    exp_w=exp_weight(alpha,len(ret))
    ex_ret=sum(ret*exp_w)
    return ex_ret





def ew_mean_multiple(m_ret,alpha):
    m_ret=np.array(m_ret)
    exp_w=exp_weight(alpha,m_ret.shape[0])
    exp_ret=[]
    for i in range(m_ret.shape[1]):
        exp_ret.append(sum(m_ret[:,i]*exp_w))
    return exp_ret
 

   
    
def ew_cov(m_ret,alpha):
    m_ret=np.array(m_ret)
    exp_w=exp_weight(alpha,m_ret.shape[0])    
    m=m_ret.shape[1]
    cov=np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            exp_ret_i=ew_mean(m_ret[:,i],alpha)
            exp_ret_j=ew_mean(m_ret[:,j],alpha)
            cov[i,j]=sum((m_ret[:,i]-exp_ret_i)*(m_ret[:,j]-exp_ret_j)*exp_w)
            cov[j,i]=cov[i,j]
    return cov


    

def risk_parity2(alpha=0.8):
    weights=[]
    for j in range(data.shape[0]):
        ## calculate exponential weights
        exp_w=np.array([(1-alpha)*alpha**i for i in range(j,-1,-1)])
        exp_w=exp_w/sum(exp_w)
        exp_ret=np.array(data.iloc[0:(j+1),:])
        w=[]        
        for i in range(exp_ret.shape[1]):
            ##calculate exponential weighted return            
            exp_ret[:,i]=exp_ret[:,i]*exp_w
            exp_var=sum((data.iloc[0:(j+1),i]-sum(exp_ret[:,i]))**2*exp_w)
            w.append(1/np.sqrt(exp_var))
        weights.append(w)
    weights=np.array(weights)
    weights = weights[12:,:]
    return weights
    
    


##mean variance without trading cost
#r: forecasted asset return, dataframe
#c: estimated covariance matrix, matrix
# lbd: risk aversion, float
def model0(r,c,lbd):
    r=np.array(r)
    n=len(r)
    x=Variable(n)
    p=Problem(Maximize(r*x-lbd*(quad_form(x, c))),[x>=0,sum_entries(x)==1])
    p.solve()
    w=x.value
    return np.array(w)



##mean variance with trading cost
#r: forecasted asset return, dataframe
#c: estimated covariance matrix, matrix
# lbd: risk aversion, float
#p: start portfolio value, array
def model1(r,c,lbd,p):
    r=np.array(r)
    n=len(r)
    x=Variable(n)
    p=Problem(Maximize(r*x-lbd*(quad_form(x, c))-trading_costs*abs(np.sum(p)*x-p)-holding_costs*(np.sum(p)*x)),[x>=0,sum_entries(x)==1])
    p.solve()
    w=x.value
    return np.array(w)
    
    
    
  
    

def mv_portfolio(data,legend,lbd):  
    weights=[]
    ##the first alpha and cov
    alpha=data.iloc[11,:]
    cov=np.cov(data.iloc[0:12,:].T) 
    w0=model0(alpha,cov,lbd)
    w0=w0.reshape(len(w0))
    weights.append(w0)
    p0 = 1 * w0/np.sum(w0) # initial dollar value 
    p_sum = [1]
    for i in range(12,len(data)):
        ## forecasted asset return: simply take current period's return
        alpha=data.iloc[i,:]
        ## estimated asset covariance: simply use previous 12 months' covariance
        cov=np.cov(data.iloc[i-11:i+1,:].T)
        w1=model1(alpha,cov,lbd,p0)
        w1=w1.reshape(len(w1))
        weights.append(w1)
        p1 = p0 * np.exp(data.ix[i] - holding_costs)
        diff = np.abs(p1 - w1 * np.sum(p1))
        p1 = w1 * (np.sum(p1) - np.dot(diff, trading_costs))
        p_sum.append(np.sum(p1))
        p0=p1
    p_sum=pd.Series(p_sum)
    p_ret =  np.log(p_sum)-np.log(p_sum).shift(1)
    p_ret=p_ret[1:]    
    plt.plot(pd.to_datetime(data.index[11:]), p_sum, label=legend)
    plt.legend(bbox_to_anchor=(0.5,-0.05),loc=0)
    #plt.legend(loc='best')
    return p_sum, np.array(weights)




def mv_portfolio_ew(data,legend,lbd,alpha):  
    weights=[]
    ##the first alpha and cov
    ret=ew_mean_multiple(data.iloc[0:12,:],alpha)
    cov=ew_cov(data.iloc[0:12,:],alpha) 
    w0=model0(ret,cov,lbd)
    w0=w0.reshape(len(w0))
    weights.append(w0)
    p0 = 1 * w0/np.sum(w0) # initial dollar value 
    p_sum = [1]
    for i in range(12,len(data)):
        ## forecasted asset return: simply take current period's return
        ret=ew_mean_multiple(data.iloc[i-11:i+1,:],alpha)
        ## estimated asset covariance: simply use previous 12 months' covariance
        cov=ew_cov(data.iloc[i-11:i+1,:],alpha)
        w1=model1(ret,cov,lbd,p0)
        w1=w1.reshape(len(w1))
        weights.append(w1)
        p1 = p0 * np.exp(data.ix[i] - holding_costs)
        diff = np.abs(p1 - w1 * np.sum(p1))
        p1 = w1 * (np.sum(p1) - np.dot(diff, trading_costs))
        p_sum.append(np.sum(p1))
        p0=p1
    p_sum=pd.Series(p_sum)
    p_ret =  np.log(p_sum)-np.log(p_sum).shift(1)
    p_ret=p_ret[1:]    
    plt.plot(pd.to_datetime(data.index[11:]), p_sum, label=legend)
    plt.legend(bbox_to_anchor=(0.5,-0.05),loc=0)
    #plt.legend(loc='best')
    return p_sum, np.array(weights)
    
#==============================================================================
# TRADING/HOLDING COSTS
#==============================================================================
# please refer to page 5 of the project description
# the following are estimates 

trading_costs = np.array([0.0005, 0.0010, 0.0015, 0.0000, 0.0030, 0.0040, 0.0100,0.0100])
holding_costs = np.array([0.0000, 0.0010, 0.0005, 0.0000, 0.0015, 0.0025, 0.0000,0.0000])/12


