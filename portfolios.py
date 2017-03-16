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
import statsmodels.api as sm    
imp.reload(DC)
plt.style.use('ggplot')




def summary_stats(p_ret, p_cumret, b_ret,rf_ret):
    # max drawdown
    x = p_cumret
    i = np.argmax(np.maximum.accumulate(x) - x) # end of the period
    j = np.argmax(x[:i]) # start of period
    #plt.plot(x)
    #plt.plot([i, j], [x[i], x[j]], 'o', color='Red', markersize=10)
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
    output = pd.DataFrame([mean_ret, std_ret, total_ret, IR, SR, max_dd, max_dd_period]).T
    output.columns = ['mean_ret', 'std_ret', 'total_ret', 'IR', 'SR','max_dd', 'max_dd_period']
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
    alpha=np.mean(data.iloc[0:12,:],axis=0)
    cov=np.cov(data.iloc[0:12,:].T) 
    w0=model0(alpha,cov,lbd)
    w0=w0.reshape(len(w0))
    weights.append(w0)
    p0 = 1 * w0/np.sum(w0) # initial dollar value 
    p_sum = [1]
    for i in range(12,len(data)):
        ## forecasted asset return: simply take current period's return
        alpha=np.mean(data.iloc[i-11:i+1,:],axis=0)
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

    #plt.plot(pd.to_datetime(data.index[11:]), p_sum, label=legend)
    #plt.legend(bbox_to_anchor=(0.5,-0.05),loc=0)
    ##plt.legend(loc='best')
    return p_ret, p_sum, np.array(weights)




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
        ret=ew_mean_multiple(data.iloc[:i+1,:],alpha)
        ## estimated asset covariance: simply use previous 12 months' covariance
        cov=ew_cov(data.iloc[:i+1,:],alpha)
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
    #plt.plot(pd.to_datetime(data.index[11:]), p_sum, label=legend)
    #plt.legend(bbox_to_anchor=(0.5,-0.05),loc=0)
    #plt.legend(loc='best')
    return p_ret, p_sum, np.array(weights)
 

def align_data(df_assets, df_factors):
    
    # concatenate factors and assets data
    df_all = pd.concat([df_assets, df_factors], axis = 1)
    df_all.dropna(inplace = True)

    N_factors = df_factors.shape[1]
    N_assets = df_assets.shape[1]

    if (N_assets + N_factors) != df_all.shape[1]:
        print('error!')
    
    df_assets = df_all.iloc[:, :N_assets]
    df_factors = df_all.iloc[:, N_assets:]
    
    return df_assets, df_factors

# do regression
def factor_decomposition(df_assets, df_factors):
    df_pvalues = pd.DataFrame(index = df_factors.columns, columns = df_assets.columns)
    #df_betas =  pd.DataFrame(index = df_factors.columns, columns = df_assets.columns)
    df_betas = pd.DataFrame(columns = df_assets.columns)
    
    for asset in df_assets.columns:
        Y = df_assets[asset]
        X = df_factors
        X = sm.add_constant(X)

        model = sm.OLS(Y, X)
        results = model.fit()

        df_betas[asset] = results.params
        df_pvalues[asset] = results.pvalues
        df_pvalues.loc['R2(%)', asset] = results.rsquared * 100
    
    return df_betas.iloc[1:,:], df_betas.iloc[0,:], df_pvalues
    
    
    
def mean_variance_model_TC(asset_alpha, asset_cov, df_cost, w0, lam = 10000):
    N_asset = asset_cov.shape[0]
    w = Variable(N_asset)
    gamma = Parameter(sign = 'positive')
    ret = np.array(asset_alpha).T * w
    risk = quad_form(w, np.array(asset_cov))
    trading_cost = np.array(df_cost.loc['trading',:])*abs(w - np.array(w0))
    holding_cost = np.array(df_cost.loc['holding',:])*w / 12
    
    prob = Problem(Maximize(ret - gamma * risk - trading_cost - holding_cost),\
                   [sum_entries(w) == 1, w >= 0])
                   
    gamma.value = lam
    prob.solve()
    
    df_sol = pd.Series(data = np.array(w.value).flatten(), index = asset_alpha.index)
    return df_sol   
    
    
def portfolio_df(df_assets, df_weight, df_cost, df_RF):
    '''
    asset return for each period, and weights
    '''
    # transaction cost for each month
    # first line should be cash holding
    
    pf_value = pd.Series(index = df_weight.index)
    pf_value.iloc[0] = 1

    for i in np.arange(1, df_weight.shape[0]):
        month = df_weight.index[i]
      
        trading_cost = np.abs(df_weight.iloc[i,:] - df_weight.iloc[i-1,:]).dot(\
                                df_cost.loc['trading', :]) * pf_value.iloc[i-1]
        
        pf_value.iloc[i] = pf_value.iloc[i-1] * \
            df_weight.iloc[i,:].dot(1 + df_assets.loc[month,:] + df_RF.loc[month]\
                                    - df_cost.loc['holding',:]/12) - \
            trading_cost
    
    return pf_value
 

def pf_weights(alpha_,N_skip,df_assets,df_factors,df_cost,lbd):   
    df_weight = pd.DataFrame(data = np.zeros( df_assets.shape ), index = df_assets.index, columns = df_assets.columns)
    
    for month in df_assets.index:
        
        ind = np.where(df_assets.index == month)[0][0]
        if ind < N_skip:
            continue
        # prediction on current month
        # use previous data
        df_prev_factor = df_factors.iloc[:ind,:]
        df_prev_asset = df_assets.iloc[:ind, :]
        
        n = df_prev_factor.shape[0]
        weights = (1 - alpha_) * alpha_** (n - np.arange(1, n+1))
        weights = weights / np.sum(weights)*len(weights)
        
        df_prev_factor = df_prev_factor.multiply(weights, axis = 0)
        df_prev_asset = df_prev_asset.multiply(weights, axis = 0)
        
        
        # prediction of factor alpha
        factor_alpha = df_prev_factor.mean()
        
        # covariance matrix of factors
        factor_cov = df_prev_factor.cov()
        
        # beta
        betas, consts, pvalues = factor_decomposition(df_prev_asset, df_prev_factor)
        
        # covariance between assets
        asset_alpha = betas.transpose().dot(factor_alpha) + consts
        asset_cov = betas.transpose().dot(factor_cov).dot(betas)
        
        # optimization models
        #df_weight.loc[month,:] = 1/df_weight.shape[1]
        df_weight.loc[month, : ] = mean_variance_model_TC(asset_alpha, asset_cov, df_cost, \
                                                          df_weight.iloc[ind,:], lbd)
        
    df_weight = df_weight.iloc[N_skip-1:, :]   
    df_weight.iloc[0, :] = 0
    return df_weight
#==============================================================================
# TRADING/HOLDING COSTS
#==============================================================================
# please refer to page 5 of the project description
# the following are estimates 
trading_costs = np.array([0.0005, 0.0010, 0.0015, 0.0000, 0.0030, 0.0040, 0.0100,0.0100])
holding_costs = np.array([0.0000, 0.0010, 0.0005, 0.0000, 0.0015, 0.0025, 0.0000,0.0000])/12


#==============================================================================
# load data
#==============================================================================
data = pd.read_csv('./data/asset_return_rf.csv')
#data.drop(['PE'], axis=1, inplace=True)
col = data.columns
data['Date']=pd.to_datetime(data['Date'])
data.set_index(data['Date'],inplace=True)
del data['Date']

