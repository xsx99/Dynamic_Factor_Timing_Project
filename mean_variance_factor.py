# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:06:45 2017

@author: Shuxin Xu
"""

import pandas as pd 
import numpy as np
import portfolios as pf
import imp
pf=imp.reload(pf)

root_dir = 'C:\Users\Shuxin Xu\Dropbox (Personal)\Presentation on Feb 16\Dynamic_Factor_Timing_Project_YMA'
# read factor data
df_factors = pd.read_csv(root_dir + '/data' + \
                        '/Monthly_Factors_No_Liq.csv')

df_factors.set_index('Date', inplace = True)
df_factors.index = pd.to_datetime(df_factors.index)
df_factors.index = df_factors.index.map(lambda x:x.strftime('%Y-%m'))

# read the assets
df_assets  = pd.read_csv(root_dir + '/data/asset_return.csv')
df_assets = df_assets.rename(columns = {df_assets.columns[0]:'Date'})
df_assets.set_index('Date', inplace = True)
df_assets.sort_index(inplace = True)
df_assets, df_factors =pf.align_data(df_assets, df_factors)
df_assets = df_assets.subtract(df_factors['RF'], axis = 0)

# read trading cost
df_cost = pd.read_csv(root_dir + '/data/asset_cost.csv', index_col = 0)

#factor cleaning
del df_factors['BXM Volatility']
df_RF = df_factors['RF']
del df_factors['RF']

# use all data
betas, consts, pvalues = pf.factor_decomposition(df_assets, df_factors)

# covariance prediction
alpha= 0.85
df_weight=pf.pf_weights(alpha,12,df_assets,df_factors,df_cost,10)
pf = pf.portfolio_df(df_assets.astype(float), df_weight.astype(float), df_cost, df_RF)


