# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 12:08:00 2017

@author: Shuxin Xu
"""
import portfolios as pf
import matplotlib.pyplot as plt
import numpy as np
import imp
pf=imp.reload(pf)

plt.figure(figsize=(10,5))                         
data=pf.data
if 'RF' in data.columns:
    del data['RF']
      
# UCRP
#p_ucrp, p_ucrp_sum = pf.portfolio(data.iloc[12:,:],np.array([[0.25,0.25,0.13,0.02,0.025,0.07,0.1,0.1]]*len(data)),'UCRP')
# 60/40
#p_6040, p_6040_sum = pf.portfolio(data.iloc[12:,:],np.array([[0.6/2, 0.6/2, 0.4/3, 0.4/3, 0.4/3, 0, 0, 0]]*len(data)),'60/40')
# Equally Weighted 
p_eq, p_eq_sum = pf.portfolio(data.iloc[11:,:],np.array([[1.00/8.00]*8]*len(data)),'equally weighted')

##equal weight
#pf.portfolio(data.iloc[12:,:],np.array([[1,1,1,1,1,1,1,1]]*len(data)),'Equal Weights')
# risk parity
#p_rp, p_rp_sum = pf.portfolio(data.iloc[12:,:],pf.risk_parity2(0.94),'ewm risk parity alpha=0.94')
# mean variance
p1_ret,p1,weights1=pf.mv_portfolio(data,'mean_var 1000',1000)
p2_ret,p2,weights2=pf.mv_portfolio(data,'mean_var 10',10)

# mean variance with exponential weighted return and covariance 
p3_ret, p3,weights3=pf.mv_portfolio_ew(data,'mean_var 1000 alpha=0.94',1000,0.94)
p4_ret,p4,weights4=pf.mv_portfolio_ew(data,'mean_var 1000 alpha=0.5',1000,0.5)

plt.title('Portfolios')
plt.ylabel('Return')
plt.show()

s1=pf.summary_stats(p1_ret,p1,p_eq)
s2=pf.summary_stats(p2_ret,p2,p_eq)
s3=pf.summary_stats(p3_ret,p3,p_eq)
s4=pf.summary_stats(p4_ret,p4,p_eq)

import pandas as pd
s=pd.concat([s1,s2,s3,s4],axis=0)


###plot asset return
fig=plt.figure()
for i in range(data.shape[1]):
    plt.plot(data.index,np.cumprod(1+data.iloc[:,i]),label=data.columns[i])
plt.legend(fontsize=5,loc=0)
plt.savefig('asset return',dpi=200)


#plot mean variance weights
for i in range(data.shape[1]):
    fig=plt.figure()
    plt.plot(data.index[11:],weights1[:,i])
    plt.legend([data.columns[i]])
    plt.show()
    fig.clear()
    plt.savefig(data.columns[i]+' weights')

