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
p_eq, p_eq_sum = pf.portfolio(data.iloc[12:,:],np.array([[1/8]*8]*len(data)),'equally weighted')
##equal weight
#pf.portfolio(data.iloc[12:,:],np.array([[1,1,1,1,1,1,1,1]]*len(data)),'Equal Weights')
# risk parity
#p_rp, p_rp_sum = pf.portfolio(data.iloc[12:,:],pf.risk_parity2(0.94),'ewm risk parity alpha=0.94')
# mean variance
p1,weights1=pf.mv_portfolio(data,'mean_var 1000',1000)
p2,weights2=pf.mv_portfolio(data,'mean_var 10',10)
# mean variance with exponential weighted return and covariance 
p3,weights3=pf.mv_portfolio_ew(data,'mean_var 1000 alpha=0.94',1000,0.94)
p3,weights3=pf.mv_portfolio_ew(data,'mean_var 1000 alpha=0.5',1000,0.5)
plt.title('Basic Portfolios')
plt.ylabel('Return')




###plot asset return
fig=plt.figure()
for i in range(data.shape[1]):
    plt.plot(np.cumprod(1+data.iloc[:,i]))
plt.legend(data.columns,fontsize=5,loc=0)
#plt.savefig('asset return',dpi=200)


#plot mean variance weights
for i in range(data.shape[1]):
    fig=plt.figure()
    plt.plot(data.index[11:],weights1[:,i])
    plt.legend([data.columns[i]])
    plt.show()
    fig.clear()


