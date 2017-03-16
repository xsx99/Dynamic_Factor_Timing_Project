# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 12:08:00 2017

@author: Shuxin Xu
"""
import portfolios as pf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imp
pf=imp.reload(pf)
                        
data=pf.data
rf=data['RF'][13:]/100
if 'RF' in data.columns:
    del data['RF']
 
##simple benchmark  
 
# UCRP
p_ucrp, p_ucrp_sum = pf.portfolio(data.iloc[12:,:],np.array([[0.25,0.25,0.13,0.02,0.025,0.07,0.1,0.1]]*len(data)),'UCRP')
# 60/40
p_6040, p_6040_sum = pf.portfolio(data.iloc[12:,:],np.array([[0.6/2, 0.6/2, 0.4/3, 0.4/3, 0.4/3, 0, 0, 0]]*len(data)),'60/40')
# Equally Weighted 
p_eq, p_eq_sum = pf.portfolio(data.iloc[12:,:],np.array([[1.00/8.00]*8]*len(data)),'equally weighted')


s1=pf.summary_stats(p_ucrp,p_ucrp_sum,p_ucrp,rf)
s1['method']='ucrp'
s2=pf.summary_stats(p_6040,p_6040_sum,p_ucrp,rf)
s2['method']='6040'
s3=pf.summary_stats(p_eq,p_eq_sum,p_ucrp,rf)
s3['method']='equally weighted'



#naive risk parity
s=pd.DataFrame()
fig=plt.figure()
p_rp, p_rp_sum = pf.portfolio(data.iloc[12:,:],pf.risk_parity(),'naive risk parity')
s1=pf.summary_stats(p_rp,p_rp_sum,p_ucrp,rf)
s1['alpha']='NA'
s1['method']='naive risk parity'
s=pd.concat([s,s1],axis=0)

alpha_lst=[0.5,0.85,0.94]
for alpha in alpha_lst:    
    p_erp, p_erp_sum = pf.portfolio(data.iloc[12:,:],pf.risk_parity2(alpha),'risk parity alpha='+str(alpha))
    s1=pf.summary_stats(p_erp,p_erp_sum,p_ucrp,rf)
    s1['alpha']=alpha
    s1['method']='exponential risk parity'
    s=pd.concat([s,s1],axis=0)
    
    
# simple mean variance
fig=plt.figure()
risk_aversion=[1,2,5,10,20,50,100]
for r in risk_aversion:
    p1_ret,p1,weights1=pf.mv_portfolio(data,'mean_var risk:{risk}'.format(risk=r),r)
    plt.plot(data.iloc[11:,:].index,p1,label='mean var risk='+str(r))
plt.legend(bbox_to_anchor=(0.5,-0.05),loc=0)


# exponential mean variance
fig=plt.figure()
risk_aversion=[1,2,5,10,20,50,100]
alpha=0.85
for r in risk_aversion:
    p1_ret,p1,weights1=pf.mv_portfolio_ew(data,'mean var',r,alpha)
    plt.plot(data.iloc[11:,:].index,p1,label='mean var risk='+str(r)+' alpha='+str(alpha))
plt.legend(bbox_to_anchor=(0.5,-0.05),loc=0)



#plot asset return
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
#plt.savefig(data.columns[i]+' weights')



