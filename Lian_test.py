import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

 
from portfolios import portfolio,  summary_stats


    
asset_file = 'data/asset_return.csv'
factor_file = 'data/With_Liquidity_Monthly_Factors_Log_Returns_Aggregated.csv'


asset_data = pd.read_csv(  asset_file, index_col= 0 )
asset_data.index = pd.to_datetime( asset_data.index)
asset_data.dropna(inplace=True)
asset_data.index = asset_data.index.map( lambda x: x.strftime('%Y-%m')  )



soimod =  lambda x: 1.0 /  (1 + np.exp(-x))

asset_data =  asset_data.applymap(soimod)


factor_data = pd.read_csv(factor_file, index_col= 0 )
factor_data.index = pd.to_datetime( factor_data.index)
factor_data.dropna(inplace=True)
factor_data.index = factor_data.index.map( lambda x: x.strftime('%Y-%m')  )


merged_data = pd.concat([asset_data, factor_data], axis=1, join='inner')


asset_class = asset_data.columns

factor_data.drop( ['RF'], axis=1, inplace=True )


factor_class = factor_data.columns

#factor_class =  [u'Momentum', u'BXM Level (Volatility)', u'Mkt-RF', u'SMB', u'HML',
       #u'Last_Price (High Yield Credit Spread)',
       #u'Credit Spread Investment Grade', u'Liquidity']

dic1 = {}
for i,col in enumerate( merged_data.columns):
    dic1[col]=i
    
    
asset_class_num = [ dic1[col] for col in asset_class]


factor_class_num =  [ dic1[col] for col in factor_class]




asset_class = asset_data.columns
factor_class = factor_data.columns


for i,col in enumerate(asset_class ):
    
    dic1[col]=i
    
    
    
    

def conduct_regression( data ):

    xs= sm.add_constant( data[ factor_class ] )
    
    coef_result =pd.DataFrame( columns= factor_class, index= asset_class )
    t_stat = pd.DataFrame( columns= factor_class, index= asset_class )
    
    for y in asset_class:
        model = sm.OLS( data[ y], xs)
        results = model.fit()     
        coef_result.loc[ y, factor_class] = results.params
        t_stat.loc[ y, factor_class] = results._results.tvalues[1:]
 
    return coef_result,  t_stat
        


conduct_regression( merged_data )


result = pd.DataFrame( columns= factor_class)
t_stat = pd.DataFrame( columns= factor_class)



for i in range(60,  len( merged_data.index)):
    
    data =  merged_data.loc[ merged_data.index[i-60:i], :    ]
    
    coef, t_s =  conduct_regression(data)
    
    coef['date'] =  merged_data.index[i]
    coef.reset_index(inplace=True)
    t_s.reset_index(inplace=True)
    
    result =  result.append( coef, ignore_index=True)
    t_stat =  t_stat.append( t_s, ignore_index=True)


result.to_csv( 'regression_coef.csv')
t_stat.to_csv('t_stats.csv')


#fac = [   u'Momentum', u'BXM Level (Volatility)', u'Mkt-RF', u'SMB', u'HML',
        #u'RF', u'Last_Price (High Yield Credit Spread)', u'PS_VWF (Liquidity)',
        #u'Credit Spread Investment Grade']

#fac = [   u'Momentum', u'BXM Level (Volatility)', u'Mkt-RF', u'SMB', u'HML',
         #u'Last_Price (High Yield Credit Spread)', u'PS_VWF (Liquidity)',
        #u'Credit Spread Investment Grade'] 

#result[   result['index'] == 'Equity_russel_3000'][fac].plot()



result[   result['index'] == 'Equity_russel_3000'].plot()
plt.show()


t_stat[ t_stat['index'] == 'Equity_russel_3000'].plot()
plt.show()


#merged_data.rolling( 1 ).apply( lambda x: conduct_regression(x) )



