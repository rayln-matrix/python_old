import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt



#Read data file
plt.style.use('seaborn-colorblind')
employ_data=pd.read_csv('aat1_csv.csv')
home_price_data=pd.read_csv('real-home-price-index_csv.csv')
cpi_data=pd.read_csv('consumer-price-index_csv.csv')
sp_hist=pd.read_csv('data_csv.csv')
bond_yeild_data=pd.read_csv('monthly_csv.csv')
gold_price_data=pd.read_csv('annual_csv.csv')
gdp_data=pd.read_csv('year_csv.csv')

#Make all data set within the same date range and apply the average of each year
partial_df=employ_data[employ_data['year']==1947].groupby('year')[employ_data.columns].mean()
employ_data=employ_data[employ_data['year'] !=1947]
employ_data=employ_data.append(partial_df).sort_values(by='year').reset_index()
home_price_af1940=home_price_data[(home_price_data['Date']>=1941.0) & (home_price_data['Date'] <= 2011.0)]
cpi_af1940=cpi_data[(cpi_data['Date']>=1941.0) & (cpi_data['Date']<= 2011.0)]

#Conusmer Price Index and real home price index
def average_year(data_df,data_name): 
    for year in data_df['Date']:
        indx=data_df[data_df['Date']==year].index.values
        data_df.loc[indx,'Year']=int(year)
    data_group=data_df.groupby(['Year'])[data_name].mean()
    data_group.index=data_group.index.astype(int)
    return data_group 
cpi_avg=average_year(cpi_af1940,'Consumer Price Index')
home_avg=average_year(home_price_af1940,'Real Home Price Index')

# employed and unemployed percent
year_df=employ_data['year'].astype(int)
employ_rate_df=employ_data['employed_percent']
unemployed_rate_df=employ_data['unemployed_percent']
unem_em=unemployed_rate_df/employ_rate_df

#S&P500
for year in sp_hist['Date']:
    indx=sp_hist[sp_hist['Date']==year].index.values
    sp_hist.loc[indx,'Year']=year[:4]
    sp_group=sp_hist.groupby(['Year'])['Real Price'].mean()
sp_group.index=sp_group.index.astype(int)
sp_group_af1940=sp_group[(sp_group.index >=1941) & (sp_group.index <= 2010)]

#10 year goverment bond,the data starts at 1953
for date in bond_yeild_data['Date']:
    indx=bond_yeild_data[bond_yeild_data['Date']==date].index.values
    bond_yeild_data.loc[indx,'Year']=date[:4]
    bond_year_avg=bond_yeild_data.groupby(['Year'])['Rate'].mean()
bond_year_avg.index=bond_year_avg.index.astype(int)
bond_year_avg_af1940=bond_year_avg[(bond_year_avg.index >= 1941) & (bond_year_avg.index <= 2010)]
bond_year_df=year_df[year_df >=1953]


#Gold_price:--->not plotted 
gold_data_year=[y for y in range(1950,2011)]
gold_price_af1950=gold_price_data['Price'][:61]


#GDP:
gdp_af1940=gdp_data[(gdp_data['date'] >= 1941) & (gdp_data['date'] <= 2010)]
gdp_time=gdp_af1940['date']
gdp_change_current=gdp_af1940['change-current']
gdp_change_chained=gdp_af1940['change-chained']

fig=plt.figure()
fig.suptitle('US economics after 1930 Great Depression',fontsize=36)
plt.subplot(221)
plt.plot(year_df,unemployed_rate_df,color='r',label='unemployed percent')
plt.axvline(x=1980,color='k',label='Early 1980s recession')
plt.axvline(x=1990,color='k',label='Early 1990s recession')
plt.axvline(x=2000,color='k',label='2000 Dot-com bubble')
plt.axvline(x=2008,color='k',label='2008 Great Recession')
plt.legend()

plt.subplot(222)
plt.plot(year_df,cpi_avg,label='Consumer Price Index')
plt.plot(year_df,home_avg,label='Real Home Price Index')
plt.axvline(x=1980,color='k',label='Early 1980s recession')
plt.axvline(x=1990,color='k',label='Early 1990s recession')
plt.axvline(x=2000,color='k',label='2000 Dot-com bubble')
plt.axvline(x=2008,color='k',label='2008 Great Recession')
plt.legend()

plt.subplot(223)
plt.plot(year_df,sp_group_af1940,color='b',label='S&P500')
plt.axvline(x=1980,color='k',label='Early 1980s recession')
plt.axvline(x=1990,color='k',label='Early 1990s recession')
plt.axvline(x=2000,color='k',label='2000 Dot-com bubble')
plt.axvline(x=2008,color='k',label='2008 Great Recession')
plt.legend()

plt.subplot(224)
#plt.plot(gold_data_year,gold_price_af1950,color='y',label='Gold price')
#plt.plot(gdp_time,gdp_change_current,color='g',label='GDP change current')
plt.plot(gdp_time,gdp_change_chained,color='y',label='GDP percent change')
plt.axvline(x=1980,color='k',label='Early 1980s recession')
plt.axvline(x=1990,color='k',label='Early 1990s recession')
plt.axvline(x=2000,color='k',label='2000 Dot-com bubble')
plt.axvline(x=2008,color='k',label='2008 Great Recession')
plt.legend()


plt.show()