import pandas as pd
import numpy as np

energy=pd.read_excel('Energy Indicators.xls')
e=energy.drop(energy.columns[[0,1]],axis=1).rename(columns={'Environmental Indicators: Energy':'Country','Unnamed: 3':'Energy Supply','Unnamed: 4':'Energy Supply per Capita','Unnamed: 5':'% Renewable'})
e=e.iloc[16:243].reset_index().drop(['index'],axis=1)
e=e.replace({'...':np.NaN})
e['Energy Supply']=e['Energy Supply']*1000000
e=e.replace({"United Kingdom of Great Britain and Northern Ireland19": "United Kingdom",
             "United States of America20": "United States",
             "Republic of Korea": "South Korea",
             "China, Hong Kong Special Administrative Region3": "Hong Kong"})
s={'0','1','2','3','4','5','6','7','8','9'}

for ctryname in e['Country']:
    if ctryname[-1:] in s :
        if ctryname[-2] in s:
            ctryname1=ctryname[:-2]
        else:    
            ctryname1=ctryname[:-1]
        e=e.replace({ctryname:ctryname1})       


for ctryname in e['Country']:
    if ctryname[-1:]==')':
        #print(ctryname)
        strindx=ctryname.find('(')
        ctryname2=ctryname[:strindx-1]    
        e=e.replace({ctryname:ctryname2})

e=e.set_index(['Country'])
#print(e)
        
GDP=pd.read_csv('world_bank.csv',skiprows=4)
GDP=GDP.replace({"Korea, Rep.": "South Korea", 
"Iran, Islamic Rep.": "Iran",
"Hong Kong SAR, China": "Hong Kong"})
GDP=GDP.rename(columns={'Country Name':'Country'})
GDPX=GDP.set_index(['Country'])

ScimEn=pd.read_excel('scimagojr-3.xlsx')
Sci=ScimEn.set_index(['Country'])
Sci=Sci[Sci['Rank']<16]
GDPX=GDPX[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]

Join=pd.merge(Sci,e,how='left',left_index=True, right_index=True)
answer=pd.merge(Join,GDPX, how='left',left_index=True,right_index=True)
print(answer)
