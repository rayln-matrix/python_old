import pandas as pd

census_df = pd.read_csv('census.csv')
'''colums_to_keep=['SUMLEV','STNAME','CTYNAME','CENSUS2010POP']
df=census_df[colums_to_keep]
df=df[df['SUMLEV']==50]
df=df.set_index(['STNAME'])
#取出的Columns如果只有一個資料(通常是數字)則該Columns是以資料的型態傳送、處理，並不是用一個以上資料時的Series,例如下列例子：
#testn='District of Columbia'
#test=df.loc[testn]
#print(test)
#print(test['CENSUS2010POP'])
#print(test.dtype)
#print(test['CENSUS2010POP'].dtype)

def answer_six(dfd):
    
    dflt=dfd['CTYNAME']
    stnmlist=dfd.index.unique()
    #print(type(dflt))，type()輸出指定物件的資料型態，df.dtype()輸出的是dataframe內容物的資料型態
    stdic={}
    stndfd=pd.Series(stdic)
    print(stndfd)
    for stnd in stnmlist:
        sb=dfd.loc[stnd]
        ln=sb['CENSUS2010POP']
        #print(ln.dtype)
        #print(type(ln))
        if type(ln)==type(dflt):
            sb=sb.nlargest(3,'CENSUS2010POP',keep='first')
            pp=sb['CENSUS2010POP']
            ppn=pp.sum()
            sb['POPnumber']=ppn
            #print(ppn)
            #sb=sb.groupby(['POPnumber'])
            stndfd.loc[stnd]=ppn
          
    print(stndfd)
    best_three=stndfd.nlargest(3)
    print(best_three)
    return best_three.index

            
   
print(list(answer_six(df)))'''


'''census_df=census_df[census_df['SUMLEV']==50]
columns_to_keep=['CTYNAME','POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012',
                 'POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']
df=census_df[columns_to_keep]
#df=df[df['SUMLEV']==50]
df=df.set_index('CTYNAME')
#print(df.head(50))'''

'''def answer_seven(dfd):
    ctypopdic={}
    ctypopseries=pd.Series(ctypopdic)
    #dfd=dfd.drop(['SUMLEV'])
    test=dfd.loc['Young County']
    print(test)
    #print(type(test))
    test=test.reset_index()#   -->名字相同的BOUNTY多
    print(test)
    for i in test.index:
        testlst=test.iloc[i]
        #testlst=testlst.drop(['CTYNAME','SUMLEV'])
        print(testlst)
        #print(testlst['POPESTIMATE2013'])
        print(type(testlst))
        #print(testlst.max()-testlst.min())
        #mi=testlst.min()
        #dif=mx-mi
        #print(dif)
    #dfd=dfd.sort_index()
    for ctyn in dfd.index:
        eachcty=dfd.loc[ctyn]
        print(type(eachcty))
        #eachcty=eachcty.reset_index()
        for i in eachcty.index:
            poplst=eachcty.iloc[i]
            #print(poplst)
            #poplst=poplst.drop(['CTYNAME','SUMLEV'])
            #popdif=poplst.max()-poplst.min()
            #ctypopseries.loc[ctyn]=popdif

    #best=ctypopseries.idxmax()
        
        
    #best=ctypopseries.idxmax()
    #return best

print(answer_seven(df))'''

'''census_df=census_df[census_df['SUMLEV']==50]
columns_to_keep=['STNAME','CTYNAME','REGION','POPESTIMATE2014','POPESTIMATE2015']
df=census_df[columns_to_keep]
df=df[(df['REGION']==1) | (df['REGION']==2)]
df=df[df['POPESTIMATE2015']>df['POPESTIMATE2014']]
df=df[['STNAME','CTYNAME']]
#print(df)
#wst=df[df['CTYNAME'][:11]=='Washington']
#print(wst)

print(df.index)
#df['origin_index']=df.index
df=df.reset_index()
#c=df.iloc[0]
#print(c)

for i in df.index:
    wst=df.iloc[i]
    #print(wst)
    ctyname=wst['CTYNAME']
    #print(ctyname[:10]=='Washington')
    if ctyname[:10]=='Washington':
        newdfd=pd.DataFrame(wst.T)
        print(newdfd)

#def answer_eight(dfd):
    #return '''

census_df=census_df[census_df['SUMLEV']==50]
columns_to_keep=['STNAME','CTYNAME','REGION','POPESTIMATE2014','POPESTIMATE2015']
df=census_df[columns_to_keep]
df=df[(df['REGION']==1) | (df['REGION']==2)]
df=df[df['POPESTIMATE2015']>df['POPESTIMATE2014']]
df=df[['STNAME','CTYNAME']]
#print(df)
#wst=df[df['CTYNAME'][:11]=='Washington']
#print(wst)

#print(df.index)
#df['origin_index']=df.index
df=df.reset_index()
#c=df.iloc[0]
#print(c)
dfnew=pd.DataFrame()
for i in df.index:
    wst=df.iloc[i]
    #print(wst)
    ctyname=wst['CTYNAME']
    if ctyname[:10]=='Washington':
        indx=wst['index']
        stn=wst['STNAME']
        ctn=wst['CTYNAME']
        dfn=df[(df['STNAME']==stn) & (df['CTYNAME']==ctn)]
        dfnew=dfnew.append(dfn)
       

    
dfnew=dfnew.set_index('index')       
print(dfnew)
#print(dfnew['index'])
#print(type(dfnew['index']))
#print(df.head())
#print(df['STNAME'])
#print(dfnew['STNAME'])
#df=df[dfnew]
#df=df[(df['STNAME']==dfnew['STNAME']) & (df['CTYNAME']==dfnew['CTYNAME'])]
#print(df)


def answer_eight(dfd):
    return 

answer_eight(df)


