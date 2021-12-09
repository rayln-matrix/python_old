#import csv
 #precision 2
#with open("mpg.csv") as csvfile:
   #mpg=list(csv.DictReader(csvfile))

#print(mpg)


#numpy notes

# x.dot(Y)
# eye(n)--> I matrix
#.arange(start, end, interval)
#.diag-->對角線
#.linspace(start, end, how many number)
#.ones()
#.zeros()
#.repeat([],number)--->依照指定個數複製指定向量中每個元素並放到新的陣列中
#.array([]*3)-->整個陣列重複3次
#.vstack()/.hstack()
#.atype()---轉換資料型態
#.max()/.min()/.sum/.mean()/.std()/.argmax()--maxium position
#切割原本array後得到新的array，如果對新array做修改會影響到舊的---除非用.copy()
#iterate in two array-->.zip  : for i, j in zip(x,y)
#lambda:return function reference

#--------------------------------

#pandas:
#(1)series: list-->series，string=object, int=int
#nan-->not a number:數字List中的None轉換成Series後自動轉換成的資料(格式:float)，String中的None則被轉成object
#.index:列出series中的index
#pd.series([],index:[])
#pd.iloc[]/pd.loc[]-->用index位子找item/index值找item，這兩個不是Method 是 attributes(?)
#a.head()--??
#a method of computatiom: vactorization
#(%%)timeit n= 100 ---->計算迴圈次數與運算數度 (Default=1000/ 可以指定n)
#.set_value(label,value)
#.append() : 修改物件衍伸的資料卻不會修改物件本身
#可以直接對Series運算, ex: data=data*2, data=data==2 (輸出True or False, dtype: bool)
#數字運算：.sum()/.max()/.prod()/ .mean()/.median()/.std()/.nlargest(n)取前n大的數字/.nsmallest(n)
#字串運算：data.str.upper()/.lower()/.len()-->把字串內容轉換為其長度/.cat(sep='')-->把字串串起來，中間用""連接/.contains('P')-->判斷每個字串是否有此字串，輸出布林/.replace("b","a")
#Series是一維資料--資料-索引

#(2)DataFrame 
#.size: 數量/ .shape: 行列數
#cleaning tasks?
#.DataFrame(字典,index=[]):輸入的配對為欄位，可以用index自訂索引
#print(,sep="\n")-->分隔
#.DataFrame([],index=[])--->與numpy的array類似
#可以用.loc(row index, column index)->根據索引，注意.iloc/.loc()輸出的格式:輸出類似字典格式(key:value)-->根據序列
#可以直接用值呼叫對應的欄: ex: data[column]
#取得特定欄或列--->變成Series,可用Series的運算
#data["new column"]-->直接建立新欄/ 或是用正是語法:pd.series()
#可以直接對欄為運算(對Series)後指派給新的欄位： data["new"]=data["n"]*data['m']
#.T--->transpose
#chaining: .log[][]--->python have to coppy the data, may cost problems
#.drop()--->return a copy of data which drop the specific data(rows )
#和Series一樣，要增加新的資料可以直接用Data[]
#可以直接對欄位進行運算: df['Cost'] *= 0.8
#-->Series到DataFrame的index是怎麼分布的??
#index_col(?)--設定讀取檔案或資料時轉成DataFrame的Index的欄位(將指定的欄轉成index)，
# skiprows=n -->Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.
#Boolean Masking: a good querying  --用.where(條件) 來mask
#.dropna()-->去掉NAN資料 Default=0--row
#Boolean mask可以做多重邏輯條件--&,|  
#df['Name'][df['Cost']>3]
#.set_index()/.reset_index()---?
#df['country'] = df.index---->為何能保存? 
#df = df.set_index('Gold')
#df.head()
#


#2020/03/17 (week3)
#.merge
#pd.merge(dataframe1,dataframe2,how='inner/outer/right/left',left_index=/left_on=,right_index=/right_on=)
#  -->inner:交集，outer:聯集,right/left-->將資料併入左邊或右邊的dataFrame
# -->left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
#--->right: use only keys from right frame, similar to a SQL right outer join; preserve key order.
#  -->left/right_on: 用column 併入

##########################################

#小結：
# 1.python中String是object, 且不可直接更改單一內容(測試)
# 2. Series是一維資料，但也有index(會被一起列出)，index和Series都可以用list表示或迴圈(iterate)，可以指定index(用list)
# 3. Dataframe是二維資料，用字典形式的方式指派，每一個Key:value是一個欄，value和key都可以用list/tuple(??)表示。
# 4. Dataframe也有Index 可以用各種方式指派index也能用巢狀方式指派，可以將整欄指派為新的Index，然後用index取出資料

##########################################


#idiom

import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

print(df)
print(df.drop(df[df["Item Purchased"=="Dog Food"].index]))



df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)


names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()