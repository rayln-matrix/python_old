import pandas as pd 
import numpy as np 
import seaborn as sbn 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation


class_type=['T1','T2','T3']
class_type_list=np.random.choice(class_type,730)
#print(class_type_list)
test_data=pd.DataFrame({'A':np.random.normal(0,12,730),'B':np.random.normal(12,12,730),'C':np.random.normal(24,12,730)},
                      index=pd.date_range('1/1/2017', periods=730))
#print(test_data)

test_data['Name']=class_type_list
print(test_data)
#test_data.plot('A','B',c='B' ,kind='scatter',colormap='viridis')
#test_data.plot.kde()
''' kind argument:
    'line' : line plot (default)
    'bar' : vertical bar plot
    'barh' : horizontal bar plot
    'hist' : histogram
    'box' : boxplot
    'kde' : Kernel Density Estimation plot
    'density' : same as 'kde'
    'area' : area plot
    'pie' : pie plot
    'scatter' : scatter plot
    'hexbin' : hexbin plot
'''
#plt.show()

def growth(frame):
    
    pd.plotting.parallel_coordinates(test_data[:frame],'Name')

    
'''
fig=plt.figure()
a =animation.FuncAnimation(fig, growth, interval=10)
plt.show()'''



np.random.seed(1234)

v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')
plt.figure()
plt.hist([v1, v2], histtype='barstacked') #histtype, normed
v3 = np.concatenate((v1,v2))
sbn.kdeplot(v3)
#plt.show()
sbn.jointplot(test_data['A'],test_data['B'])

plt.show()
