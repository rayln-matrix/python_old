import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


'''np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])

#df1=df.groupby(level=0)[df.columns].mean()
year_list=[y for y in range(1992,1996)]
mean_list=[]
std_list=[]
for year in range(1992,1996):
    mean_list.append(df.loc[year].mean())    

for year in range(1992,1996):
    std_list.append(df.loc[year].std())


print(mean_list)
print(std_list)
print(year_list)

#plt.figure()
bar1=plt.bar(year_list,mean_list,width=0.3,yerr=std_list)
ax=plt.gca()
plt.xticks([1992,1993,1994,1995],year_list)
#plt.show()
#fig,ax=plt.subplot(1,1,1)
#fig.plt.barchar


class colored_bar():
    
    def __init__(self,fig):
        self.fig=fig
        self.line=None
        self.ys = [bar.get_height() for bar in fig]
        self.press = self.fig.figure.canvas.mpl_connect('button_press_event', self)
        self.release= self.fig.figure.canvas.mpl_connect('button_release_event',self)
        
    
    def daw_line(self,event):
        if event ==self.release:
            y_value=event.ydata   
            plt.gca().set_title('The Y-value is {}'.format(y_value))
            self.line=plt.axhline(y=y_value, color='r', linestyle='-')
            plt.show()
            self.line=None
        
        if event == self.press:
            self.line.remove()
        
        
       

fig = plt.figure()
ax = fig.add_subplot(111)
rects = ax.bar(range(10), 20*np.random.rand(10))


#color_rects=colored_bar(bar1)
print(dir(bar1))
print(dir(fig))


'''

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])

#df1=df.groupby(level=0)[df.columns].mean()
year_list=[y for y in range(1992,1996)]
mean_list=[]
std_list=[]
for year in range(1992,1996):
    mean_list.append(df.loc[year].mean())    

for year in range(1992,1996):
    std_list.append(df.loc[year].std())

#print(len(df.columns))
st_er=std_list/np.sqrt(3650)
margin_err=st_er*1.96
#yerr = std_list / np.sqrt(3650) * stats.t.ppf(1-0.05/2, n - 1)
#print(margin_err)
#print(mean_list)
#print(std_list)
#print(year_list)

#def onrelease(event):
 #   plt.remove(line)
    


def onclick(event):
    #plt.cla()
    #plt.show()
    #plt.gca().set_title('Event at pixels {},{} \nand data {},{}'.format(event.x, event.y, event.xdata, event.ydata))
    y_value=event.ydata
    #x=event.xdata
    #print(event)
    plt.gca().set_title('The Y-value is {}'.format(y_value))
    plt.axhline(y=y_value, color='r', linestyle='-')
    plt.show()
    #plt.gcf().canvas.mpl_connect('button_press_event', onrelease)
    #rangelist=[]
    #for err in margin_err:
    #    if err < y_value:
    #        print(err)
    
    
#print(df.loc[1992])
#print(df.loc[1992].std())
#plt.figure()
#fig,bar=plt.subplot()
barchart=plt.bar(year_list,mean_list,yerr=margin_err,capsize=10)
ax=plt.gca()

print(len(barchart))
print(barchart[0].get_height())
for bar in barchart:
    print(bar.get_height())

plt.xticks([1992,1993,1994,1995],year_list)
plt.show()
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
