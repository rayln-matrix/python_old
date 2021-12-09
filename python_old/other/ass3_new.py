import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])

year_list=[y for y in range(1992,1996)]
mean_list=[]
std_list=[]
for year in range(1992,1996):
    mean_list.append(df.loc[year].mean())    

for year in range(1992,1996):
    std_list.append(df.loc[year].std())

st_er=std_list/np.sqrt(3650)
margin_err=st_er*1.96

class plot_bar:
    
    def __init__(self, mean_list, year_list,margin_err):
        self.mean=mean_list
        self.years=year_list
        self.y_err=margin_err
     

    def plot_fig(self):
        print("The min of means is %f "%min(self.mean))
        print("The max of means is %f "%max(self.mean))
        y_entered=float(input('Enter a Y-value:'))
        margin_len=len(self.y_err)
        margin_half=self.y_err/2
        bar_color=['b','b','b','b']
        for m in range(margin_len):
            if  y_entered >= self.mean[m]+margin_half[m]:
                bar_color[m]='r'
            if self.mean[m]-margin_half[m] <= y_entered <= self.mean[m]+margin_half[m]:
                bar_color[m]='w'
            if  y_entered <=self.mean[m]-margin_half[m]:
                bar_color[m]='b'
        
        label_list=['above','above','above','above']
        for i in range(len(bar_color)):
            if bar_color[i]=='r':
                label_list[i]='above'
            if bar_color[i]=='w':
                label_list[i]='within'
            if bar_color[i]=='b':
                label_list[i]='below'

            
        fig,ax=plt.subplots()
        ax.bar(self.years, self.mean, yerr=self.y_err,capsize=10,color=bar_color,edgecolor=['gray','gray','gray','gray'])
        plt.xticks([1992,1993,1994,1995],year_list)
        plt.gca().set_title('The Y-value is {}'.format(y_entered))
        plt.axhline(y=y_entered, color='r', linestyle='-')
        #plt.legend(handles=bar_color,labels=label_list)
        plt.show()


bar1=plot_bar(mean_list,year_list,margin_err)
bar1.plot_fig()



#bar1.plot_fig()
#print(bar1.mean)
#print(margin_err)
#print(bar1.height)
#change_color(bar1).connect()
'''class change_color:
     
    def __init__(self,maindata):
        self.maindata=maindata
        self.mainfig=maindata.plot_fig()
        self.line=None
        #self.press=plt.gcf().mpl_connect('button_press_event')
        #self.releas=plt.gcf().mpl_connect('button_release_event')

    def connect(self):
        'connect to all the events we need'
        self.press = self.mainfig.rect.figure.canvas.mpl_connect('button_press_event', self.y_line)
        #self.release = self.mainfig.figure.canvas.mpl_connect('button_release_event', self.on_release)
        #self.motion = self.mainfig.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
    
    
    def y_line(self,event):
        while self.line != None:
            y_value=event.ydata
            #x=event.xdata
            #print(event)
            plt.gca().set_title('The Y-value is {}'.format(y_value))
            self.line=plt.axhline(y=y_value, color='r', linestyle='-')
            plt.show(self.line)

    #def colored(self,event):'''

        
