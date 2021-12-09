import numpy as np
from matplotlib import pyplot as plt
#from scipy.integrate import quad


t= np.arange(0,15,0.0001)
## s = a1 t**3 + a2 t**2 + a3t + a4: 3次多項式，可以用更多次方，會得到不同的結果
## condition: s(0)=0, s(15)=415.75, v(t)=s'(t), v(0)=v(15)=0 --> 用這些邊界條件來找多項式係數
a1 = -415.75*2/(15**3)
a2 = 3*415.75/15**2
s = a1*(t**3) +a2*(t**2) 
v = 3*a1*(t**2) + 2*a2*t
a = 6*a1*t +2*a2
ax1 = plt.subplot(311)
ax1.set_title("S")
ax1.set_ylim([0,415.75])
plt.plot(t,s)
ax2 = plt.subplot(312,sharex=ax1)
ax2.set_title("V")
plt.plot(t,v)
ax3 = plt.subplot(313,sharex=ax1)
ax3.set_title("a")
plt.plot(t,a)
plt.subplots_adjust(hspace=0.5)
plt.show()