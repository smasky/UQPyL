from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import MultipleLocator
y_major_locator=MultipleLocator(0.1)
import os
os.chdir('./Test')
#plt.rcParams['font.sans-serif'] = "arial"
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams['axes.linewidth'] = 2
data=np.loadtxt('./RBF_count.txt')
data=data[:14,:].transpose()/80

# pd_data=DataFrame(columns=["Value","Type","Problem"])
# #pd_data=DataFrame(index=["ZDT1","ZDT3","ZDT6","DTLZ2","DTLZ5","DTLZ7"],columns=["NSGA-II","NSGA-III","MOEA/D","RVEA"],data=data)
M=data.shape[0]
N=data.shape[1]
#temp=data[:,0].copy()
#data[:,0]=data[:,-2].copy()+4
#data[:,-2]=temp
Problem={1:"ZDT1",2:"ZDT3",3:"ZDT6",4:"DTLZ2",5:"DTLZ5",6:"DTLZ7"}
# Al={1:"NSGA-II",2:"NSGA-III",3:"MOEA/D",4:"RVEA"}
Al={1:"Origin", 2:"Ridge", 3:"Lasso"}
labels=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14"]


'''
for i in range(0,M):
    for j in range(0,N):
        value=data[i,j]
        p=Problem[i+1]
        a=Al[j+1]
        new_data=pd.DataFrame({"Value":value,"Type":a,"Problem":p},index=[1])
        pd_data=pd_data.append(new_data,ignore_index=True)
num=0
for i in range(0,M):
    for j in range(0,N-1):
        if j==0:
            t="CB"
        elif j==1:
            t="LN"
        elif j==2:
            t="TPS"
        elif j==3:
            t="GS"
        elif j==4:
            t="MQ"
        num+=1
        if int(data[i,N-1])==1:
            #O=r'$1^{th}\;Objective$'
            O="1th"
        elif int(data[i,N-1])==2:
            #O=r'$2^{nd}\,Objective$'
            O="2nd"
        else:
            O="3rd"
        
        new_data=pd.DataFrame({"number":data[i,j],"types":t,"obj":O},index=[num])
        pd_data=pd_data.append(new_data,ignore_index=True)
'''

        
fig=plt.figure(dpi=300,figsize=(14,6))
#print(pd_data)
a=sns.color_palette("Set3")
colors=[a[0]]+a[2:5]+a[6:7] #'CB', 'MQ', 'LN', 'TPS', 'GAS'
ax=plt.bar(labels,data[0,:],linewidth=2,edgecolor='.2',label="CB",color=colors[0],width=0.5)
ax=plt.bar(labels,data[1,:],linewidth=2,edgecolor='.2',bottom=data[0,:],label="MQ",color=colors[1],width=0.5)
ax=plt.bar(labels,data[2,:],linewidth=2,edgecolor='.2',bottom=data[1,:]+data[0,:],label="LN",color=colors[2],width=0.5)
ax=plt.bar(labels,data[3,:],linewidth=2,edgecolor='.2',bottom=data[2,:]+data[1,:]+data[0,:],label="TPS",color=colors[3],width=0.5)
ax=plt.bar(labels,data[4,:],linewidth=2,edgecolor='.2',bottom=data[3,:]+data[2,:]+data[1,:]+data[0,:],label="GAS",color=colors[4],width=0.5)
#ax=sns.barplot(x="Problem", y="Value", hue="Type", linewidth=2,edgecolor='.2',data=pd_data,capsize=.05,palette="Set3")
data2=data.copy()
data1=data.copy()
data1[0,:]=data2[0,:]/2
data1[1,:]=data2[1,:]/2+data2[0,:]
data1[2,:]=data2[2,:]/2+data2[0,:]+data2[1,:]
data1[3,:]=data2[3,:]/2+data2[2,:]+data2[0,:]+data2[1,:]
data1[4,:]=data2[4,:]/2+data2[3,:]+data2[2,:]+data2[0,:]+data2[1,:]
# data1[3,:]=data2[3,:]/2+data2[0,:]+data2[1,:]+data2[2,:]
data1=data1-0.02
for i in range(M):
    for j in range(N):
        if data[i,j]>0.03:
            plt.text(j,data1[i,j],'%.1f'%(data[i,j]*100)+'%', ha="center",va="bottom")



plt.gca().legend().set_title('')
#ax.set_xlabel("Objectives",fontsize=15)
#ax.set_ylabel("Counts",fontsize=15)
#ax.set_xticklabels(["1th","2nd","3rd"],fontsize=15)
#ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)
#ax.tick_params(labelsize=15,width=2,length=5)
#ax.set_yticklabels(range(0, 22, 2), fontsize = 15)
plt.ylabel("Percentage",fontsize=20)
plt.xlabel("Cases",fontsize=20)
plt.tight_layout()
plt.xticks(labels,fontsize=20)
plt.yticks(np.linspace(0,1,11), fontsize = 20)
plt.legend(fontsize=20, loc="lower center", frameon=1,bbox_to_anchor=(0.5,-0.22),ncol=5)
'''
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_visible(True)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_visible(True)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_visible(True)
ax.spines['right'].set_linewidth(2)
'''
fig.savefig("{}.svg".format("DTLZ6_B_M_t"),bbox_inches='tight')
fig.savefig("{}.eps".format("DTLZ6_B_M_t"),bbox_inches='tight')
fig.savefig("{}.pdf".format("DTLZ6_B_M_t"),bbox_inches='tight')