import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
#from scipy.stats import norm
#from scipy.stats import chi2, chisquare
import pandas as pd 


###########  R script for Chapter 2   ####################################################
###########  of Statistics and Data Analysis for Financial Engineering, 2nd Edition ######
###########  by Ruppert and Matteson  ####################################################

################################################################
########## Code for figure 2.1  ################################
################################################################
plt.figure()
x = np.linspace(-0.25,0.25,num=200)
plt.plot(x,np.log(x+1),'k-',linewidth=2,label="log(x+1)")  
plt.plot(x,x,'r--', linewidth=2, label="x")  
plt.legend()


################################################################
########## Code for figure 2.2  ################################
################################################################
plt.figure()
x=np.linspace(0,10,num=300)
plt.plot(x,x/2,'k-',linewidth=2,label='mean')
plt.plot(x,x/2+np.sqrt(x),'k--',linewidth=2,label='mean + SD')
plt.plot(x,x/2-np.sqrt(x),'k--',linewidth=2,label='mean - SD')
plt.xlabel('time')



dat = pd.read_csv("./../datasets/Stock_bond.csv")
plt.figure()
plt.plot(dat.GM_AC)
plt.xlabel('Index')
plt.ylabel('GM_AC')

plt.figure()
plt.plot(dat.F_AC)
plt.xlabel('Index')
plt.ylabel('F_AC')

plt.figure()
n=len(dat.GM_AC)
GMReturn = dat.GM_AC.values[1:]/dat.GM_AC.values[:-1] - 1.0
FReturn = dat.F_AC.values[1:]/dat.F_AC.values[:-1] - 1.0 
plt.plot(GMReturn,FReturn,'o')
plt.xlabel('GMReturn')
plt.ylabel('FReturn')


#########  problem 3  ##########
plt.figure()
MSFTReturn = dat.MSFT_AC.values[1:]/dat.MSFT_AC.values[:-1] - 1
MRKReturn = dat.MRK_AC.values[1:]/dat.MRK_AC.values[:-1] - 1
plt.plot(MSFTReturn,MRKReturn,'o')
plt.xlabel('MSFTReturn')
plt.ylabel('MRKReturn')

#def cor(x, y):
#    std_x = (x - x.mean())/x.std(ddof = 0)
#    std_y = (y - y.mean())/y.std(ddof = 0)
#    return (std_x * std_y).mean()
#
#def average(x):
#    assert len(x) > 0
#    return float(sum(x)) / len(x)
#
#def pearson_def(x, y):
#    assert len(x) == len(y)
#    n = len(x)
#    assert n > 0
#    avg_x = average(x)
#    avg_y = average(y)
#    diffprod = 0
#    xdiff2 = 0
#    ydiff2 = 0
#    for idx in range(n):
#        xdiff = x[idx] - avg_x
#        ydiff = y[idx] - avg_y
#        diffprod += xdiff * ydiff
#        xdiff2 += xdiff * xdiff
#        ydiff2 += ydiff * ydiff
#
#    return diffprod / np.sqrt(xdiff2 * ydiff2)

print(sst.pearsonr(MSFTReturn,MRKReturn)[0])
#print(pearson_def(MSFTReturn,MRKReturn))
