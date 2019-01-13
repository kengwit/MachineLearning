import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
#from scipy.stats import norm
#from scipy.stats import chi2, chisquare
import pandas as pd 
from IPython.display import display

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
plt.show()

################################################################
########## Code for figure 2.2  ################################
################################################################
plt.figure()
x=np.linspace(0,10,num=300)
plt.plot(x,x/2,'k-',linewidth=2,label='mean')
plt.plot(x,x/2+np.sqrt(x),'k--',linewidth=2,label='mean + SD')
plt.plot(x,x/2-np.sqrt(x),'k--',linewidth=2,label='mean - SD')
plt.xlabel('time')
plt.show()

dat = pd.read_csv("./../datasets/Stock_bond.csv")
fig = plt.figure()
fig.add_subplot(121)
plt.plot(dat.GM_AC)
plt.xlabel('Index')
plt.ylabel('GM_AC')

fig.add_subplot(122)
plt.plot(dat.F_AC)
plt.xlabel('Index')
plt.ylabel('F_AC')
plt.show()

plt.figure()
n=len(dat.GM_AC)
GMReturn = dat.GM_AC.values[1:]/dat.GM_AC.values[:-1] - 1.0
FReturn = dat.F_AC.values[1:]/dat.F_AC.values[:-1] - 1.0 
plt.plot(GMReturn,FReturn,'o')
plt.xlabel('GMReturn')
plt.ylabel('FReturn')
plt.show()

#########  problem 3  ##########
plt.figure()
MSFTReturn = dat.MSFT_AC.values[1:]/dat.MSFT_AC.values[:-1] - 1
MRKReturn = dat.MRK_AC.values[1:]/dat.MRK_AC.values[:-1] - 1
plt.plot(MSFTReturn,MRKReturn,'o')
plt.xlabel('MSFTReturn')
plt.ylabel('MRKReturn')
plt.show()
print('Correlation between MSFTReturn and MRKReturn = %f\n'%sst.pearsonr(MSFTReturn,MRKReturn)[0])

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
#
# print(pearson_def(MSFTReturn,MRKReturn))
#

################################################################
########## Code for simulations   ##############################
################################################################
np.random.seed(2009)

next_days = 45
niter = 1000
yearly_mean = 0.05 # per year mean of daily log returns on the stock have a 
yearly_std_dev = 0.23 # per year.
num_trading_days = 253
daily_mean = yearly_mean/num_trading_days
daily_std_dev = yearly_std_dev/np.sqrt(num_trading_days)
below = np.zeros(niter)

for i in range(0,niter):
    r = np.random.normal(loc=daily_mean, scale=daily_std_dev,size=next_days)
    logPrice = np.log(1e6) + np.cumsum(r)
    minlogP  = np.min( logPrice )
    below[i] = (minlogP < np.log(950000))

print("Probability that the value of the stock will be below $950,000 at the close of at least one of the next 45 trading days = %f\n"%below.mean())




################################################################
########## Code for Simulating a geometric random walk   #######
################################################################
np.random.seed(2012)
num_trading_days = 253
yearly_mean = 0.05 # per year mean of daily log returns on the stock have a 
yearly_std_dev = 0.23 # per year.
num_trading_days = 253
daily_mean = yearly_mean/num_trading_days
daily_std_dev = yearly_std_dev/np.sqrt(num_trading_days)

fig, ax_arr = plt.subplots(nrows=3,ncols=3)
fig.set_figwidth(15)
fig.set_figheight(15)
for i, ax in zip(range(0,9), ax_arr.flatten()):
    logr = np.random.normal(loc=daily_mean, scale=daily_std_dev,size=num_trading_days) 
    price = np.append([120.0], 120*np.exp(np.cumsum(logr)))
    
    #ax = plt.subplot(9,i,j)
    ax.plot(price,marker='o',fillstyle='none')

plt.show()


