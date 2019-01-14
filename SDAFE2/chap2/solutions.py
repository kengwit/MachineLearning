import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
#from scipy.stats import norm
#from scipy.stats import chi2, chisquare
import pandas as pd 
from IPython.display import display


################################################################
#
# Problem 12. Code to produce the plot is below.
# We see that the return and log return for any day are almost equal.  This is
# reasonable in light of the discussion in Section 2.1.3, especially Figure 2.1.
################################################################
data = pd.read_csv("./../datasets/MCD_PriceDaily.csv")
adjPrice = data['Adj Close'].values
logReturn = np.diff( np.log(adjPrice) )
Return = adjPrice[1:]/adjPrice[:-1] - 1
plt.plot(Return,logReturn,'ko',fillstyle='none')
plt.xlabel('Return')
plt.ylabel('log Return')

x = Return
slope = 1
intercept = 0
y = slope*x + intercept
plt.plot(x,y,'r-')
plt.show()    
    
