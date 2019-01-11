import numpy as np
from scipy.stats import norm

n = 20
nsim = 10000
x = np.random.normal(0, 1, (20,nsim))
# create four bins with cutpoints at the quartiles of a standard normal: -0.675, 0, +0.657,
# i.e., create for 4 bins of standard normal for probabilities [0., 0.25, 0.5, 0.75, 1.0]
bins = norm.ppf( np.linspace(0,1,num=5) )  

# Compute statistics.
#
x = np.zeros(shape=(5,2))
for i in range(0,5):
    x[i,0] = i
    x[i,1] = 2*i
    
m = x.mean(axis=0)
s = np.std(x, axis=0, ddof=0) # note: ddof=0 is the max likelihood estimate, ddof=1 is the unbiased estimate
##s = np.std(x, axis=0, ddof=1) # note: ddof=0 is the max likelihood estimate, ddof=1 is the unbiased estimate

#check = x[:,0]
#check = check - check.mean()
#std_dev = np.sqrt( (check**2).sum() / ( len(check) - 1 ) )



# for a given column, bin the elements in that column