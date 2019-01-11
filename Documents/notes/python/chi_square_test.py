import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
from scipy.stats import norm
from scipy.stats import chi2, chisquare
#np.random.seed(42)

n = 20
nsim = 3
x = np.random.normal(0, 1, (n,nsim))
# create four bins with cutpoints at the quartiles of a standard normal: -0.675, 0, +0.657,
# i.e., create for 4 bins of standard normal for probabilities [0., 0.25, 0.5, 0.75, 1.0]
bins = norm.ppf( np.linspace(0,1,num=5) )  

# Compute statistics.
#
#x = np.zeros(shape=(5,2))
#for i in range(0,5):
#    x[i,0] = i/10
#    x[i,1] = 2*i/10
    
m = x.mean(axis=0)
s = np.std(x, axis=0, ddof=0) # note: ddof=0 is the max likelihood estimate, ddof=1 is the unbiased estimate
##s = np.std(x, axis=0, ddof=1) # note: ddof=0 is the max likelihood estimate, ddof=1 is the unbiased estimate

#check = x[:,0]
#check = check - check.mean()
#std_dev = np.sqrt( (check**2).sum() / ( len(check) - 1 ) )



# for a given column, bin the elements in that column
#def get_counts(x,bin_seq):
#    ncols = np.shape(x)[1]
#    nbins = len(bin_seq)
#    counts = np.zeros(shape=(nbins,ncols))
#    for i in range(0,ncols):        
#        print(np.histogram(x[:,i], bins=bin_seq))
#    
#    return counts
#
#counts = get_counts(x,bins)


#def my_func(a):
#    return a.sum()
#b = np.array([[1,2,3], [4,5,6], [7,8,9]])
#print( np.apply_along_axis(my_func, 0, b) )
#counts2, bin_edges = np.apply_along_axis(lambda a: np.histogram(a, bins=bins), 0, x)
def get_counts(x,bin_seq):
    ncols = np.shape(x)[1]
    nbins = len(bin_seq)-1
    counts = np.zeros(shape=(nbins,ncols))
    for i in range(0,ncols):
        xi = x[:,i]
        hist, bin_edges = np.histogram(xi, bins=bin_seq)
        counts[:,i] = hist
    return counts

counts = get_counts(x,bins)

def get_expectations(x,m,s,bin_seq):
    
    n     = np.shape(x)[0]
    ncols = np.shape(x)[1]
    nbins = len(bin_seq)-1
    expectations = np.zeros(shape=(nbins,ncols))
    
    for i in range(0,ncols):
        mi = m[i]
        si = s[i]
        
        # note: 
        # The equivalent of the R pnorm() function is: scipy.stats.norm.cdf() with python 
        # The equivalent of the R qnorm() function is: scipy.stats.norm.ppf() with python
        cdf_i = sst.norm.cdf(bin_seq,loc=mi,scale=si)
        
        exp_i = n*np.diff(cdf_i)
        # bin the cdf
        #exp_i, bin_edges_i = np.histogram(exp_i, bins=bin_seq)
        
        # get the counts in each bin
        #exp_i = n*exp_i
        #print(exp_i.reshape(4,5))
        #print( np.shape( exp_i ))
        expectations[:,i] = exp_i
    
    return expectations

expectations = get_expectations(x,m,s,bins)

plt.figure()
plt.hist(m, bins=20,density=True)  # arguments are passed to np.histogram
x_axis = np.arange(-2, 2, 0.001)

# note: 
# If X1,X2,…,Xn are n independent observations from a 
# population that has a mean μ and standard deviation σ, 
# then the variance of the total T=(X1+X2+⋯+Xn) is nσ2.
#
# The variance of T/n must be (nσ^2)/n^2=σ^2/n. 
# And the standard deviation of T/n must be sqrt(variance) = σ/sqrt(n). 
# Of course, T/n is the sample mean \bar{x}.
#
plt.plot(x_axis, norm.pdf(x_axis,loc=0.0,scale=1/np.sqrt(n)))
plt.xlim(-0.75,0.75)
plt.xlabel('Mean')
plt.ylabel('Density')

plt.figure()
#plt.hist(s**2, bins=20,density=True)  # arguments are passed to np.histogram
#x =  x.asarray()*(n-1)
#x2 = np.sort( x.ravel() )
df=n-1
#x_axis = np.linspace(chi2.ppf(0.001, df), chi2.ppf(0.999, df), 100)
#plt.plot(x_axis /df, chi2.pdf(x_axis , df )*df )

plt.plot(x*df, 1.0 )
#plt.xlim(0,1.0)
#plt.plot(x,chisquare(x, axis=None))
plt.xlabel('$s^2$')
plt.ylabel('Density')