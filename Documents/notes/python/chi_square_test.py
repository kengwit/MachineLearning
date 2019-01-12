import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
from scipy.stats import norm
from scipy.stats import chi2, chisquare
#np.random.seed(42)

n = 20
nsim = 1000
nbins = 4
nbins_plot = 40
nbin_edges = nbins+1
x = np.random.normal(0, 1, (n,nsim))
# create four bins with cutpoints at the quartiles of a standard normal: -0.675, 0, +0.657,
# i.e., create for 4 bins of standard normal for probabilities [0., 0.25, 0.5, 0.75, 1.0]
bins = norm.ppf( np.linspace(0,1,num=nbin_edges) )  
#bins = [0.001, 0.25, 0.5, 0.75, 0.999]
# Compute statistics.
#
#x = np.zeros(shape=(5,2))
#for i in range(0,5):
#    x[i,0] = i/10
#    x[i,1] = 2*i/10
    
m = x.mean(axis=0)
ddof = 1 # note: ddof=0 is the max likelihood estimate (divide by n), ddof=1 is the unbiased estimate (divide by (n-1))
s = np.std(x, axis=0, ddof=ddof) 

# =====================================
# alternatively, calculate your own
#xdiff = x - m
#s_squared = np.apply_along_axis(lambda a: (a**2).sum(),0,xdiff) / ( n - ddof ) 
#s = np.sqrt( s_squared ) 
# =====================================

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
    #m      = np.zeros(ncols)
    #s      = np.zeros(ncols)
    for i in range(0,ncols):
        xi = x[:,i]
        hist, bin_edges = np.histogram(xi, bins=bin_seq)
        #print(bin_edges)
        #mids = 0.5*(bin_edges[1:] + bin_edges[:-1])    
        #print(mids,hist)
        #mean = np.average(mids, weights=hist)
        #var  = np.average((mids - mean)**2, weights=n)
        #print(mean)
        counts[:,i] = hist
        #m[i]        = norm.ppf(mean)
        #s[i] = np.sqrt(var)
        
    return counts

counts = get_counts(x,bins)


#def get_hist_stats(counts):
#    nbins = np.shape(counts)[0]
#    ncols = np.shape(counts)[1]
#    for i in range(0,ncols):
#        bins = counts[:,i]
#        mids = 0.5*(bins[1:] + bins[:-1])
#        print(mids)
#
#get_hist_stats(counts)
    
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

chisquared = ( (counts - expectations)**2 / expectations ).sum(axis=0)



plt.figure()
plt.hist(m, bins=nbins_plot,density=True)  # arguments are passed to np.histogram
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
# note: 
# (n-1)s^2/sig^2 ~ chi^2_{n-1}
# since sig=1 (normal(0,1))
# (n-1)s^2 ~ chi^2_{n-1}

df=n-ddof    
plt.hist(df*(s**2), bins=nbins_plot,density=True)  # arguments are passed to np.histogram
#plt.hist(s**2/m**2, bins=20,density=True)  # arguments are passed to np.histogram
#x =  x.asarray()*(n-1)
#x2 = np.sort( x.ravel() )
chi2_val0 = chi2.ppf(0.001, df)
chi2_val1 = chi2.ppf(0.999, df)
chi2_axis = np.linspace(chi2_val0, chi2_val1, 100)
plt.plot(chi2_axis, chi2.pdf(chi2_axis, df ) )
#plt.plot(x_axis /df, chi2.pdf(x_axis , df )*df )

#plt.plot(x*df, 1.0 )
#plt.xlim(0,1.0)
#plt.plot(x,chisquare(x, axis=None))
plt.xlabel('$s^2$')
plt.ylabel('Density')

plt.figure()
plt.hist(chisquared, bins=nbins_plot,density=True)  # arguments are passed to np.histogram
chi2_df1_val0 = chi2.ppf(0.001, 1)
chi2_df1_val1 = chi2.ppf(0.999, 1)
chi2_df1_axis = np.linspace(chi2_df1_val0, chi2_df1_val1, 100)
plt.plot(chi2_df1_axis, chi2.pdf(chi2_df1_axis, 1 ) )

chi2_df2_val0 = chi2.ppf(0.001, 2)
chi2_df2_val1 = chi2.ppf(0.999, 2)
chi2_df2_axis = np.linspace(chi2_df2_val0, chi2_df2_val1, 100)
plt.plot(chi2_df2_axis, chi2.pdf(chi2_df2_axis, 2 ) )

plt.xlim(0,12)
plt.ylim(0,0.6)
plt.xlabel('$\chi^2$')
plt.ylabel('Density')
