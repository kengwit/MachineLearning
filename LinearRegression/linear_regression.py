import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

# using tensorflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

import scipy as sp
# get advertising data: sales (in thousands of units) as a function of 
# advertising budgets (in thousands of dollars) for TV, radio, newspaper media
advertising = pd.read_csv('./../Data/Advertising.csv',usecols=[1,2,3,4])
credit = pd.read_csv('./../Data/Credit.csv',usecols=list(range(1,12)))

#advertising.info()
#print(advertising.head(3))
#credit.info()
#print(credit.Student)
#print(credit.head(3))

### map credit.Student no-yes vector to 0-1 vector
###credit['Student2'] = credit.Student.map({'No': 0, 'Yes':1})
###print(credit.head(3))

### read 'Auto.csv' data, equate '?' with N/A and drop those rows with N/A
##auto = pd.read_csv('./../Data/Auto.csv',na_values='?').dropna()
###print(len(auto))

#
# Perform linear regression
# Parameters are:
#
#
# order : int, optional
# If order is greater than 1, use numpy.polyfit to estimate a polynomial regression.
# order = 1 (LINEAR FIT)
#
# ci : int in [0, 100] or None, optional
# Size of the confidence interval for the regression estimate. This will be drawn using translucent bands around the regression line. The confidence interval is estimated using a bootstrap; for large datasets, it may be advisable to avoid that computation by setting this parameter to None.
#
# {scatter,line}_kws : dictionaries
# Additional keyword arguments to pass to plt.scatter and plt.plot.
# {'color':'r', 's':9} --> color = red , size = 9
#
p = sns.regplot(advertising.TV,advertising.sales,order=1,ci=None,scatter_kws={'color':'r', 's':119})
plt.xlim(-10,310)
plt.ylim(ymin=0)
plt.show()

# =================================================================
# Regression coefficients using sp.stats.linregress
#print(p.get_lines()[0].get_xdata())
#p.get_lines()[0].get_ydata()
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x=p.get_lines()[0].get_xdata(),y=p.get_lines()[0].get_ydata())
print("intercept from sp.stats.linregress = ",intercept)
print("slope from sp.stats.linregress = ",slope)

# =================================================================
# Regression coefficients using skl_lm.LinearRegression()
regr = skl_lm.LinearRegression()
### X = scale(advertising.TV,with_mean=False,with_std=False).reshape(-1,1) # reshape to (200,1), Note: csv data has already been centered
X = (advertising.TV).values.reshape(-1,1) 
y = advertising.sales
regr.fit(X,y)
print("intercept from skl_lm.LinearRegression() = ",regr.intercept_)
print("slope from skl_lm.LinearRegression() = ",regr.coef_)


# plot scale
plot_scale = 1000.

# create grid coordinates for plotting
B0 = np.linspace(regr.intercept_-2, regr.intercept_+2,50)
B1 = np.linspace(regr.coef_-0.02,regr.coef_+0.02,50)
b0_mesh,b1_mesh = np.meshgrid(B0,B1,indexing='xy')
Z = np.zeros((B0.size,B1.size))
# calculate Z-values (residual sum-of-squares)
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = ((y-(b0_mesh[i,j]+X.ravel()*b1_mesh[i,j]))**2.0).sum()/plot_scale # X.ravel() reshapes back to original so that it is the same shape as y

    
# Minimized RSS
min_RSS_label = r'$\beta_0$, $\beta_1$ for minimized RSS'
min_rss = np.sum((regr.intercept_+regr.coef_*X - y.values.reshape(-1,1))**2)/plot_scale
print(min_rss)


fig = plt.figure(figsize=(15,6))
fig.suptitle('RSS - Regression coefficients', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(b0_mesh, b1_mesh, Z, cmap=plt.cm.Set1, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax1.scatter(regr.intercept_, regr.coef_[0], c='r', label=min_RSS_label)
ax1.clabel(CS, inline=True, fontsize=10, fmt='%1.1f')

# Right plot
ax2.plot_surface(b0_mesh, b1_mesh, Z, rstride=3, cstride=3, alpha=0.3)
ax2.contour(b0_mesh, b1_mesh, Z, zdir='z', offset=Z.min(), cmap=plt.cm.Set1,
            alpha=0.4, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax2.scatter3D(regr.intercept_, regr.coef_[0], min_rss, c='r', label=min_RSS_label)
ax2.set_zlabel(r'RSS ($\times 10^3$)')
ax2.set_zlim(Z.min(),Z.max())
ax2.set_ylim(0.02,0.07)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\beta_0$', fontsize=17)
    ax.set_ylabel(r'$\beta_1$', fontsize=17)
    ax.set_yticks([0.03,0.04,0.05,0.06])
    ax.legend()

