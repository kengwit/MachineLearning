

# %load ../standard_import.txt
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

plt.style.use('seaborn-white')

# ==============================================
# Load Datasets
# ==============================================
advertising = pd.read_csv('Ref/Data/Advertising.csv', usecols=[1,2,3,4])
advertising.info()

credit = pd.read_csv('Ref/Data/Credit.csv', usecols=list(range(1,12)))
credit['Student2'] = credit.Student.map({'No':0, 'Yes':1})
credit.head(3)

auto = pd.read_csv('Ref/Data/Auto.csv', na_values='?').dropna()
auto.info()

# ==============================================
# 3.1 Simple Linear Regression
# Figure 3.1 - Least squares fit
# ==============================================
sns.regplot(advertising.TV, advertising.Sales, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.xlim(-10,310)
plt.ylim(ymin=0)

# ==============================================
# Figure 3.2 - Regression coefficients - RSS 
# ==============================================
# Regression coefficients (Ordinary Least Squares)
regr = skl_lm.LinearRegression()

X = scale(advertising.TV, with_mean=True, with_std=False).reshape(-1,1)
y = advertising.Sales

regr.fit(X,y)
print(regr.intercept_)
print(regr.coef_)

# Create grid coordinates for plotting
B0 = np.linspace(regr.intercept_-2, regr.intercept_+2, 50)
B1 = np.linspace(regr.coef_-0.02, regr.coef_+0.02, 50)
xx, yy = np.meshgrid(B0, B1, indexing='xy')
Z = np.zeros((B0.size,B1.size))

# Calculate Z-values (RSS) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    # note: 
    #  X was reshaped from (200,) into (200,1)
    #  X.ravel converts this back to (200,)
    ypred = xx[i,j]+X.ravel()*yy[i,j] # this has size (200,)    
    Z[i,j] =( (y - ypred)**2 ).sum()/1000 # in units of thousands
    
# Minimized RSS
min_RSS = r'$\beta_0$, $\beta_1$ for minimized RSS'
min_rss = np.sum(( regr.intercept_ + regr.coef_*X - y.values.reshape(-1,1) )**2)/1000
print(min_rss)

fig = plt.figure(figsize=(15,6))
fig.suptitle('RSS - Regression coefficients', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(xx, yy, Z, cmap=plt.cm.Set1, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax1.scatter(regr.intercept_, regr.coef_[0], c='r', label=min_RSS)
ax1.clabel(CS, inline=True, fontsize=10, fmt='%1.1f')

# Right plot
ax2.plot_surface(xx, yy, Z, rstride=3, cstride=3, alpha=0.3)
ax2.contour(xx, yy, Z, zdir='z', offset=Z.min(), cmap=plt.cm.Set1,
            alpha=0.4, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax2.scatter3D(regr.intercept_, regr.coef_[0], min_rss, c='r', label=min_RSS)
ax2.set_zlabel('RSS')
ax2.set_zlim(Z.min(),Z.max())
ax2.set_ylim(0.02,0.07)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\beta_0$', fontsize=17)
    ax.set_ylabel(r'$\beta_1$', fontsize=17)
    ax.set_yticks([0.03,0.04,0.05,0.06])
    ax.legend()

# ===================================================================   
# Confidence interval on page 67 & Table 3.1 & 3.2 using Statsmodels
# ===================================================================   
est = smf.ols('Sales ~ TV', advertising).fit()
print(est.summary().tables[1])

# RSS with regression coefficients
Sales_pred = est.params[0] + est.params[1]*advertising.TV
RSS = ((advertising.Sales - Sales_pred)**2).sum()
RSE = math.sqrt(RSS/( len(advertising.Sales) - 2 ))
print(RSE)

TSS = ((advertising.Sales - np.mean(Sales_pred))**2).sum()
R2 = 1.0 - RSS/TSS
print(R2)

# ===================================================================   
# Table 3.1 & 3.2 using Scikit-learn
# ===================================================================   
regr = skl_lm.LinearRegression()

X = advertising.TV.values.reshape(-1,1)
y = advertising.Sales

regr.fit(X,y)
print(regr.intercept_)
print(regr.coef_)
# RSS with regression coefficients
RSS = ((advertising.Sales - (regr.intercept_ + regr.coef_*advertising.TV))**2).sum()
RSE = math.sqrt(RSS/( len(advertising.Sales) - 2 ))
print(RSE)

mean_sales = np.mean(advertising.Sales.values)
print("percent error = %f\n"%(RSE/mean_sales*100))

Sales_pred = regr.predict(X)
R2 = r2_score(y, Sales_pred)
print(R2)

