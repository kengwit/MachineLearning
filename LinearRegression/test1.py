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

# get advertising data: sales (in thousands of units) as a function of 
# advertising budgets (in thousands of dollars) for TV, radio, newspaper media
advertising = pd.read_csv('./../Data/Advertising.csv',usecols=[1,2,3,4])
credit = pd.read_csv('./../Data/Credit.csv',usecols=list(range(1,12)))

#advertising.info()
#print(advertising.head(3))
#credit.info()
#print(credit.Student)
#print(credit.head(3))

# map credit.Student no-yes vector to 0-1 vector
credit['Student2'] = credit.Student.map({'No': 0, 'Yes':1})
#print(credit.head(3))

# read 'Auto.csv' data, equate '?' with N/A and drop those rows with N/A
auto = pd.read_csv('./../Data/Auto.csv',na_values='?').dropna()
#print(len(auto))

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
sns.regplot(advertising.TV,advertising.sales,order=1,ci=None,scatter_kws={'color':'r', 's':119})
plt.xlim(-10,310)
plt.ylim(ymin=0)

