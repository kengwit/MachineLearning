import numpy as np
import pandas as pd
from scipy import stats

# import data, tab delimited, ignore first column
df = pd.read_csv('../data/prostate/prostate.data', delimiter='\t', index_col=0)

# remove columns lpsa and train
mask_train = df.pop('train')
df_y = df.pop('lpsa')



# Table 3.1. Correlations of predictors in the prostate cancer data
print( df[ mask_train=='T' ].corr().to_string()     )

""" TABLE 3.2. Linear model fit to the prostate cancer data. The Z score is the
coefficient divided by its standard error (3.12). Roughly a Z score larger than two
in absolute value is significantly nonzero at the p = 0.05 level.
"""
class LinearRegression:
    #def __init__(self):
    #    self.beta
    #    self.stderr
    #    self.z_score
        
    def fit(self, X, y):
        """ form X = [ 1 x_1 ]
                     [ 1 x_2 ]
                     [ 1 x_3 ]
                     [ :  :  ]
                     [ :  :  ]
                     [ 1 x_N ]            
        
        """
        X = np.c_[np.ones((X.shape[0], 1)), X] # c_ is "concatenate"
        XX_inv = np.linalg.inv(X.T @ X)
        self.beta = XX_inv @ X.T @ y
        
        # calculate variance estimate:
        var = np.sum((X @ self.beta - y)**2) / (X.shape[0] - X.shape[1])
        
        self.stderr  = np.sqrt(np.diag(XX_inv * var))
        self.z_score = self.beta / self.stderr
        #print(self.z_score)
        
    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X @ self.beta

#print(df.head())



# =============================================================================
# IMPORTANT!!! 
#  normalize each column of the table by calculating 
#  the z-score for each column!
#  However, the Z values for the parameters except the intercept would NOT be
#  be affected
#
#
# for each column calculate, zscore vector is calculated as:
#
# v=df['lcavol'].values
# mean=np.mean(v)
# z=(v - np.mean(v))/np.std(v,ddof=0)
# print(z)
# =============================================================================
df = df.apply(lambda a: stats.zscore(a, axis=0)) 
print(df.head())

train_x = df[mask_train == 'T']
train_y = df_y[mask_train == 'T']
model = LinearRegression()
model.fit(train_x.values,train_y.values)

df2 = pd.DataFrame(data = {'Coefficient': model.beta, 
                     'Std. Error': model.stderr,
                     'Z Score' : model.z_score}, 
             index = ["Intercept", *df.columns.tolist()])

print( df2.to_string() )

rss1 = np.sum( ( model.predict( train_x.values ) - train_y.values )**2 )

train_x_hyp = train_x.drop(columns=['age', 'lcp', 'gleason', 'pgg45'])
model_hyp = LinearRegression()
model_hyp.fit(train_x_hyp,train_y.values)
rss0 = np.sum( ( model_hyp.predict(train_x_hyp.values) - train_y.values )** 2 )

p1_plus_1 = len(model.beta)
p0_plus_1 = len(model_hyp.beta)
N  = len(train_x)
dfn = p1_plus_1 - p0_plus_1
dfd = N-p1_plus_1
f_stats = ( ( rss0 - rss1 )/dfn ) / (rss1/dfd)

# calculate p-value using F-distribution
prob = 1 - stats.f.cdf(f_stats, dfn = dfn, dfd = dfd)
print ('RSS1 = ', rss1)
print ('RSS0 = ', rss0)
print ('F =', f_stats)
print (f'Pr(F({dfn}, {dfd}) > {f_stats:.2f}) = {prob:.2f}')



