from tracemalloc import Snapshot
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# dataset not part of project files
# download from:
# https://www.kaggle.com/datasets/jillanisofttech/amazon-product-reviews
df = pd.read_csv('./data/Reviews.csv')

# drop NaNs
df = df.dropna(subset=['ProfileName'])
df = df.dropna(subset=['Summary'])

# condense helpfulness into more useful values
df['HelpfulPercent'] = np.where(df['HelpfulnessDenominator'] > 0, df['HelpfulnessNumerator']/df['HelpfulnessDenominator'], -1)
indices = df[df['HelpfulPercent'] == -1].index
df.drop(indices,inplace=True)
df['UpvotePercent'] = pd.cut( df['HelpfulPercent'] , bins = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = ['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])

df_s = df.groupby(['Score','UpvotePercent']).agg({'Id':'count'}).reset_index()

# ordinary least squares linear regression
from sklearn.linear_model import LinearRegression
arr = np.array(df['Score']).reshape(-1,1)
reg = LinearRegression().fit(arr, df['HelpfulPercent'])
pred = reg.predict(arr)
print("Coefs: ", reg.coef_)
print("MSE: ", mean_squared_error(df['HelpfulPercent'],pred))

import matplotlib.pyplot as plt

'''
plt.scatter(df['Score'], df['HelpfulPercent'], alpha=0.1)
plt.plot(df['Score'], pred, color="black")
plt.xlabel("Score")
plt.ylabel("Helpfulness Ratio")
plt.show()
'''

y = df['HelpfulPercent']

# add total wordcount feature
df['Wordcount'] = df['Text'].str.split().str.len()
print(df['Wordcount'])

# print min & max wordcounts
print(f"Min wc: {df['Wordcount'].min()}")
print(f"Num reviews w min wc: {len(np.where(df['Wordcount']==df['Wordcount'].min()))}")
print(f"Max wc: {df['Wordcount'].max()}")

# least squares regression using just length of review
arr2 = np.array(df['Wordcount']).reshape(-1,1)
reg2 = LinearRegression().fit(arr2,df['HelpfulPercent'])
pred2 = reg2.predict(arr2)
print("Coefs: ", reg2.coef_)
print("MSE: ", mean_squared_error(df['HelpfulPercent'],pred2))

'''
plt.scatter(df['Wordcount'], df['HelpfulPercent'], alpha=0.5)
plt.plot(df['Wordcount'], pred2, color="black")
plt.xlabel("Wordcount")
plt.ylabel("Helpfulness Ratio")
plt.show()
'''

# least squares using both??
arr3 = df[['Score', 'Wordcount']].to_numpy()
reg3 = LinearRegression().fit(arr3, df['HelpfulPercent'])
pred3 = reg3.predict(arr3)
print("Coefs: ", reg3.coef_)
print("MSE: ", mean_squared_error(df['HelpfulPercent'], pred3))

from mpl_toolkits import mplot3d
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Score'], df['Wordcount'], df['HelpfulPercent'], alpha=0.5)
ax.set_xlabel("Score")
ax.set_ylabel("Wordcount")
ax.set_zlabel("Helpfulness Ratio")
coefs = reg3.coef_
intercept = reg3.intercept_
'''

# score 1-5, wc 0-3500
'''
xs = np.tile(np.arange(6, step=0.5), (12,1))
ys = np.tile(np.arange(3500, step=292), (12,1)).T
print(xs)
zs = xs * coefs[0] + ys * coefs[1] + intercept
ax.plot_surface(xs,ys,zs, alpha=0.5, color="black")
plt.show()
'''

# extract features from text
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(df['Text'])
print(f"X_count shape: {X_counts.shape}")

# get term frequencies
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
X_tf = tf_transformer.transform(X_counts)

y = df['HelpfulPercent']

# pca using term frequencies
# keep 20 components
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components=20)
principalComponents = pca.fit_transform(X_tf)
principal_df = pd.DataFrame(data = principalComponents)
print(principal_df)

# least squares using all 20 components
# can't visualize but we can look at mse
reg4 = LinearRegression().fit(principalComponents, df['HelpfulPercent'])
pred4 = reg4.predict(principalComponents)
print("Coefs: ", reg4.coef_)
print("20 component MSE: ", mean_squared_error(df['HelpfulPercent'], pred4))

# least squares using 20 term freq components + score + wc
df = df.reset_index(drop=True)
principal_df = principal_df.join(df['Score'].astype(float))
principal_df = principal_df.join(df['Wordcount'].astype(float))
print(df['Score'])
print(principal_df)
arr5 = principal_df.to_numpy()
print(f"score nan count: {df['Score'].isna().sum()}\nwc nan count: {df['Wordcount'].isna().sum()}")
print(f"score nan count: {principal_df['Score'].isna().sum()}\nwc nan count: {principal_df['Wordcount'].isna().sum()}")
reg5 = LinearRegression().fit(arr5, df['HelpfulPercent'])
pred5 = reg5.predict(arr5)
print("Coefs: ", reg5.coef_)
print("20 component + 2 MSE: ", mean_squared_error(df['HelpfulPercent'], pred5))

# least squares using 1 component to visualize
arr6 = np.array(principal_df[[0,1]])
reg6 = LinearRegression().fit(arr6, df['HelpfulPercent'])
pred6 = reg6.predict(arr6)
print("Coefs: ", reg6.coef_)
print("1 component mse: ", mean_squared_error(df['HelpfulPercent'], pred6))

'''
print(principal_df[0].shape)
print(pred6.shape)
print(df['HelpfulPercent'].shape)
plt.scatter(principal_df[0], df['HelpfulPercent'], alpha=0.5, s=1)
plt.plot(principal_df[0], pred6, color="black")
plt.xlabel("Principal Component 1")
plt.ylabel("Helpfulness Ratio")
plt.show()
'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_df[0], principal_df[1], df['HelpfulPercent'], alpha=0.2)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Helpfulness Ratio")
coefs = reg6.coef_
intercept = reg6.intercept_

xs = np.tile(np.arange(1, step=0.1), (10,1))
ys = np.tile(np.arange(stop=0.6,start=-0.4, step=0.1), (10,1)).T
zs = xs * coefs[0] + ys * coefs[1] + intercept
ax.plot_surface(xs,ys,zs, alpha=0.5, color="black")

plt.show()

# pca w 2 components, including score & wc this time
import scipy
from sklearn.preprocessing import StandardScaler
np_arr_from_df = df[['Score', 'Wordcount']].to_numpy()
df_arr = scipy.sparse.csr_matrix(StandardScaler().fit_transform(np_arr_from_df))
arr7 = scipy.sparse.hstack([X_tf, df_arr])
pca = TruncatedSVD(n_components=2)
principalComponents2 = pca.fit_transform(arr7)
prin_df = pd.DataFrame(data = principalComponents2)
print(prin_df)

'''
plt.scatter(prin_df[0], prin_df[1], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
'''

reg8 = LinearRegression().fit(prin_df, df['HelpfulPercent'])
pred8 = reg8.predict(prin_df)
print("Coefs: ", reg8.coef_)
print("MSE: ", mean_squared_error(df['HelpfulPercent'], pred8))

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(prin_df[0], prin_df[1], df['HelpfulPercent'], alpha=0.5)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Helpfulness Ratio")
'''

'''
coefs = reg8.coef_
intercept = reg8.intercept_
xs = np.tile(np.arange(25), (25,1))
ys = np.tile(np.arange(25), (25,1)).T
zs = xs * coefs[0] + ys * coefs[1] + intercept
ax.plot_surface(xs,ys,zs, alpha=0.5, color="black")

plt.show()
'''

# ridge regression
from sklearn.linear_model import Ridge
ridge = Ridge().fit(prin_df, df['HelpfulPercent'])
pred_ridge = ridge.predict(prin_df)
print("Coefs: ", ridge.coef_)
print("MSE: ", mean_squared_error(df['HelpfulPercent'], pred_ridge))

'''
coefs = ridge.coef_
intercept = ridge.intercept_
xs = np.tile(np.arange(25), (25,1))
ys = np.tile(np.arange(25), (25,1)).T
zs = xs * coefs[0] + ys * coefs[1] + intercept
ax.plot_surface(xs,ys,zs, alpha=0.5, color="black")

plt.show()
'''

# lasso
from sklearn.linear_model import Lasso
lasso = Lasso().fit(prin_df, df['HelpfulPercent'])
pred_lasso = lasso.predict(prin_df)
print("Coefs: ", lasso.coef_)
print("MSE: ", mean_squared_error(df['HelpfulPercent'], pred_lasso))

'''
coefs = lasso.coef_
intercept = lasso.intercept_
xs = np.tile(np.arange(25), (25,1))
ys = np.tile(np.arange(25), (25,1)).T
zs = xs * coefs[0] + ys * coefs[1] + intercept
ax.plot_surface(xs,ys,zs, alpha=0.5, color="black")

plt.show()
'''

# least squares; score from wordcount
arr9 = np.array(df['Wordcount']).reshape(-1,1)
reg9 = LinearRegression().fit(arr9, df['Score'])
pred9 = reg9.predict(arr9)
print("Coefs: ", reg9.coef_)
print("MSE: ", mean_squared_error(df['Score'], df['Wordcount']))

plt.scatter(df['Wordcount'], df['Score'], alpha=0.5, s=1)
plt.plot(df['Wordcount'], pred9, color="black")
plt.xlabel("Wordcount")
plt.ylabel("Score")
plt.show()
