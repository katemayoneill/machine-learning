import numpy as np
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# id:21-42-21
# i (a)

df=pd.read_csv("week3.csv",header=0,names=['x1','x2','y'])

x1=df['x1']
x2=df['x2']
y=df['y']


fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

ax.scatter(x1,x2,y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')


plt.savefig('fig1')

# i (b)

from sklearn.preprocessing import PolynomialFeatures

x=df[['x1','x2']]
poly=PolynomialFeatures(degree=5,include_bias=False)
x_poly=poly.fit_transform(x)

print(x_poly)



