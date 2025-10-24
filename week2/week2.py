import numpy as np
import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt

# (a)
# (i)

df = pd.read_csv("week2.csv", header=0, names=['X1','X2','Y'])

plus = df[df['Y']>0]
minus = df[df['Y']<0]

plt.figure()

plt.scatter(plus['X1'],plus['X2'],marker='P',color='green',label="y = +1")
plt.scatter(minus['X1'],minus['X2'],marker='o',color='red',label="y = -1")


plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()

plt.savefig('fig1')

# (ii)

from sklearn.linear_model import LogisticRegression

x = df[['X1','X2']]
y = df['Y']

model = LogisticRegression(penalty=None)
model.fit(x,y)

print(f"intercept : {model.intercept_}, coefficients : {model.coef_}")

# (iii)

plt.figure()
plt.scatter(plus['X1'],plus['X2'],marker='P',color='green',label="y = +1")
plt.scatter(minus['X1'],minus['X2'],marker='o',color='red',label="y = -1")

x1 = df['X1']

y_hat = model.predict(x)

predictions = x.copy()
predictions.insert(2, 'Y_hat', y_hat)


plus = predictions[predictions['Y_hat']>0]
minus = predictions[predictions['Y_hat']<0]


plt.scatter(plus['X1'],plus['X2'],marker='+',color='lightgreen',label="y_hat = +1")
plt.scatter(minus['X1'],minus['X2'],marker='.',color='pink',label="y_hat = -1")

x2 = -( model.intercept_ + model.coef_[0][0] * x1) / model.coef_[0][1]

plt.plot(x1, x2, color='magenta', label="decision boundary")



plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()

plt.savefig('fig2')

# (b)
# (i)

from sklearn.svm import LinearSVC


c_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

row = 0
col = 0

fig = plt.figure(figsize=(15, 20))

axs = fig.subplots(4, 2)

for c in c_values:
    model = LinearSVC(C=c, max_iter=10000)
    model.fit(x, y)
    print(f"C = {c}: intercept : {model.intercept_[0]}, coefficients : {model.coef_[0]}")

    plus = df[df['Y']>0]
    minus = df[df['Y']<0]

    y_hat = model.predict(x)
    predictions = x.copy()

    predictions.insert(2, 'Y_hat', y_hat)

    axs[row,col].set_title(f"C = {c}")

    axs[row,col].scatter(plus['X1'],plus['X2'],marker='P',color='green',label="y = +1")
    axs[row,col].scatter(minus['X1'],minus['X2'],marker='o',color='red',label="y = -1")

    plus = predictions[predictions['Y_hat']>0]
    minus = predictions[predictions['Y_hat']<0]

    axs[row,col].scatter(plus['X1'],plus['X2'],marker='+',color='lightgreen',label="y_hat = +1")
    axs[row,col].scatter(minus['X1'],minus['X2'],marker='.',color='pink',label="y_hat = -1")

    x2 = -( model.intercept_ + model.coef_[0][0] * x1) / model.coef_[0][1]

    axs[row,col].plot(x1, x2, color='magenta', label="decision boundary")

    if(row==3 and col<=1) :
        col=1
        row=0

    elif(row<3): row+=1

axs[0,0].legend()



plt.savefig('fig3')

# (c)
# (i) (ii)
df.insert(2, 'X3', np.square(df['X1']))
df.insert(3, 'X4', np.square(df['X2']))

plus = df[df['Y']>0]
minus = df[df['Y']<0]

x = df[['X1','X2','X3','X4']]
y = df['Y']

model = LogisticRegression(penalty=None)
model.fit(x,y)

print(f"intercept : {model.intercept_}, coefficients : {model.coef_}")

plt.figure()
plt.scatter(plus['X1'],plus['X2'],marker='P',color='green',label="y = +1")
plt.scatter(minus['X1'],minus['X2'],marker='o',color='red',label="y = -1")


y_hat = model.predict(x)

predictions = x.copy()
predictions.insert(2, 'Y_hat', y_hat)


plus = predictions[predictions['Y_hat']>0]
minus = predictions[predictions['Y_hat']<0]


plt.scatter(plus['X1'],plus['X2'],marker='+',color='lightgreen',label="y_hat = +1")
plt.scatter(minus['X1'],minus['X2'],marker='.',color='pink',label="y_hat = -1")

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.savefig('fig4')


# (iii)

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

baseline = DummyClassifier()

baseline.fit(x,y)
baseline_y = baseline.predict(x)


baseline_score = accuracy_score(y, baseline_y)
accuracy_score = accuracy_score(y, y_hat)

print(f"baseline accuracy score: {baseline_score}")
print(f"logistic regression classifier accuracy score: {accuracy_score}")

