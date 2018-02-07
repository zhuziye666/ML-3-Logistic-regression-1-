# ML-3-Logistic-regression-1-
practise about machine learnning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#导入数据并预处理
path='ex2data1.txt'
data=pd.read_csv(path,header=None,names=['Exam1','Exam2','Admitted'])
print(data.head())
#数据预处理
data.insert(0,'Ones',1)
print(data.head())
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
print(X.head())
print(y.head())
X=np.array(X.values)
y=np.array(y.values)
theta=np.zeros([cols-1])
#print(X,y,theta)
print(X.shape,y.shape,theta.shape)

#创建两个分数的散点图，并用颜色来可视化
positive=data[data['Admitted'].isin([1])]
negative=data[data['Admitted'].isin([0])]

fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam1'],positive['Exam2'],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam1'],negative['Exam2'],s=50,c='r',marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.show()
#定义sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))
#定义代价函数以及梯度下降
def costFunction(theta,X,y):
    m=len(y)
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    #first=np.multiply(-y,np.log(sigmoid(X*theta.T)))
    #second=np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    #J=np.sum(first-second)/m
    first=y.T*(np.log(sigmoid(X*theta.T)))
    second=(1-y).T*(np.log(1-sigmoid(X*theta.T)))
    J=-(first+second)/m
    return J
print(costFunction(theta,X,y))
def gradient(theta,X,y):
    m=len(X)
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    #grad=np.zeros(3)
    #inner1=(sigmoid(X*theta.T)-y).T*X
    #grad=inner1/m
    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)
    error=sigmoid(X*theta.T)-y
    for i in range(parameters):
        term=np.multiply(error,X[:,i])
        grad[i]=np.sum(term)/m
    return grad
print(gradient(theta,X,y))
#求解器qiujie
import scipy.optimize as opt
#result=opt.fmin_tnc(func=costFunction,x0=theta,fprime=gradient,args=(X,y))
result=opt.minimize(fun=costFunction,x0=theta,args=(X,y),method='Newton-CG',jac=gradient)
print(result)
#定义预测函数模型并分析结果
def predict(theta,X):
    probability=sigmoid(X@theta.T)
    return[1 if x>=0.5 else 0 for x in probability] 
from sklearn.metrics import classification_report  
theta_min=result.x
p=predict(theta_min,X)
#print(p)
print(classification_report(y,p))
#绘制边界
coef=-(result.x/result.x[2])
print(coef)
x=np.arange(130,step=0.1)
y=coef[0]+coef[1]*x
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(x,y,'g',label='decision boundary')
ax.scatter(positive['Exam1'],positive['Exam2'],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam1'],negative['Exam2'],s=50,c='r',marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
ax.set_title('Decision Boundary')
plt.show()
