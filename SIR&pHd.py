#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2


# In[2]:


def SIR(X, y, H, K):
    cov_x = np.cov(X, rowvar=False)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    Z = np.matmul(X-np.mean(X,0), r)

    width = (np.max(y) - np.min(y)) / H

    V_hat = np.zeros([X.shape[1], X.shape[1]])
    for h in range(H):
        h_index = np.logical_and(np.min(y)+h*width <= y, y < np.min(y)+(h+1)*width)
        ph_hat = np.mean(h_index)
        if ph_hat == 0:
            continue
        mh = np.mean(Z[h_index, :], axis=0)
        V_hat += ph_hat * np.matmul(mh[:, np.newaxis], mh[np.newaxis, :])

    # 特征值和特征向量（列向量）
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argpartition(np.abs(eigenvalues), X.shape[1]-K) >= X.shape[1]-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est = np.matmul(r, K_largest_eigenvectors)

    return edr_est


# In[3]:


def pHd(X,y,K):
    p,n=np.shape(X)
    cov_x = np.cov(X)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    X=np.mat(X)
    Z = np.dot(r,(X-np.mean(X,1)))
    Y = y - np.mean(y) #1*n
    Y=np.mat(Y)
    
    alpha=np.dot(np.linalg.inv(np.cov(Z)),np.cov(Z,Y)[0:p,p])  #p*1
    H1=np.zeros([p,p])
    for i in range(n):
        mi=np.dot(Z[:,i],Z[:,i].T)
        H1=H1+np.float(Y[:,i])*mi
    H1=H1/n
    e=Y-np.matmul(alpha.T,Z)
    
    M=np.matmul(H1,H1.T)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    K_index = np.argpartition(np.abs(eigenvalues), p-K) >= p-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est = np.matmul(r, K_largest_eigenvectors)
    
    return edr_est


# In[14]:


def gassiandata(p,n):
    mean = np.zeros(p)
    cov = np.identity(p)
    data = np.round(np.random.multivariate_normal(mean,cov,n),3)
    return data
#example 1
sample=400
dimension=10
x1=gassiandata(dimension,sample).T
noise1=np.random.normal(0,0.5,sample).T


# In[15]:


y1 = x1[0,:]*(x1[1,:]+x1[2,:]+1)+noise1
y2 =x1[0,:]*(x1[1,:]+x1[2,:]+2)+np.power((x1[3,:]+x1[4,:]+2),3)+noise1
def XuguanTest1(X,y):
    p, n = np.shape(X)
    cov_x = np.cov(X)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    X = np.mat(X)
    Z = np.dot(r, (X- np.mean(X, 1)))
    Y = y - np.mean(y) #1*n
    Y=np.mat(Y)
    alpha = np.matmul(np.linalg.inv(np.cov(Z)), np.cov(Z, Y)[0:p,p])  # p*1
    H1 = np.zeros([p, p])
    for i in range(n):
        mi = np.matmul(Z[:, i], Z[:, i].T)
        H1 = H1 + np.float(Y[:,i]) * mi
    H1 = H1 / n
    e = Y - np.matmul(alpha.T, Z)

    M = np.matmul(H1, H1.T)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    #print(M)

    N = n/(2*np.var(e))
    A = np.sort(eigenvalues)
    A = A[::-1]
    #print(A)
    for j in range(p):
        s = sum(A[p-j:p])
        # interval(alpha, df, loc=0, scale=1)
        x_2 = chi2.isf(0.05, (p-j-1)*(p-j)/2)
        # Endpoints of the range that contains alpha percent of the distribution
        S = N*s
        if S > x_2:
            return p-j
        else:
            continue

print(XuguanTest1(x1,y1))
print(XuguanTest1(x1,y2))


# In[16]:


def XuguanTest2(X,y,H):
    n,p=np.shape(X)
    cov_x = np.cov(X, rowvar=False)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    Z = np.matmul(X-np.mean(X,0), r)
    #y = y - np.mean(y)

    width = (np.max(y) - np.min(y)) / H

    V_hat = np.zeros([p,p])
    for h in range(H):
        h_index = np.logical_and(np.min(y)+h*width <= y, y < np.min(y)+(h+1)*width)
        ph_hat = np.mean(h_index)
        if ph_hat == 0:
            continue
        mh = np.mean(Z[h_index, :], axis=0)
        V_hat += ph_hat * np.matmul(mh[:, np.newaxis], mh[np.newaxis, :])
    
    #print(V_hat)
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)

    A = np.sort(eigenvalues)
    A = A[::-1]
    #print(A)
    for j in range(p):
        s = 20*dimension*sum(A[p-j:p])
        # interval(alpha, df, loc=0, scale=1)
        x_2 = chi2.isf(0.05, (p-j-1)*(H-j-2)/2)
        # Endpoints of the range that contains alpha percent of the distribution
        if s > x_2:
            return p-j
        else:
            continue

print(XuguanTest2(x1.T,y1.T,10))
print(XuguanTest2(x1.T,y2.T,10))


# In[17]:




#设置参数  c表示每一个切片中y的数目 
#tao -->1 的特征值数目
#k 降维结果
# n 样本个数  p特征数

#生成x，y
def gassiandata1(p,n):
    mean = np.zeros(p)
    cov = np.identity(p)
    data = np.round(np.random.multivariate_normal(mean,cov,n),3)
    return data
#example 1
x=gassiandata1(10,400)
y=x[0,:]*(x[1,:]+x[2,:]+1)

c= 10
n,p = np.shape(x)  



H = (n+c-1)/c 
H = math.floor(H)

sumx = np.zeros([p,p])
for h in range(H):
    for j in range(c):
        
        sumx = sumx + np.dot(np.mat(x[c*h+j,:]-1/c*sum(item for item in x[c*h:c*h+c-1 ,:])).T, np.mat(x[c*h+j,:]-1/c*sum(item for item in x[c*h:c*h+c-1 ,:])))

gama_p = 1/(H*(c-1)) * sumx 

sigma_x = np.cov(x.T)-gama_p
fi = sigma_x + np.eye(p) 

theta , eign = np.linalg.eig(fi)

theta = sorted(theta,reverse=True)

tao = 0
    
for i in range(p):
    if theta[i] > 1:
        tao = tao+1

theta_eva = np.log(theta) + 1 - theta

#C_n = 1/c * (0.5*np.log(n) + 0.1*pow(n,1/3))/2
#G_k = lambda k: n/2 * sum(item for item in theta_eva[min(tao,k):p]) -  C_n*k*(k+1)/p
#C_n = 10
C_n = pow(n,1/2)
G_k = lambda k: n/2 * sum(item for item in theta_eva[0:k])/sum(item for item in theta_eva[0:p]) -  C_n*k*(k+1)/p


for i in range(p):
    print(G_k(i+1),i)


# In[18]:


#example 1
beta11=np.mat([1,0,0,0,0,0,0,0,0,0])
beta12=np.mat([0,1,1,0,0,0,0,0,0,0])
sir1=SIR(x1.T,y1.T,10,2)
#sir1=pHd(x1,y1,2)
R11=np.dot(np.dot(beta11,np.cov(x1)),sir1[:,0])**2/(np.dot(np.dot(beta11,np.cov(x1)),beta11.T))*(np.dot(np.dot(sir1[:,0].T,np.cov(x1)),sir1[:,0]))
R12=np.dot(np.dot(beta11,np.cov(x1)),sir1[:,1])**2/(np.dot(np.dot(beta11,np.cov(x1)),beta11.T))*(np.dot(np.dot(sir1[:,1].T,np.cov(x1)),sir1[:,1]))
print(sorted([R11,R12])[1])
R13=np.dot(np.dot(beta12,np.cov(x1)),sir1[:,0])**2/(np.dot(np.dot(beta12,np.cov(x1)),beta12.T))*(np.dot(np.dot(sir1[:,0].T,np.cov(x1)),sir1[:,0]))
R14=np.dot(np.dot(beta12,np.cov(x1)),sir1[:,1])**2/(np.dot(np.dot(beta12,np.cov(x1)),beta12.T))*(np.dot(np.dot(sir1[:,1].T,np.cov(x1)),sir1[:,1]))
print(sorted([R13,R14])[1])


# In[19]:


#example2
beta21=np.mat([1,0,0,0,0,0,0,0,0,0])
beta22=np.mat([0,1,1,0,0,0,0,0,0,0])
beta23=np.mat([0,0,0,1,1,0,0,0,0,0])
sir2=SIR(x1.T,y2.T,10,3)
#sir2=pHd(x1,y2,3)

R21=np.dot(np.dot(beta21,np.cov(x1)),sir2[:,0])**2/(np.dot(np.dot(beta21,np.cov(x1)),beta21.T))*(np.dot(np.dot(sir2[:,0].T,np.cov(x1)),sir2[:,0]))
R22=np.dot(np.dot(beta21,np.cov(x1)),sir2[:,1])**2/(np.dot(np.dot(beta21,np.cov(x1)),beta21.T))*(np.dot(np.dot(sir2[:,1].T,np.cov(x1)),sir2[:,1]))
R23=np.dot(np.dot(beta21,np.cov(x1)),sir2[:,2])**2/(np.dot(np.dot(beta21,np.cov(x1)),beta21.T))*(np.dot(np.dot(sir2[:,2].T,np.cov(x1)),sir2[:,2]))
print(sorted([R21,R22,R23])[2])
R24=np.dot(np.dot(beta22,np.cov(x1)),sir2[:,0])**2/(np.dot(np.dot(beta22,np.cov(x1)),beta22.T))*(np.dot(np.dot(sir2[:,0].T,np.cov(x1)),sir2[:,0]))
R25=np.dot(np.dot(beta22,np.cov(x1)),sir2[:,1])**2/(np.dot(np.dot(beta22,np.cov(x1)),beta22.T))*(np.dot(np.dot(sir2[:,1].T,np.cov(x1)),sir2[:,1]))
R26=np.dot(np.dot(beta22,np.cov(x1)),sir2[:,2])**2/(np.dot(np.dot(beta22,np.cov(x1)),beta22.T))*(np.dot(np.dot(sir2[:,2].T,np.cov(x1)),sir2[:,2]))
print(sorted([R24,R25,R26])[2])
R27=np.dot(np.dot(beta23,np.cov(x1)),sir2[:,0])**2/(np.dot(np.dot(beta23,np.cov(x1)),beta23.T))*(np.dot(np.dot(sir2[:,0].T,np.cov(x1)),sir2[:,0]))
R28=np.dot(np.dot(beta23,np.cov(x1)),sir2[:,1])**2/(np.dot(np.dot(beta23,np.cov(x1)),beta23.T))*(np.dot(np.dot(sir2[:,1].T,np.cov(x1)),sir2[:,1]))
R29=np.dot(np.dot(beta23,np.cov(x1)),sir2[:,2])**2/(np.dot(np.dot(beta23,np.cov(x1)),beta23.T))*(np.dot(np.dot(sir2[:,2].T,np.cov(x1)),sir2[:,2]))
print(sorted([R27,R28,R29])[2])


# In[ ]:





# In[ ]:




