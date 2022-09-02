#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.linalg import sqrtm
import math
from scipy.stats import chi2


# In[2]:


def save(X, y, H, K):

    cov_x = np.cov(X, rowvar=False)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    Z = np.matmul(X - np.mean(X, axis=0), r)

    width = (np.max(y) - np.min(y)) / H

    V_hat = np.zeros([X.shape[1], X.shape[1]])
    for h in range(H):
        h_index = np.logical_and(np.min(y) + h * width <= y, y < np.min(y) + (h + 1) * width)
        ph_hat = np.mean(h_index)
        #print(ph_hat)
        if ph_hat == 0:
            continue
        meanh=np.mat(np.mean(Z[h_index, :],0))
        #print(np.shape(meanh))
        temp = np.eye(X.shape[1]) - np.matmul((Z[h_index, :]-meanh).T, (Z[h_index, :]-meanh)) / (ph_hat*X.shape[0])
        V_hat += ph_hat*np.matmul(temp, temp)

    # 特征值和特征向量（列向量）
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argsort(-eigenvalues)
    K_largest_eigenvectors = eigenvectors[:, K_index[:K]]
    edr_est = np.matmul(r, K_largest_eigenvectors)

    return edr_est


# In[576]:


def gassiandata(p,n):
    mean = np.zeros(p)
    cov = np.identity(p)
    data = np.round(np.random.multivariate_normal(mean,cov,n),3)
    return data
#example 1
sample=400
dimension=10
x1=gassiandata(dimension,sample).T
noise1=np.random.normal(0,0.5,sample)


# In[577]:


def STest(X,y,H):
    n,p=np.shape(X)
    cov_x = np.cov(X, rowvar=False)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    Z = np.matmul(X-np.mean(X,0), r)

    width = (np.max(y) - np.min(y)) / H

    V_hat = np.zeros([X.shape[1], X.shape[1]])
    for h in range(H):
        h_index = np.logical_and(np.min(y) + h * width <= y, y < np.min(y) + (h + 1) * width)
        ph_hat = np.mean(h_index)
        if ph_hat == 0:
            continue
        meanh=np.mat(np.mean(Z[h_index, :],0))
        temp = np.eye(X.shape[1]) - np.matmul((Z[h_index, :]-meanh).T, (Z[h_index, :]-meanh)) / (ph_hat*X.shape[0])
        V_hat += ph_hat*np.matmul(temp, temp)
    
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)

    for j in range(p):
        K_index= np.argsort(eigenvalues)
        #print(K_index)
        seita = eigenvectors[:, K_index[:j+1]]
        #print(seita)
        Tn=0
        for i in range(H):
            i_index = np.logical_and(np.min(y) + i * width <= y, y < np.min(y) + (i + 1) * width)
            ph = np.mean(i_index)
            if ph == 0:
                continue
            else:
                #Ak = np.eye(p) - np.cov(Z[i_index, :].T)/(ph*n)
                meani=np.mat(np.mean(Z[i_index, :],0))
                Ak = np.eye(p) - np.matmul((Z[i_index, :]-meani).T, (Z[i_index, :]-meani)) / (ph*n)
                mat = np.matmul(np.matmul(seita.T,Ak),seita)
                Tn=Tn+np.trace(np.dot(mat,mat))
        s =0.5*n*Tn
        x_2 = chi2.isf(0.05, (H-1)*j*(j+1)/2)

        if s > x_2:
            return p-j-6
        else:
            continue


# In[578]:


y1 = x1.T[:,0]*(x1.T[:,1]+x1.T[:,2]+1)+noise1
y2 =x1.T[:,0]*(x1.T[:,1]+x1.T[:,2]+2)+np.power((x1.T[:,3]+x1.T[:,4]+2),3)+noise1
#print(STest(x1,y1,10))
print(STest(x1.T,y1,10))
print(STest(x1.T,y2,10))


# In[595]:


slices=10
save1=save(x1.T,y1,slices,2)
for j in range(2):
    save1[:,j]=save1[:,j]/np.sqrt(np.dot(save1[:,j].T,save1[:,j]))
    if save1[0,j]<0:
        save1[:,j]=(-1)*save1[:,j]
    else:
        continue
        
for i in range(99):
    x11=gassiandata(dimension,sample)
    noise11=np.random.normal(0,0.5,sample)
    y11 = x11[:,0]*(x11[:,1]+x11[:,2]+1)+noise11
    temp1=save(x11,y11,slices,2)
    for j in range(2):
        temp1[:,j]=temp1[:,j]/np.sqrt(np.dot(temp1[:,j].T,temp1[:,j]))
        if temp1[0,j]<0:
            temp1[:,j]=(-1)*temp1[:,j]
        else:
            continue
    save1=save1+temp1
save1=save1/100
print(save1)


# In[596]:


save2=save(x1.T,y2,slices,3)
for j in range(3):
    save2[:,j]=save2[:,j]/np.sqrt(np.dot(save2[:,j].T,save2[:,j]))
    if save2[0,j]<0:
        save2[:,j]=(-1)*save2[:,j]
    else:
        continue
        
for i in range(99):
    x12=gassiandata(dimension,sample)
    noise12=np.random.normal(0,0.5,sample)
    y12 =x12[:,0]*(x12[:,1]+x12[:,2]+2)+np.power((x12[:,3]+x12[:,4]+2),3)+noise12
    temp2=save(x12,y12,slices,3)
    for j in range(3):
        temp2[:,j]=temp2[:,j]/np.sqrt(np.dot(temp2[:,j].T,temp2[:,j]))
        if temp2[0,j]<0:
            temp2[:,j]=(-1)*temp2[:,j]
        else:
            continue
    save2=save2+temp2
save2=save2/100
print(save2)


# In[597]:


#example 1
beta11=np.hstack((np.mat([1]),np.mat(np.zeros(dimension-1))))
beta12=np.hstack((np.mat([0,1,1]),np.mat(np.zeros(dimension-3))))
R11=np.dot(np.dot(beta11,np.cov(x1)),save1[:,0])**2/(np.dot(np.dot(beta11,np.cov(x1)),beta11.T))*(np.dot(np.dot(save1[:,0].T,np.cov(x1)),save1[:,0]))
R12=np.dot(np.dot(beta11,np.cov(x1)),save1[:,1])**2/(np.dot(np.dot(beta11,np.cov(x1)),beta11.T))*(np.dot(np.dot(save1[:,1].T,np.cov(x1)),save1[:,1]))
print(sorted([R11,R12])[1])
R13=np.dot(np.dot(beta12,np.cov(x1)),save1[:,0])**2/(np.dot(np.dot(beta12,np.cov(x1)),beta12.T))*(np.dot(np.dot(save1[:,0].T,np.cov(x1)),save1[:,0]))
R14=np.dot(np.dot(beta12,np.cov(x1)),save1[:,1])**2/(np.dot(np.dot(beta12,np.cov(x1)),beta12.T))*(np.dot(np.dot(save1[:,1].T,np.cov(x1)),save1[:,1]))
print(sorted([R13,R14])[1])


# In[598]:


#example2
beta21=np.hstack((np.mat([1]),np.mat(np.zeros(dimension-1))))
beta22=np.hstack((np.mat([0,1,1]),np.mat(np.zeros(dimension-3))))
beta23=np.hstack((np.mat([0,0,0,1,1]),np.mat(np.zeros(dimension-5))))

R21=np.dot(np.dot(beta21,np.cov(x1)),save2[:,0])**2/(np.dot(np.dot(beta21,np.cov(x1)),beta21.T))*(np.dot(np.dot(save2[:,0].T,np.cov(x1)),save2[:,0]))
R22=np.dot(np.dot(beta21,np.cov(x1)),save2[:,1])**2/(np.dot(np.dot(beta21,np.cov(x1)),beta21.T))*(np.dot(np.dot(save2[:,1].T,np.cov(x1)),save2[:,1]))
R23=np.dot(np.dot(beta21,np.cov(x1)),save2[:,2])**2/(np.dot(np.dot(beta21,np.cov(x1)),beta21.T))*(np.dot(np.dot(save2[:,2].T,np.cov(x1)),save2[:,2]))
print(sorted([R21,R22,R23])[2])
R24=np.dot(np.dot(beta22,np.cov(x1)),save2[:,0])**2/(np.dot(np.dot(beta22,np.cov(x1)),beta22.T))*(np.dot(np.dot(save2[:,0].T,np.cov(x1)),save2[:,0]))
R25=np.dot(np.dot(beta22,np.cov(x1)),save2[:,1])**2/(np.dot(np.dot(beta22,np.cov(x1)),beta22.T))*(np.dot(np.dot(save2[:,1].T,np.cov(x1)),save2[:,1]))
R26=np.dot(np.dot(beta22,np.cov(x1)),save2[:,2])**2/(np.dot(np.dot(beta22,np.cov(x1)),beta22.T))*(np.dot(np.dot(save2[:,2].T,np.cov(x1)),save2[:,2]))
print(sorted([R24,R25,R26])[2])
R27=np.dot(np.dot(beta23,np.cov(x1)),save2[:,0])**2/(np.dot(np.dot(beta23,np.cov(x1)),beta23.T))*(np.dot(np.dot(save2[:,0].T,np.cov(x1)),save2[:,0]))
R28=np.dot(np.dot(beta23,np.cov(x1)),save2[:,1])**2/(np.dot(np.dot(beta23,np.cov(x1)),beta23.T))*(np.dot(np.dot(save2[:,1].T,np.cov(x1)),save2[:,1]))
R29=np.dot(np.dot(beta23,np.cov(x1)),save2[:,2])**2/(np.dot(np.dot(beta23,np.cov(x1)),beta23.T))*(np.dot(np.dot(save2[:,2].T,np.cov(x1)),save2[:,2]))
print(sorted([R27,R28,R29])[2])


# In[ ]:





# In[ ]:




