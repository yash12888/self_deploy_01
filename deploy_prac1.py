#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl


# In[12]:


df = pd.DataFrame({
    'Roll no':[1,2,3,4,5,6,7,8,9,10],
    'Marks':[84,66,95,42,36,78,95,46,26,72],
    'degree':['bsc','msc','btech','btech','mtech','bsc','bsc','msc','btech','btech'],
    'salary':[480,562,236,556,148,258,336,1045,147,654]
})


# In[13]:


df


# In[14]:


# dropping roll no
df.drop(columns='Roll no',axis=1,inplace=True)


# In[15]:


# splitting X and y
X=df.drop(columns='salary',axis=1)
y=df.salary


# In[16]:


#feature scaling
from sklearn.preprocessing import MinMaxScaler
X_num=X.select_dtypes(include='number')
ss=MinMaxScaler()
X_num_scaled=ss.fit_transform(X_num)
X_num_scaled=pd.DataFrame(X_num_scaled,columns=X_num.columns,index=X_num.index)


# In[17]:


X_cat=X.select_dtypes(include='object')
X_cat_encoded=pd.get_dummies(X_cat,dtype=int)


# In[18]:


X=pd.concat([X_num_scaled,X_cat_encoded],axis=1)


# In[19]:


X


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


model=LinearRegression()
model.fit(X,y)


# In[22]:


# save the trained model using pickle


# In[23]:


with open('model_lr.pkl','wb') as file:
    pkl.dump(model,file)


# In[ ]:




