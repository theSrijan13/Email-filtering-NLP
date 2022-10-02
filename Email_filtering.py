#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import os


# In[3]:


df=pd.read_csv(r'C:\Users\HP\Downloads\sms_spam.csv')
print(df)


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[8]:


df['Spam']=df['type'].apply(lambda x:1 if x=='spam' else 0)
df.head(5)


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.text,df.Spam,test_size=0.25)


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer


# In[12]:


from sklearn.naive_bayes import MultinomialNB


# In[14]:


from sklearn.pipeline import Pipeline
dlf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])


# In[15]:


dlf.fit(X_train,y_train)


# In[16]:


emails=[' Am also doing in cbe only. But have to pay.','complimentary 4 STAR Ibiza Holiday or £10,000 ...']


# In[17]:


dlf.predict(emails)


# In[20]:


dlf.score(X_test,y_test)


# In[ ]:




