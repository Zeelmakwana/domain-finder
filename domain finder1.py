#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[ ]:


df= pd.read_csv('E:\my projects\domain finder\domain.csv')
df.head()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df['lang1'].value_counts().plot(kind='bar')


# In[ ]:



x = df["domain"]
y = df['lang1']
plt.xticks(rotation='vertical')
plt.plot(x, y)

plt.show()


# In[ ]:


X = df.drop(columns=['domain'])

y = df['domain']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1)


# In[ ]:


model=LogisticRegression()
model.fit(X_train,y_train)

# In[ ]:


i=(20,22,23)
arr=np.array(i)
rs=arr.reshape(1,-1)
p=model.predict(rs)
p


# In[ ]:




