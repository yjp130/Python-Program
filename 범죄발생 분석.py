#!/usr/bin/env python
# coding: utf-8

# In[26]:


import platform

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [12,6]


# In[5]:


df=pd.read_csv("C:/Users/ktm/Desktop/Python-Program/public_crime_stat_2016.csv", encoding='EUC-KR')


# In[11]:


df.head(10)


# In[13]:


df.describe()


# In[16]:


df_index=pd.Series(df['계'].values, index=df['범죄중분류'].values)
df_index


# In[17]:


df_index.values


# In[19]:


df_index.values[2]


# In[20]:


type(df_index.values[2])


# ## 문자열을 정수형으로 변환

# In[21]:


df_index.values


# In[27]:


df_index.plot(kind='bar');


# In[29]:


plt.rcParams["figure.figsize"]=[14,8]


# In[30]:


df_index.plot(kind='bar');


# # 범죄대분류 별 건수

# In[47]:


get_ipython().run_line_magic('pinfo', 'pd.Series')


# In[48]:


df_crime=pd.Series(index=df['범죄대분류'].values,data=df['계'].values)
df_crime


# In[46]:


i=0
rows=[]
for i in range(len(df)):
    element=df['범죄대분류'][i], df['계'][i]
    rows.append(element)
    print(element)


# In[56]:


def get_crime_cnt(d_frame):
    crime_cnt = {}
    for idx in range(len(d_frame.index)):
        # print(idx, d_frame.index[idx], d_frame[idx], end="  \t =>")
        # print(idx, crime_cnt)

        crime = d_frame.index[idx]
        count  = d_frame[idx]

        if crime_cnt.get(crime):
            crime_cnt[crime] += count
        else:
            crime_cnt[crime] = count
    
    return crime_cnt
    
crime_dict = get_crime_cnt(df_crime)
crime_dict


# In[66]:


df.groupby("범죄대분류").agg({"계":"sum"})


# In[68]:


crime_stat=Series(crime_dict)
crime_stat.name="범죄발생수"
crime_stat


# In[69]:


plt.rcParams["figure.figsize"]=[15,15]
crime_stat.plot(kind='pie', title='2016년 범죄발생 현황');


# In[ ]:




