#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("C:/Users/ktm/Desktop/Titanic/train.csv")


# In[4]:


df.head(10)


# In[5]:


df.groupby("Sex").size()


# In[6]:


df.groupby("Pclass").size()


# In[9]:


df.groupby(["Sex","Pclass"]).size()


# In[10]:


import math


# In[12]:


age_series=df.Age.dropna().apply(lambda age: math.floor(age/10)*10)
age_series.name="Age_Group"


# In[14]:


df=pd.concat(
[df, age_series],
axis=1,
)
df.head(10)


# In[15]:


age_group_df=df.Age_Group.fillna("확인불명")


# In[16]:


df.Age_Group.head(10)


# In[17]:


age_group_df=df.groupby("Age_Group").size()
age_group_df


# In[18]:


df.groupby("Age_Group").size()


# In[24]:


df.groupby("Sex").agg({"Survived": "mean"})


# In[25]:


df.groupby("Sex").agg({"Survived": "sum"})


# #  pd.crosstab : Compute a simple cross-tabulation of two (or more) factors.

# In[30]:


pd.crosstab(
    df.Sex,
    df.Pclass,
    margins=True,
)


# ## df.pivot_table : Create a spreadsheet-style pivot table as a DataFrame

# In[34]:


df.pivot_table(
    "Survived",
    "Sex",
    "Pclass",
  
)


# In[38]:


df.groupby(["Pclass", "Sex"]).size().unstack(1)


# In[40]:


temp_df = df.groupby(["Pclass", "Sex"]).size().unstack(0)
temp_df


# In[41]:


# temp_df.sum(axis=1)
result_df = temp_df.div(temp_df.sum(axis=1), axis=0)
# 비율을 연산하는 과정                .sum() => .div()
result_df


# ## 데이터 시각화

# In[52]:


from matplotlib import font_manager, rc

font_name=font_manager.FontProperties(fname="C:/Windows/Fonts/MALGUN.TTF").get_name()
rc('font',family=font_name)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


plt.rcParams['figure.figsize']=(12,8)


# In[54]:


bar_plot=result_df.plot.bar();


# In[55]:


result_df.plot(kind="barh",title="Titanic");


# In[56]:


result_df.plot(kind="barh", title="Titanic", stacked=True);


# In[75]:


age_group_df.index


# In[77]:


plt.rcParams['figure.figsize'] = (12,12)

labels = list(age_group_df.index)
labels = ['10세미만', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대']
sizes  = list(age_group_df.values)
explode = (0, 0, 0.1, 0, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:




