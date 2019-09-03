#!/usr/bin/env python
# coding: utf-8

# In[4]:



import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [12,6]


# In[7]:


df = pd.read_csv('C:/Users/ktm/Desktop/Python-Program/public_old_buildings_20171016.csv', encoding='EUC-KR')
df.head(10)


# In[8]:


df.info()


# In[10]:


df.sort_values(by='준공일자', ascending=1).tail(10)


# In[11]:


df.columns


#  # 문자열인 준공일자를 Date 객체로 변환 

# In[13]:


df['준공일자']=pd.to_datetime(df['준공일자'], format='%Y%m%d')
df.head(10)


# #  준공일자 칼럼을 인덱스로 지정 

# In[14]:


df.set_index('준공일자', inplace=True)
df.head(10)


# In[15]:


df.info()


# In[16]:


df.info()


# In[17]:


df.index.year


# In[18]:


df['준공년']=df.index.year
df.head(10)


# # # 연도별 준공건수

# In[21]:


df_year=df['준공년'].value_counts()
df_year.head(10)


# In[22]:


df_year.index


# In[23]:


plt.rcParams["figure.figsize"]=[20,12]


# In[24]:


df_year.plot(kind='bar')


# # # 연도별로 정렬

# In[25]:


df_year=df_year.sort_index()
df_year.head(10)


# In[26]:


df_year.plot(kind='bar')


# In[27]:


df.head()


# In[34]:


df['시설물소재지'].values[:30]


# In[29]:


tmp='부산광역시 부산진구 부암동'


# In[30]:


tmp.find(' ')


# In[31]:


tmp2=tmp[0:(tmp.find(' '))]


# In[32]:


tmp2


# In[35]:


tmp2=tmp[:(tmp.find(' '))]


# In[36]:


tmp2


# In[37]:


len(df)


# In[38]:


df['시설물소재지'].values[1][0:5]


# In[39]:


tmp3=df['시설물소재지'].values[1].find(' ')


# In[40]:


df['시설물소재지'].values[1][0:tmp3]


# In[41]:


df['state']=' '
df.head(5)


# ### 시설물소재지를 시/도별로 구분하는 state 컬럼 추가

# In[44]:


for n in np.arange(len(df)):
    endN=df['시설물소재지'].values[n].find(' ')
    df['state'].values[n]=df['시설물소재지'].values[n][0:endN]


# In[45]:


df.head(10)


# In[62]:


df['state'].unique()


# ## 시/도별 준공건수 

# In[64]:


df['state'].value_counts()


# In[65]:


plt.rcParams["figure.figsize"]=[15,15]
df_state.plot(kind='pie')
plt.show()


# In[66]:


plt.rcParams["figure.figsize"] = [12,6]
df_state.plot(kind='barh')
plt.show()


# In[67]:


df_state.sort_values(ascending=False).head(5)


# In[68]:


df_state.sort_values(ascending=True).head(5)


# In[70]:



plt.figure(figsize=(20,20))

plt.subplot(211) 
plt.title('노후공공시설물 설립년도별 현황')
df_year.plot(kind='bar')

plt.subplot(223) 
plt.title("시도별 노후공공시설 현황")
df_state.plot(kind='pie')

plt.subplot(224)
plt.title("시도별 노후공공시설 순위 TOP10")
df_state.sort_values(ascending=False).head(10).plot(kind='barh', color='y')

plt.show()


# In[ ]:




