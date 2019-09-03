#!/usr/bin/env python
# coding: utf-8

# # 서울시 공공데이터 분석

# In[14]:


# from images import bigpycraft_bda as bpc
from IPython.display import Image 

import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

plt.rcParams["figure.figsize"] = [12,6]

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


population_xls = 'C:/Users/ktm/Desktop/Python-Program/seoul_data/Report_seoul_population_2019_2Q.xls'


# In[6]:


df_pop_seoul = pd.read_excel('C:/Users/ktm/Desktop/Python-Program/seoul_data/Report_seoul_population_2019_2Q.xls',  encoding='utf-8')
df_pop_seoul.head(10)


# In[7]:


population=pd.read_excel(population_xls,
                        header=2,
                        usecols="B,C,D,G,J,N",
                        encoding="utf-8"
                        )
population.head()


# ## unique 체크후, 합계오 nan 부분을 drop

# In[8]:


population = population.drop([0])
population


# In[9]:


population['자치구'].unique()


# In[12]:


population.rename(columns = {'계':'인구수'}, inplace=True)
population.rename(columns = {'자치구':'구'}, inplace=True)
population.head()


# In[15]:


sns.regplot(x="인구수", y="세대", data=population);


# In[16]:


sns.regplot(x="인구수", y="65세이상고령자", data=population);


# In[17]:


import json
import folium
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


# In[19]:


geo_path = 'C:/Users/ktm/Desktop/Python-Program/seoul_data/skorea_municipalities_geo_simple.json'

geo_str = json.load(open(geo_path, encoding='utf-8'))
# geo_str


# In[20]:


guDat = pd.DataFrame({'gu':population['구'], 'counts':population['인구수']})
guDat.head()


# In[21]:


map = folium.Map(location=[37.5502, 126.982], zoom_start=11, tiles='Stamen Toner')
map


# In[22]:


map = folium.Map(location=[37.5502, 126.982], zoom_start=11, tiles='Stamen Toner')

map.choropleth(geo_data=geo_str,
              data=guDat,
              columns=['gu', 'counts'],
              fill_color='YlGnBu', #PuRd, YlGnBu
              key_on='feature.id')


# In[23]:


map


# In[24]:


guDat = pd.DataFrame({'gu':population['구'], 'counts':population['65세이상고령자']})

map = folium.Map(location=[37.5502, 126.982], zoom_start=11, tiles='Stamen Toner')

map.choropleth(geo_data=geo_str,
              data=guDat,
              columns=['gu', 'counts'],
              fill_color='PuRd', #PuRd, YlGnBu
              key_on='feature.id')


# In[25]:


map


# In[27]:


guDat = pd.DataFrame({'gu':population['구'], 'counts':population['세대당인구']})

map = folium.Map(location=[37.5502, 126.982], zoom_start=11, tiles='Stamen Toner')

map.choropleth(geo_data=geo_str,
              data=guDat,
              columns=['gu', 'counts'],
              fill_color='PuRd', #PuRd, YlGnBu
              key_on='feature.id')


# In[ ]:




