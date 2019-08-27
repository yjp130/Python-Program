#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re 


# In[7]:


with open('C:/Users/ktm/Anaconda_src/sample_emails.txt', encoding='utf-8') as fp:
    text = fp.read()


# In[8]:


text

def write_txt(filepath):
    
    count=1
    data=[]
    
    valid_email_end=['com','gov','co.kr','net']
    
    result = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    
    for email in result:
        
        temp=email.split('.')
        
        if temp[-1] in valid_email_end:
            data.append(email+'\n')
            
                
            
            
    f=open(filepath, 'w')
    f.writelines(data)
    f.close()
    
    ret="완료"
    
    return ret
# In[20]:


filepath='C:/Users/ktm/Anaconda_src/emails_list.txt'
write_txt(filepath)


# In[23]:


from IPython.display import Image
from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# In[25]:


df = DataFrame([[1.4, np.nan], [7.1, -4.5],[np.nan, np.nan], [0.75,-1.3]],
               index=['a','b','c','d'],
               columns=['one','two'])
df


# In[26]:


df.sum()


# In[28]:


df.sum(axis=1)


# In[29]:


df.mean(axis=1, skipna=False)


# In[31]:


df.mean(axis=1, skipna=True)


# In[33]:


df


# In[34]:


df.idxmax()


# # Unique values, value counts, and membership

# In[35]:


obj=Series(['c','a','d','a','a','b','b','c','c'])


# In[36]:


uniques=obj.unique()
uniques


# In[37]:


obj.value_counts()


# In[38]:


pd.value_counts(obj.values, sort=False)


# In[39]:


pd.value_counts(obj.values, sort=True)


# In[40]:


mask=obj.isin(['b','c'])
mask


# In[41]:


obj[mask]


# In[42]:


data=DataFrame({'Qu1':[1,3,4,3,4],
               'Qu2':[2,3,1,2,3],
               'Qu3':[1,5,2,4,4]})
data


# In[43]:


result=data.apply(pd.value_counts).fillna(0)
result


# # Handling Missing data

# In[44]:


string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data


# In[45]:


string_data.isnull()


# In[46]:


string_data[0] = None
string_data.isnull()


# # Filtering out missing data # 

# In[47]:


from numpy import nan as NA
data = Series([1, NA, 3.5, NA, 7])
data.dropna()


# In[48]:


data[data.notnull()]


# In[49]:


data = DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
data


# In[50]:


cleaned = data.dropna()
cleaned


# In[51]:


# 모든 값이 NA인 로우만 제외
data.dropna(how='all')


# In[52]:


data[4] = NA
data


# In[53]:


data.dropna(axis=1, how='all')


# In[54]:


df = DataFrame(np.random.randn(7, 3))
df


# In[55]:


df.iloc[:4, 1] = NA 
df.iloc[:2, 2] = NA
df


# In[56]:


df.dropna(thresh=3)


# In[57]:


df.dropna(thresh=2)


# In[58]:


df.dropna(thresh=1)


# In[59]:


df


# In[60]:


df.fillna(0)


# In[61]:


# fillna에 사전값을 넣어서 각 칼럼마다 다른값을 채워넣을수도 있다.
df.fillna({1: 0.5, 3: -1})


# In[62]:


# fillna는 값을 채워 넣은 객체의 참조를 반환한다.
_ = df.fillna(0, inplace=True)
df


# In[63]:


df = DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA 
df.iloc[4:, 2] = NA
df


# In[64]:


df.fillna(method='ffill')


# In[65]:


df.fillna(method='ffill', limit=2)


# In[66]:


data = Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())


# # Merge

# In[68]:


df=pd.DataFrame(np.random.randn(10,4))
df


# In[69]:


pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)


# In[70]:


pieces = [df[:3], df[3:7], df[7:]]


# In[71]:


pd.concat(pieces)


# # Join

# In[72]:


left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})


# In[73]:


right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})


# In[74]:


left


# In[75]:


right


# In[76]:


pd.merge(left, right, on='key')


# In[77]:


left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})


# In[78]:


right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})


# In[79]:


left


# In[80]:


right


# In[81]:


pd.merge(left, right, on='key')


# # Grouping

# In[82]:


df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                             'foo', 'bar', 'foo', 'foo'],
                       'B': ['one', 'one', 'two', 'three',
                             'two', 'two', 'one', 'three'],
                       'C': np.random.randn(8),
                       'D': np.random.randn(8)})


# In[83]:


df


# In[84]:


df.groupby('A').sum()


# # matplotlib inline 모드로 그래프 작성 
# 

# In[85]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[86]:


data_list = [ 0, 1, 5, 3, 4, 5 ]

plt.figure
plt.plot(data_list)
plt.show()


# # matplotlib notebook

# In[99]:


data_list = [ 0, 1, 5, 3, 4, 5 ]

plt.figure
plt.plot(data_list)
plt.show()


# # numpy 배열 그리기

# In[103]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/MALGUN.TTF").get_name()
rc('font', family=font_name)

plt.rcParams['figure.figsize'] = (10,6)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[104]:


t = np.arange(0, 2*3.14, 0.01)


# In[105]:


plt.figure
plt.plot(t)
plt.show()


# # numpy를 이용하여 시간축과 함수 그래프

# In[106]:


import math

PI = math.pi
PI


# In[107]:


t = np.arange(0, 2*PI, 0.01)
y = np.sin(t)


# In[108]:


plt.figure(figsize=(6,4)) # figure 크기 조절하기
plt.plot(t, y)
plt.show()


# In[109]:


plt.figure(figsize=(6,4))
plt.plot(t, y)
plt.grid() # 그리드 적용하기
plt.show()


# In[110]:


plt.figure(figsize=(6,4))
plt.plot(t, y)
plt.grid()
plt.xlabel('time')       # x축 라벨 적용하기
plt.ylabel('Amplitude')  # y축 라벨 적용하기
plt.show()


# In[111]:


plt.figure(figsize=(6,4))
plt.plot(t, y)
plt.grid()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')  # 그래프의 타이틀 적용하기
plt.show()


# In[112]:


dy = np.diff(y)     # numpy의 차분 함수 사용하기
dy[:10]


# In[113]:


dy = np.insert(dy, 0, 0)/0.01   # 차분의 특성으로 제일 앞에 의미없는 값 입력해두기
dy[:10]


# In[114]:


a = np.array([[1, 1], [2, 2], [3, 3]])
a


# In[115]:


np.insert(a, 2, 5)


# In[116]:


np.insert(a, 2, 5, axis=1)


# # 변수 상태 확인 명령어 

# In[117]:


get_ipython().run_line_magic('whos', '')


# # 두 개의 그래프 작성 

# In[118]:


plt.figure(figsize=(6,4))
plt.plot(t, y)
plt.plot(t, dy)
plt.grid()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


# # 범례를 적용하기 위해 label을 적용하고 legend 실행

# In[119]:


plt.figure(figsize=(6,4))
plt.plot(t, y, label='sin')
plt.plot(t, dy, label='cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


# In[121]:


plt.figure(figsize=(6,4))
plt.plot(t, y, 'c', label='sin')
plt.plot(t, dy, 'g', label='cos')   # color : ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w' ]
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


# In[123]:


plt.figure(figsize=(6,4))
plt.plot(t, y, lw=5, label='sin')     # linewidth
plt.plot(t, dy, 'r', label='cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


# In[124]:


plt.figure(figsize=(6,4))
plt.plot(t, y, lw=3, label='sin')
plt.plot(t, dy, 'r', label='cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.xlim(0, 3.14)    # set the xlim to xmin, xmax
plt.show()


# In[125]:


plt.figure(figsize=(6,4))
plt.plot(t, y, lw=3, label='sin')
plt.plot(t, dy, 'r', label='cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.ylim(-1.2/2, 1.2/2)    # set the ylim to ymin, ymax
plt.show()


# In[126]:


# 변수 초기화
get_ipython().run_line_magic('reset', '')


# # 두 개 이상의 그래프 그리기의 다른 방법

# In[127]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import font_manager, rc

# font_name = font_manager.FontProperties(fname="/Library/Fonts/AppleGothic.ttf").get_name()
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/MALGUN.TTF").get_name()
rc('font', family=font_name)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[128]:


plt.rcParams['figure.figsize'] = (12,6)


# In[132]:


t = np.arange(0, 5, 1.0)     # 0.1, 0.5, 1.0

plt.figure
plt.plot(t, t, t, t**2, t, t**3)
plt.show()


# In[133]:



t = np.arange(0, 5, 0.1)     # 0.1, 0.5, 1.0

plt.figure
plt.plot(t, t, t, t**2, t, t**3)
plt.show()


# # 마커 적용

# In[134]:


t = np.arange(0, 5, 0.5)

plt.figure
plt.plot(t, t,    'r--')
plt.plot(t, t**2, 'bs' )
plt.plot(t, t**3, 'g^' )
plt.show()


# In[135]:


t = np.arange(0, 5, 0.5)

plt.figure
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()


# In[136]:


t = np.arange(0, 5, 0.5)

plt.figure
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()


# In[137]:


t = np.arange(0, 5, 0.5)

fig1 = plt.figure(1)
plt1 = plt.plot(t, t**2, 'rs')

fig2 = plt.figure(2)
plt2 = plt.plot(t, t**3, 'b^')

plt.show()


# In[139]:


import matplotlib.pyplot as plt

t = [0, 1, 2, 3, 4, 5, 6] 
y = [1, 4, 5, 8, 9, 5, 3]


# In[140]:


plt.figure(figsize=(10,4))
plt.plot(t,y,color='green')
plt.show()


# In[141]:


plt.figure(figsize=(10,4))
plt.plot(t,y, color='green', linestyle='dashed')
plt.show()


# In[142]:


plt.figure(figsize=(10,4))
plt.plot(t,y, color='green', linestyle='dashed', marker='o')
plt.show()


# In[143]:


import matplotlib.pyplot as plt
import numpy as np


# In[144]:


t = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])


# In[145]:


plt.figure(figsize=(10,4))
plt.scatter(t,y)
plt.show()


# In[146]:


colormap = t

plt.figure(figsize=(10,4))
plt.scatter(t,y, marker='>')
plt.show()


# In[147]:


colormap = t

plt.figure(figsize=(10,4))
plt.scatter(t,y, s=50, c=colormap, marker='>')
plt.show()


# In[148]:


colormap = t

plt.figure(figsize=(12.5,4))
plt.scatter(t,y, s=50, c=colormap, marker='>')
plt.colorbar()
plt.show()


# In[149]:


colormap = y

plt.figure(figsize=(12.5,4))
plt.scatter(t,y, s=50, c=colormap, marker='>')
plt.colorbar()
plt.show()


# # bar 그래프 작성 

# In[150]:


plt.figure(figsize=(10,4))
plt.bar(t,y)
plt.show()


# In[152]:


plt.figure(figsize=(10,4))
plt.bar(t,y, width = 0.5, color='r')    # plt.bar(left, height, width=0.8, bottom=None, hold=None, data=None, **kwargs)
plt.show()


# In[154]:


y1 = np.array([3,2,4,3,4, 9,8,7,9,8])


# In[164]:


plt.figure(figsize=(10,4))
plt.bar(t,y, color='r', width=0.3, label='apple')
plt.bar(t + 0.2, y1, color='y', width=0.3,label='banana')
plt.xlabel('data')
plt.ylabel('mount')
plt.legend()
plt.show()


# In[165]:


plt.figure(figsize=(10,4))
plt.bar(t,y, color='r', width=0.3, label='apple')
plt.bar(t + 0.4, y1, color='y', width=0.3,label='banana')
plt.xlabel('data')
plt.ylabel('mount')
plt.legend()

plt.xticks(t+0.4, ('1Q','2Q','3Q','4Q','1Q','2Q','3Q','4Q','1Q','2Q'))   # set the locations of the xticks
plt.show()


# # 좌우 bar 그래프 효과 

# In[166]:


plt.figure(figsize=(12,4))
plt.barh(t,y, color='r', label='apple')
plt.barh(t, -y1, color='y', label='banana')
plt.xlabel('data')
plt.ylabel('mount')
plt.legend()
plt.show()


# In[167]:


plt.figure(figsize=(12,4))
plt.barh(t,y, color='r', label='apple')
plt.barh(t, -y1, color='y', label='banana')
plt.xlabel('data')
plt.ylabel('mount')
plt.legend()
plt.xticks([-10,-5,0,5,10],('10','5','0','5','10'))    # Get or set the *x*-limits of the current tick locations and labels.
plt.show()


# # pie 그래프 작성 

# In[174]:


# color=('b','g','r','c','m','y','k','w')    # colors : A sequence of matplotlib color args through which the pie chart
y


# In[175]:


plt.figure(figsize=(6,6))
plt.pie(y)
plt.show()


# In[176]:



label = ['Blue', 'Orange', 'Green','Red','Violet', 
         'Brown','Pink', 'Gray','Yello', 'Cyan']
plt.figure(figsize=(6,6))
plt.pie(y, labels=label)    # labels : A sequence of strings providing the labels for each wedge
plt.show()


# In[177]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes  = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[179]:


fig, ax = plt.subplots(figsize=(10, 12), subplot_kw=dict(aspect="equal"))

recipe = ["375 g 밀가루",
          "75 g  설탕",
          "250 g 버터",
          "300 g 딸기"]

data = [float(x.split()[0]) for x in recipe]
ingredients = [x.split()[-1] for x in recipe]


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients,
          title="주요성분",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=14, weight="bold")

ax.set_title("빵의 주요 성분")

plt.show()


# # 히그토그램 작성

# In[180]:


data= np.random.normal(5,3,1000)


# In[182]:


plt.figure(figsize=(10,4))
plt.plot(data)
plt.show()


# In[183]:


plt.figure(figsize=(10,4))
plt.hist(data)
plt.show()


# In[190]:


plt.figure(figsize=(10,4))
plt.hist(data,bins=20)
plt.show()


# In[193]:


plt.figure(figsize=(10,4))
plt.hist(data,bins=10, facecolor='red',
         alpha=0.4, histtype='stepfilled')
plt.hist(data,bins=20, facecolor='green',
        alpha=0.4, histtype='stepfilled')
plt.show()


# In[194]:


x=np.random.randn(10000)


# In[195]:


plt.figure(figsize=(10,4))
plt.plot(x)
plt.show()


# In[196]:


plt.figure(figsize=(10,4))
plt.hist(x,normed=1, bins=20)
plt.show()


# # box 그래프 작성

# In[197]:


s1 = np.random.normal(loc=0,  scale=1.0, size=1000)    # loc : Mean ("centre") of the distribution.
s2 = np.random.normal(loc=5,  scale=0.5, size=1000)    # scale : Standard deviation (spread or "width") of the distribution.
s3 = np.random.normal(loc=10, scale=2.0, size=1000)    # size : Output shape


# In[198]:


plt.figure(figsize=(10,4))
plt.plot(s1, label='s1')
plt.plot(s2, label='s2')
plt.plot(s3, label='s3')
plt.legend()
plt.show()


# In[199]:


plt.figure(figsize=(10,4))
plt.boxplot((s1, s2, s3))
plt.grid()
plt.show()


# # annotation 적용

# In[200]:


t = np.arange(0,5,0.01)
y = np.cos(2*np.pi*t)

plt.figure(figsize=(10,4))
plt.plot(t,y)
plt.annotate('local max', xy=(1,1), xytext=(1.5,1.2),
            arrowprops=dict(facecolor='black',shrink=0.05))
plt.ylim(-1.1,1.4)
plt.show()


# # subplot 적용

# In[201]:


plt.figure(figsize=(16,8))
plt.subplot(221)      # 2x2에 1번째,  subplot(nrows, ncols, plot_number)
plt.subplot(222)      # 2x2에 2번째 
plt.subplot(212)      # 2x1에 2번째

plt.show()


# In[203]:


plt.figure(figsize=(16,8))
plt.subplot(411)
plt.subplot(423)
plt.subplot(424)
plt.subplot(413)
# plt.subplot(414)
plt.subplot(427)
plt.subplot(428)

plt.show()


# In[204]:


t = np.arange(0,5,0.01)

plt.figure(figsize=(16,8))

plt.subplot(411)
plt.plot(t,np.sqrt(t))

plt.subplot(423)
plt.plot(t,t**2)

plt.subplot(424)
plt.plot(t,t**3)

plt.subplot(413)
plt.plot(t,np.sin(t))

plt.subplot(414)
plt.plot(t,np.cos(t))

plt.show()


# # Quiz. 현재 영화 상영작 평점을 그래프 작성
# 

# In[265]:


import matplotlib.pyplot as plt
from IPython.display import Image

from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# In[284]:


#관람객 평점 : [-0.25, 0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75]
#평론가 평점 : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
#네티즌 평점 : [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25, 9.25]

data= DataFrame([[-0.25, 0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75],
               [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
               [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25, 9.25]])

Audience_score=np.array([9.04, 9.01, 8.34, 9.16, 8.31, 6.00, 8.70, 9.11, 8.40, 9.22])
Expert_score=np.array([7.23, 6, 2.0, 4.50, 5.30, 4.57, 5.50, 6, 5, 6.0])
Netizen_score=np.array([8.48, 8.01, 7.31, 7.88, 6.08, 7.96, 7.38, 8.85, 7.17, 9.28])
name=['액시트','분노의 질주','변신','봉오동 전투','광대들',
     '커런트 워', '안녕 티라노', '마이펫','애프터','레드슈즈']
t=np.array([0,1,2,3,4,5,6,7,8,9])


# In[302]:


plt.figure(figsize=(10,6))

plt.bar(t,Audience_score ,color='r', width=0.2,  label='관람객 평점' )
plt.bar(t+0.3,Expert_score, color='y', width=0.2, label='평론가 평점')
plt.bar(t+0.6,Netizen_score, color='g', width=0.2, label='네티즌 평점')
plt.xlabel('영화제목')
plt.ylabel('영화평점')
plt.title('현재 상영작 평점')
plt.legend()
plt.xticks(t, ('액시트','분노의 질주','변신','봉오동 전투','광대들',
     '커런트 워', '안녕 티라노', '마이펫','애프터','레드슈즈'))  
plt.show()


# In[296]:


plt.figure(figsize=(6,6))
plt.pie(Audience_score, labels=name)
plt.show()


# In[298]:


sizes=[15,30,45,20,35,40,50,17,5,10]
explode=(0, 0.1, 0, 0,0,0,0,0,0,0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=name, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:




