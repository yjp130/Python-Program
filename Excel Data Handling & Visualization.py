#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import re


# In[9]:


xls_file='/Users/ktm/Desktop/Python-Program/test_score.xlsx'


# In[11]:


df = pd.read_excel(xls_file)
df


# In[12]:


df=pd.read_excel(xls_file, sheet_name=1)
df


# In[13]:


df=pd.read_excel(xls_file, sheet_name='중간고사')
df


# In[15]:


df=pd.read_excel(xls_file, sheet_name='기말고사')
df


# In[16]:


df = pd.read_excel(xls_file, sheet_name='기말고사', index_col=0)
df


# In[17]:


df = pd.read_excel(xls_file, sheet_name='기말고사', index_col='학생')
df


# In[19]:


excel_exam_data1={
    '학생': ['A','B','C','D','E','F'],
    '국어': [80,90,95,70,75,85],
    '영어':[90,95,70,85,90,95],
    '수학':[85,95,75,80,85,100]
}
df1=pd.DataFrame(excel_exam_data1,columns=['학생','국어','영어','수학'])
df1


# In[30]:


excel_writer =pd.ExcelWriter('/Users/ktm/Desktop/Python-Program/test_score2.xlsx',engine='xlsxwriter')
df1.to_excel(excel_writer, index=False)
excel_writer.save()


# In[31]:


excel_writer = pd.ExcelWriter('/Users/ktm/Desktop/Python-Program/test_score2.xlsx', engine='xlsxwriter')
df1.to_excel(excel_writer, index=False, sheet_name='중간고사')
excel_writer.save()


# # 성적데이터2

# In[33]:


excel_exam_data2={
    '학생':['A','B','C','D','E','F'],
    '국어':[85,95,75,80,85,100],
    '영어':[80,90,95,70,75,85],
    '수학':[90,95,70,85,90,95]
}

df2=pd.DataFrame(excel_exam_data2,columns=['학생','국어','영어','수학'])
df2


# In[36]:


excel_writer=pd.ExcelWriter('/Users/ktm/Desktop/Python-Program/test_score2.xlsx', engine='xlsxwriter')
df1.to_excel(excel_writer, index=False, sheet_name='중간고사')
df2.to_excel(excel_writer, index=False, sheet_name='기말고사')
excel_writer.save()


# In[37]:


xls_file_1 = '/Users/ktm/Desktop/Python-Program/담당자별_판매실적_윤종필사원.xlsx'
xls_file_2 = '/Users/ktm/Desktop/Python-Program/담당자별_판매실적_홍성민대리.xlsx'
xls_file_3 = '/Users/ktm/Desktop/Python-Program/담당자별_판매실적_신은영과장.xlsx'


# In[38]:


excel_data_files=[xls_file_1, xls_file_2, xls_file_3 ]


# In[41]:


total_data=pd.DataFrame()


# In[42]:


for f in excel_data_files:
    df=pd.read_excel(f)
    total_data=total_data.append(df)
    
total_data


# In[43]:


total_data = pd.DataFrame()

for f in excel_data_files:
    df = pd.read_excel(f)
    total_data = total_data.append(df, ignore_index=True)

total_data


# In[44]:


import glob

glob.glob("./data/담당자별_판매실적_*.xlsx")


# In[46]:


excel_data_files1 = glob.glob("/Users/ktm/Desktop/Python-Program/담당자별_판매실적_*.xlsx")
total_data1 = pd.DataFrame()

for f in excel_data_files1:
    df = pd.read_excel(f)
    total_data1 = total_data1.append(df, ignore_index=True)

total_data1


# In[63]:


excel_file_name = '/Users/ktm/Desktop/Python-Program/마케팅팀_판매실적_통합.xlsx'

excel_total_file_writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
total_data1.to_excel(excel_total_file_writer, index=False, sheet_name='판매실적_통합')
excel_total_file_writer.save()

glob.glob(excel_file_name)


# # 엑셀데이터 핸들링

# In[49]:


df = pd.read_excel(xls_file_1)
df


# In[50]:


df.loc[2, '4분기'] = 0
df


# In[51]:


df.loc[3, '제품명'] = '노트10'
df.loc[3, '담당자'] = '김재경'
df.loc[3, '지역'] = '구로구'
df.loc[3, '1분기'] = 100
df.loc[3, '2분기'] = 150
df.loc[3, '3분기'] = 200
df.loc[3, '4분기'] = 250

df


# In[52]:


df['담당자'] = '김재경'
df


# In[54]:


xls_file_4 = '/Users/ktm/Desktop/Python-Program/담당자별_판매실적_김재경사원.xlsx'

new_excel_file = pd.ExcelWriter(xls_file_4, engine='xlsxwriter')
df.to_excel(new_excel_file, index=False)
new_excel_file.save()

# glob.glob(excel_file_name)


# # 여러 개의 엑셀 파일에서 데이터 수정

# In[55]:


file_name  = '/Users/ktm/Desktop/Python-Program/담당자별_판매실적_윤종필사원.xlsx'

new_file_name = re.sub(".xlsx", "2.xlsx", file_name)
new_file_name


# In[57]:


# 원하는 문자열이 포함된 파일을 검색해 리스트를 할당
excel_data_files = glob.glob("/Users/ktm/Desktop/Python-Program/담당자별_판매실적_*.xlsx")
excel_data_files


# In[58]:


for f in excel_data_files:
    df=pd.read_excel(f)
    
      # 특정 열의 값을 변경한다.    
    if(df.loc[1, '담당자']=='윤종필'):
        df['담당자']='윤종필사원'
    elif(df.loc[1, '담당자']=='홍성민'):
        df['담당자']='홍성민대리'
    elif(df.loc[1, '담당자']=='신은영'):
        df['담당자']='신은영과장' 
    elif(df.loc[1, '담당자']=='김재경'):
        df['담당자']='김재경사원'         
    
    # 엑셀 파일 이름에서 지정된 문자열 패턴을 찾아서 파일명을 변경한다.
    f_new = re.sub(".xlsx", "_v2.xlsx", f)
    print(f_new)
    
    # 수정된 데이터를 새로운 이름의 엑셀 파일로 저장한다.
    new_excel_file = pd.ExcelWriter(f_new, engine='xlsxwriter')
    df.to_excel(new_excel_file, index=False)
    new_excel_file.save()
    


# In[60]:


glob.glob("/Users/ktm/Desktop/Python-Program/담당자별_판매실적_*_v?.xlsx")


# # 엑셀의 필터 기능 수행

# In[66]:


df=pd.read_excel('/Users/ktm/Desktop/Python-Program/마케팅팀_판매실적_통합.xlsx')
df


# In[67]:


df['제품명']


# In[68]:


df['제품명']=='아이폰XR'


# In[71]:


iphone=df[df['제품명']=='아이폰XR']
iphone


# In[72]:


iPhone2 = df[df['제품명'].isin(['아이폰XR'])]
iPhone2


# In[73]:


apple = df[(df['제품명']== '아이폰XR') | (df['제품명']== '애플와치4')]
apple


# In[74]:


apple2 = df[df['제품명'].isin(['아이폰XR', '애플와치4'])]
apple2


# # 조건을 설정해 원하는 행만 선택

# In[75]:


df[(df['3분기'] >= 250)]


# In[76]:


df[(df['제품명'] == '갤럭시S10') & (df['3분기'] >= 250)]


# # 원하는 열만 선택

# In[77]:


df = pd.read_excel(xls_file_4)
df


# In[78]:


df[['제품명','1분기', '2분기','3분기', '4분기']]


# In[79]:


df.iloc[:,[0,3,4,5,6]]


# In[81]:


df.iloc[[0,2],:]


# # 엑셀 데이터 계산

# In[83]:


df = pd.read_excel('/Users/ktm/Desktop/Python-Program/마케팅팀_판매실적_통합.xlsx')

iPhone = df[(df['제품명']== '아이폰XR')]
iPhone


# In[85]:


iPhone.sum(axis=1)


# In[86]:


iPhone_sum = pd.DataFrame(iPhone.sum(axis=1), columns = ['연간판매량'])
iPhone_sum


# In[91]:


iPhone_total = iPhone.join(iPhone_sum)
iPhone_total


# In[92]:


iPhone_total.sort_values(by='연간판매량', ascending=True)


# In[93]:


iPhone_total.sort_values(by='연간판매량', ascending=False)


# # 열 데이터의 합계

# In[95]:


iPhone_total.sum()


# In[96]:


iPhone_sum2=pd.DataFrame(iPhone_total.sum(),columns=['합계'])
iPhone_sum2


# In[98]:


iPhone_total2=iPhone_total.append(iPhone_sum2.T)
iPhone_total2


# In[99]:


iPhone_total2.loc['합계', '제품명'] = '아이폰XR	'
iPhone_total2.loc['합계', '담당자'] = '전체'
iPhone_total2.loc['합계', '지역']   = '전체'
iPhone_total2


# In[101]:


df = pd.read_excel('/Users/ktm/Desktop/Python-Program/마케팅팀_판매실적_통합.xlsx')

# 제품명 열에서 아이폰XR	이 있는 행만 선택
product_name = '아이폰XR'
iPhone = df[(df['제품명']== product_name)]

# 행별로 합계를 구하고 마지막 열 다음에 추가한다.
iPhone_sum = pd.DataFrame(iPhone.sum(axis=1), columns = ['연간판매량'])
iPhone_total = iPhone.join(iPhone_sum)

# 열별로 합해 분기별 합계와 연간판매량 합계를 구하고 마지막 행 다음에 추가한다.
iPhone_sum2 = pd.DataFrame(iPhone_total.sum(), columns=['합계'])
iPhone_total2  = iPhone_total.append(iPhone_sum2.T)

# 지정된 항목의 문자열을 변경한다.
iPhone_total2.loc['합계', '제품명'] = product_name
iPhone_total2.loc['합계', '담당자'] = '전체'
iPhone_total2.loc['합계', '지역'  ] = '전체'

# 결과를 확인한다.
iPhone_total2


# In[102]:


def get_product_total(xls_file, product_name):
    # 제품명 열에서 아이폰XR	이 있는 행만 선택
    # product_name = '아이폰XR'
    iPhone = df[(df['제품명']== product_name)]

    # 행별로 합계를 구하고 마지막 열 다음에 추가한다.
    iPhone_sum = pd.DataFrame(iPhone.sum(axis=1), columns = ['연간판매량'])
    iPhone_total = iPhone.join(iPhone_sum)

    # 열별로 합해 분기별 합계와 연간판매량 합계를 구하고 마지막 행 다음에 추가한다.
    iPhone_sum2 = pd.DataFrame(iPhone_total.sum(), columns=['합계'])
    iPhone_total2  = iPhone_total.append(iPhone_sum2.T)

    # 지정된 항목의 문자열을 변경한다.
    iPhone_total2.loc['합계', '제품명'] = product_name
    iPhone_total2.loc['합계', '담당자'] = '전체'
    iPhone_total2.loc['합계', '지역'  ] = '전체'

    # 결과를 확인한다.
    return iPhone_total2


# In[103]:


xls_file = './data/마케팅팀_판매실적_통합.xlsx'
prd_name = '아이폰XR'

iPhone_total = get_product_total(xls_file, prd_name)
iPhone_total


# In[104]:


prd_name = '갤럭시S10'

galaxy_total = get_product_total(xls_file, prd_name)
galaxy_total


# In[105]:


prd_name = '애플와치4'

aWatch_total = get_product_total(xls_file, prd_name)
aWatch_total


# # 엑셀데이터 시각화

# In[106]:


from IPython.display import Image

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family']        = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

get_ipython().run_line_magic('matplotlib', 'inline')


# # 엑셀 파일에 그래프 삽입

# In[107]:


import matplotlib

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False


# In[108]:


sales={'시간':[9,10,11,12,13,14,15],
      '제품1':[10,15,12,11,12,14,13],
      '제품2':[9,11,14,12,13,10,12]
}
df=pd.DataFrame(sales, index=sales['시간'], columns=['제품1', '제품2'])
df


# In[109]:


#index 라벨 추가
df.index.name='시간'
df


# In[110]:


product_plot=df.plot(grid=True, style =['-*','-o'])
product_plot;


# In[116]:


get_ipython().system('mkdir figures')


# In[117]:


product_plot = df.plot(grid= True, style=['-*','-o'])
product_plot.set_title('시간대별 생산량')
product_plot.set_ylabel("생산량")

fig_file='/Users/ktm/Anaconda_src/data/fig_for_excel1.png'
plt.savefig(fig_file, dpi=400)

plt.show()


# In[118]:


Image(fig_file)


# # 엑셀 이미지 추가

# In[121]:


# (1) pandas의 ExcelWriter 객체 생성
excel_file   = '/Users/ktm/Anaconda_src/data/data_image_to_excel.xlsx'
excel_writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')

# (2) DataFrame 데이터를 지정된 엑셀 시트(Sheet)에 쓰기
df.to_excel(excel_writer, index=True, sheet_name='Sheet1')

# (3) ExcelWriter 객체에서 워크시트(worksheet) 객체 생성
worksheet = excel_writer.sheets['Sheet1']

# (4) 워크시트에 차트가 들어갈 위치를 지정해 이미지 넣기
worksheet.insert_image('D2', fig_file, {'x_scale': 0.7, 'y_scale': 0.7})
# worksheet.insert_image(1, 3, fig_file, {'x_scale': 0.7, 'y_scale': 0.7})

# (5) ExcelWriter 객체를 닫고 엑셀 파일 출력
excel_writer.save()


# In[122]:


get_ipython().system('dir data\\*.xlsx')


# # 엑셀 차트 생성

# In[124]:


# (1) pandas의 ExcelWriter 객체 생성
excel_file  = '/Users/ktm/Anaconda_src/data/data_chart_in_excel.xlsx'
excel_chart = pd.ExcelWriter(excel_file, engine='xlsxwriter')

# (2) DataFrame 데이터를 지정된 엑셀 시트(Sheet)에 쓰기
df.to_excel(excel_chart, index=True, sheet_name='Sheet1')

# (3) ExcelWriter 객체에서 워크북(workbook)과 워크시트(worksheet) 객체 생성
workbook  = excel_chart.book
worksheet = excel_chart.sheets['Sheet1']

# (4) 차트 객체 생성(원하는 차트의 종류 지정)
chart = workbook.add_chart({'type': 'line'})

# (5) 차트 생성을 위한 데이터값의 범위 지정 
chart.add_series({'values': '=Sheet1!$B$2:$B$8'})
chart.add_series({'values': '=Sheet1!$C$2:$C$8'})

# (6) 워크시트에 차트가 들어갈 위치를 지정해 차트 넣기
worksheet.insert_chart('D2', chart)

# (7) ExcelWriter 객체를 닫고 엑셀 파일 출력
excel_chart.save()


# # 차트 생성을 위한 데이터값의 범위 지정

# In[128]:


chart.add_series({'values'    : '=Sheet1!$B$2:$B$8', 
                  'categories': '=Sheet1!$A$2:$A$8',
                  'name'      : '=Sheet1!$B$1',})

chart.add_series({'values'    : '=Sheet1!$C$2:$C$8', 
                  'categories': '=Sheet1!$A$2:$A$8',
                  'name'      : '=Sheet1!$C$1',})


# # 엑셀 차트에 제목과 x,y축 라벨 추가

# In[129]:


chart.set_title ({'name': '시간대별 생산량'})
chart.set_x_axis({'name': '시간'})
chart.set_y_axis({'name': '생산량'})


# # Final 엑셀 차트

# In[131]:


# (1) pandas의 ExcelWriter 객체 생성
excel_file2 = '/Users/ktm/Anaconda_src/data/data_chart_in_excel2.xlsx'
excel_chart = pd.ExcelWriter(excel_file2, engine='xlsxwriter')

# (2) DataFrame 데이터를 지정된 엑셀 시트(Sheet)에 쓰기
df.to_excel(excel_chart, index=True, sheet_name='Sheet1')

# (3) ExcelWriter 객체에서 워크북(workbook)과 워크시트(worksheet) 객체 생성
workbook  = excel_chart.book
worksheet = excel_chart.sheets['Sheet1']

# (4) 차트 객체 생성 (원하는 차트의 종류 지정)
chart = workbook.add_chart({'type': 'line'})

# (5) 차트 생성을 위한 데이터값의 범위 지정
chart.add_series({'values'    : '=Sheet1!$B$2:$B$8', 
                  'categories': '=Sheet1!$A$2:$A$8',
                  'name'      : '=Sheet1!$B$1',})

chart.add_series({'values'    : '=Sheet1!$C$2:$C$8', 
                  'categories': '=Sheet1!$A$2:$A$8',
                  'name'      : '=Sheet1!$C$1',})

# (5-1) 엑셀 차트에 x, y축 라벨과 제목 추가
chart.set_title ({'name': '시간대별 생산량'})
chart.set_x_axis({'name': '시간'})
chart.set_y_axis({'name': '생산량'})

# (6) 워크시트에 차트가 들어갈 위치를 지정해 차트 넣기
worksheet.insert_chart('D2', chart)

# (7)  ExcelWriter 객체를 닫고 엑셀 파일 출력
excel_chart.save()

