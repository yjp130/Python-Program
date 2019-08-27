
# coding: utf-8

# In[1]:


import re 


# In[17]:


with open('./Downloads/sample_emails.txt', encoding='utf-8') as fp:
    text = fp.read()


# In[20]:


text


# In[23]:





# In[31]:


def write_txt(filepath):
    
    count=1
    data=[]
    
    result = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    for email in result:
        data.append(email+'\n')
        count+=1
    f=open(filepath, 'w')
    f.writelines(data)
    f.close()
    
    ret="완료"
    
    return ret


# In[32]:


filepath='./Downloads/emails_list.txt'
write_txt(filepath)

