#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np


# In[22]:


root_dir="D:/data/kfood/"
cat=['Chicken','Dolsotbab']
nb_classes=len(cat)


# In[23]:


image_width=64
image_height=64


# In[24]:


# 데이터 변수
X = []
Y = []

for idx, category in enumerate(cat):
    image_dir = root_dir + category
    files = glob.glob(image_dir + "/" + "*.jpg")
    print(image_dir + "/" + "*.jpg")
    
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)


# In[ ]:





# In[ ]:




