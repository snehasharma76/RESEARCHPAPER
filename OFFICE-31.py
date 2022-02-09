#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
#!{sys.executable} -m pip install pandas 
import pandas as pd
import pathlib
import cv2
import os
import glob


# In[2]:


dir_path = "D:/Office/Office-31/amazon/back_pack"
#Change string to path
data_root = pathlib.Path(dir_path)


# In[3]:


#List all file names in the root directory
def return_file_names(root_dir):
    arr = os.listdir(root_dir)
    #col=[]
    li=[]
    for i in arr: #frame_001.jpg
        dir_path = str(root_dir)+'\\'+ i
        #li= list(sorted(data_root.glob('*.*')))
        #li=[str(path) for path in li]
        li.append(dir_path)
        #col.append(li)
        
        
        data_df = pd.DataFrame()
        data_df['path_img'] = li
    return data_df
df_train = return_file_names(data_root)


# In[4]:


dir_path2 = "D:/Office/Office-31/amazon/bike"
data_root2 = pathlib.Path(dir_path2)
df_train2 = return_file_names(data_root2)


# In[5]:


df_train2.head


# In[6]:


dir_path3 = "D:/Office/Office-31/amazon/bike_helmet"
data_root3 = pathlib.Path(dir_path3)
df_train3 = return_file_names(data_root3)


# In[7]:


dir_path4 = "D:/Office/Office-31/amazon/bookcase"
data_root4 = pathlib.Path(dir_path4)
df_train4 = return_file_names(data_root4)


# In[8]:


dir_path5 = "D:/Office/Office-31/amazon/bottle"
data_root5 = pathlib.Path(dir_path5)
df_train5 = return_file_names(data_root5)


# In[9]:


dir_path6 = "D:/Office/Office-31/amazon/calculator"
data_root6 = pathlib.Path(dir_path6)
df_train6 = return_file_names(data_root6)


# In[10]:


dir_path7 = "D:/Office/Office-31/amazon/desk_chair"
data_root7 = pathlib.Path(dir_path7)
df_train7 = return_file_names(data_root7)


# In[11]:


dir_path8 = "D:/Office/Office-31/amazon/desk_lamp"
data_root8 = pathlib.Path(dir_path8)
df_train8 = return_file_names(data_root8)


# In[12]:


dir_path9 = "D:/Office/Office-31/amazon/desktop_computer"
data_root9 = pathlib.Path(dir_path9)
df_train9 = return_file_names(data_root9)


# In[13]:


dir_path10 = "D:/Office/Office-31/amazon/file_cabinet"
data_root10 = pathlib.Path(dir_path10)
df_train10= return_file_names(data_root10)


# In[14]:


#df_merge = pd.concat([df_train,df_train2,df_train3])


# In[15]:


#df_merge.head


# In[16]:


dir_path11 = "D:/Office/Office-31/amazon/headphones"
data_root11  = pathlib.Path(dir_path11)
df_train11  = return_file_names(data_root11)


# In[17]:


dir_path12 = "D:/Office/Office-31/amazon/keyboard"
data_root12 = pathlib.Path(dir_path12)
df_train12 = return_file_names(data_root12)


# In[18]:


dir_path13 = "D:/Office/Office-31/amazon/laptop_computer"
data_root13 = pathlib.Path(dir_path13)
df_train13 = return_file_names(data_root13)


# In[19]:


dir_path14 = "D:/Office/Office-31/amazon/letter_tray"
data_root14 = pathlib.Path(dir_path14)
df_train14 = return_file_names(data_root14)


# In[20]:


dir_path15 = "D:/Office/Office-31/amazon/mobile_phone"
data_root15 = pathlib.Path(dir_path15)
df_train15 = return_file_names(data_root15)


# In[21]:


dir_path16 = "D:/Office/Office-31/amazon/monitor"
data_root16 = pathlib.Path(dir_path16)
df_train16 = return_file_names(data_root16)


# In[22]:


dir_path17 = "D:/Office/Office-31/amazon/mouse"
data_root17 = pathlib.Path(dir_path17)
df_train17 = return_file_names(data_root17)


# In[23]:


dir_path18 = "D:/Office/Office-31/amazon/mug"
data_root18 = pathlib.Path(dir_path18)
df_train18= return_file_names(data_root18)


# In[24]:


dir_path19 = "D:/Office/Office-31/amazon/paper_notebook"
data_root19 = pathlib.Path(dir_path19)
df_train19 = return_file_names(data_root19)


# In[25]:


dir_path20 = "D:/Office/Office-31/amazon/pen"
data_root20 = pathlib.Path(dir_path20)
df_train20= return_file_names(data_root20)


# In[26]:


dir_path21= "D:/Office/Office-31/amazon/phone"
data_root21 = pathlib.Path(dir_path21)
df_train21 = return_file_names(data_root21)


# In[27]:


dir_path22 = "D:/Office/Office-31/amazon/printer"
data_root22 = pathlib.Path(dir_path22)
df_train22= return_file_names(data_root22)


# In[28]:


dir_path23 = "D:/Office/Office-31/amazon/projector"
data_root23 = pathlib.Path(dir_path23)
df_train23 = return_file_names(data_root23)


# In[29]:


dir_path24 = "D:/Office/Office-31/amazon/punchers"
data_root24 = pathlib.Path(dir_path24)
df_train24 = return_file_names(data_root24)


# In[30]:


dir_path25= "D:/Office/Office-31/amazon/ring_binder"
data_root25 = pathlib.Path(dir_path25)
df_train25 = return_file_names(data_root25)


# In[31]:


dir_path26 = "D:/Office/Office-31/amazon/ruler"
data_root26 = pathlib.Path(dir_path26)
df_train26 = return_file_names(data_root26)


# In[32]:


dir_path27 = "D:/Office/Office-31/amazon/scissors"
data_root27 = pathlib.Path(dir_path27)
df_train27 = return_file_names(data_root27)


# In[33]:


dir_path28 = "D:/Office/Office-31/amazon/speaker"
data_root28 = pathlib.Path(dir_path28)
df_train28 = return_file_names(data_root28)


# In[34]:


dir_path29= "D:/Office/Office-31/amazon/stapler"
data_root29 = pathlib.Path(dir_path29)
df_train29 = return_file_names(data_root29)


# In[35]:


dir_path30 = "D:/Office/Office-31/amazon/tape_dispenser"
data_root30 = pathlib.Path(dir_path30)
df_train30 = return_file_names(data_root30)


# In[36]:


dir_path31 = "D:/Office/Office-31/amazon/trash_can"
data_root31 = pathlib.Path(dir_path31)
df_train31 = return_file_names(data_root31)


# In[37]:


#df_merge = pd.concat([df_train,df_train2,df_train3,df_train4,df_train5,df_train6,df_train7,df_train8,df_train9,df_train10,df_train11,df_train12,df_train13,df_train14,df_train15,df_train16,df_train17,df_train18,df_train19,df_train20,df_train21,df_train22,df_train23,df_train24,df_train25,df_train26,df_train27,df_train28,df_train29,df_train30,df_train31])
frames = [df_train,df_train2,df_train3,df_train4,df_train5,df_train6,df_train7,df_train8,df_train9,df_train10,df_train11,df_train12,df_train13,df_train14,df_train15,df_train16,df_train17,df_train18,df_train19,df_train20,df_train21,df_train22,df_train23,df_train24,df_train25,df_train26,df_train27,df_train28,df_train29,df_train30,df_train31]
df_merge = pd.concat(frames,ignore_index=True)


# In[38]:


df_merge.head


# In[39]:


from matplotlib.pyplot import imshow
import numpy as np
import cv2


# In[40]:


from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import os
#from tqdm import tqdm
 #Limiting to 256 size image as my laptop cannot handle larger images. 
#img_data=[]


# In[41]:


#from pathlib import Path
#filepath = df_merge['path_img']


# In[42]:


#from sklearn.model_selection import train_test_split
#X_train, X_test = train_test_split(df_merge, test_size=0.20, random_state=42)


# In[43]:


#X_train.head


# In[44]:


SIZE=28
#X =pd.DataFrame()
li=[]
for i in range(0,2817):
    img_data=[]
    filepath = df_merge['path_img'][i]
    img = cv2.imread(filepath)
    img=cv2.resize(img,(SIZE, SIZE))
    img_data.append(img_to_array(img))
    img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
    img_array = np.asarray(img_array).astype(np.float32) / 255.
    li.append(img_array)
li=np.array(li)
#X['image'] = li


# In[45]:


#X.head


# In[46]:


from sklearn.model_selection import train_test_split
#X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)
X_train, X_test = train_test_split(li, test_size=0.20, random_state=42)
#Difficult to reshape as it is a dataframe.
#X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#X_train = X_train.astype("float32")/255.
#X_test = X_test.astype("float32")/255.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 3)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 3)
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.


# In[ ]:





# In[47]:


#model = Sequential()
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))


# In[48]:


#model.add(MaxPooling2D((2, 2), padding='same'))
     
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))


# In[49]:


#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#model.summary()


# In[50]:


# Encoder
from keras.layers import Input,add
x = Input(shape=(28, 28,3)) 
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
h = MaxPooling2D((2, 2), padding='same')(conv1_3)


# Decoder
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
conv2_3 = Conv2D(16, (3, 3), activation='relu')(up2)
up3 = UpSampling2D((2, 2))(conv2_3)
r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)

from keras.models import Model
autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])


# In[51]:


epochs = 100
batch_size = 256

history = autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, X_test))


# In[52]:


decoded_imgs = autoencoder.predict(X_test)
print(decoded_imgs)


# In[53]:


import matplotlib.pyplot as plt
from PIL import Image
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i+1)
    img = Image.fromarray(X_test[i].reshape(28,28,3),'RGB')
    img.show()
    #plt.imshow(X_test[i].reshape(28, 28,3))
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    
    # display reconstruction
    #ax = plt.subplot(3, n, i+n+1)
    #plt.imshow(decoded_imgs[i].reshape(28, 28,3))
    img2 = Image.fromarray(decoded_imgs[i].reshape(28,28,3),'RGB')
    img2.show()
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




