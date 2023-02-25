#!/usr/bin/env python
# coding: utf-8

# In[73]:


import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\Shashank\Downloads\linear.tflite")
interpreter.allocate_tensors()


# In[74]:


input_index=interpreter.get_input_details()

# In[75]:


from PIL import Image
from PIL.Image import Resampling
import numpy as np
img=Image.open(r"C:\Users\Shashank\Downloads\wool.jpg").resize((300,300))
Image.Resampling.LANCZOS
im=np.array(img,dtype=np.float32)
imarr=im[np.newaxis, ...]


# In[76]:


input_index=interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index,imarr)
interpreter.invoke()
output_details=interpreter.get_output_details()


# In[77]:


output_data = interpreter.get_tensor(output_details[0]['index'])
pred = np.squeeze(output_data)


# In[78]:


class_names=["Cotton","Denim","Nylon","Polyester","Silk","Wool"]
prediction = np.argmax(pred)


# In[79]:


print(class_names[prediction])


# In[ ]:




