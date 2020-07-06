#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add



# In[118]:


model=load_model("model_19.h5")
model._make_predict_function()



# In[119]:


temp_model=ResNet50(weights="imagenet",input_shape=(224,224,3))
# temp_model.summary()


# In[120]:


resnet_model=Model(temp_model.input,temp_model.layers[-2].output)
resnet_model._make_predict_function()


# In[121]:


def functiontoPreprocessImage(img):
    
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    
    #ResNet accepts a 4D Tensor so we do not feed an single Image but Batch of Image thus we expand :
    #(1,224,224,3)
    img=np.expand_dims(img,axis=0)
    
    #Normalizing the Data as per ResNet50: preprocess_input is Resnet's Inbuilt Function
    img=preprocess_input(img)
    
    return img


# In[122]:


def encode_image(img):
    
    img= functiontoPreprocessImage(img)
    feature_vector=resnet_model.predict(img)
    
    feature_vector=feature_vector.reshape(1,feature_vector.shape[1])
#     print(feature_vector.shape)
    
    return feature_vector


# In[135]:





# In[136]:


with open("word_to_idx.pkl","rb") as w2i:
    word_to_idx=pickle.load(w2i)


# In[137]:


with open("idx_to_word.pkl","rb") as i2w:
    idx_to_word=pickle.load(i2w)


# In[138]:


# print(word_to_idx)


# In[139]:


#Prediction

def prediction_caption(Picture):
    
    input_text="startseq"
    max_sentence_len=35
    for i in range(max_sentence_len):
        
        sequence=[word_to_idx[word] for word in input_text.split() if word in word_to_idx ]
        sequence=pad_sequences([sequence],maxlen=max_sentence_len,padding='post')
        
        prediction=model.predict([Picture,sequence])
        prediction=prediction.argmax()
        predicted_word=idx_to_word[prediction]
        input_text+=(' '+predicted_word)
        
        if predicted_word=='endseq':
            break
            
    final_caption = input_text.split()[1:-1]
    final_caption=' '.join(final_caption)
    
    return final_caption


# In[140]:
def caption_the_image(image):
    
    encode=encode_image(image)
    caption=prediction_caption(encode)

    return caption


