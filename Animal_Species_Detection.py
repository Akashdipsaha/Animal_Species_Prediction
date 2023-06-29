#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


# In[2]:


model = ResNet50(weights='imagenet')


# In[3]:


def predict_animal_species(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the predictions
    for _, animal, probability in decoded_predictions:
        print(f"{animal}: {probability:.2%}")
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# In[6]:


image_path = r'C:\Users\KIIT\Downloads\ray-hennessy-xUUZcpQlqpM-unsplash.jpg'
predict_animal_species(image_path)


# In[ ]:




