#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastbook
# fastbook.setup_book()
from fastai.vision.all import *
from fastai.vision.widgets import *


# In[2]:


path = Path('/Users/shamshersingh/Downloads/learn_vit.pkl')

# load_learner is fn from fastai.vision.all that loads a saved model from a file
learn_inf = load_learner(path, cpu = True)

# create a file upload button that will allow users to upload an image for prediction
file_upload_btn = widgets.FileUpload()

# creates output widget to display the uploaded image
out_img = widgets.Output()

# creates a label widget to display the model's prediction about the uploaded image
lbl_pred = widgets.Label()


# In[3]:


# on_data_change is a fn designed to be called when the data in upload widget changes.
# change parameter is a standard argument for event handlers representing information about the data change event

def on_data_change(change):
    lbl_pred.value = '' # clear previous prediction 
    img = PILImage.create(file_upload_btn.data[-1]) # create image from upload widget
    out_img.clear_output() # this clears any previous output displayed in out_img widget
    with out_img : display(img.to_thumb(256,256)) # display a thumbnail of uploaded image within out_img widget
    pred, pred_idx, prob = learn_inf.predict(img) # make predictions
    lbl_pred.value = f'Prediction = {pred} ; Probability = {prob[pred_idx]:.4f}'  # update lbl_pred widget with the prediction results. it uses f-string to format the output  


# In essence, this function handles the following:
# 
# 1. When new data is uploaded via file_upload_btn, it is converted into an image.
# 2. The image is displayed as a thumbnail.
# 3. The machine learning model learn_inf predicts the class of the image.
# 4. The prediction and its probability are displayed in the lbl_pred widget.

# In[4]:


# set up an observer for file_upload_btn widget. .observe() method allows to monitor changes to the file upload 
# button. Whenever there is change in file upload widget, .observe() method is instructed to call on_data_change
# fn. The names = ['data'] specifies which change in file upload widget should trigger the on_data_change fn. 
# Therefore, in this case it is looking for changes to the data attribute of file upload button as this attribute 
# holds the data of the uploaded file. 

'''
In essence, this line sets up an event listener that triggers a specific action (the on_data_change 
function) when a particular event (file upload) occurs.
'''

file_upload_btn.observe(on_data_change, names = ['data'])


# In[5]:


# display interactive elements of image classification app. display() - fn from IPython.display that is used
# to show objects in output area of Jupyter notebook cell.

# VBox() is a class from ipywidgets lib which is used to create vertical box layout. 

# [widgets.Label....] is a list containing the widgets that will be placed inside the vertical box. 

display(VBox([widgets.Label('Select Your Image!'), file_upload_btn, out_img, lbl_pred]))

