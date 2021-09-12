#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import cv2
import os
import numpy as np
import adain_model
import DDAdain_model
import dataset
from dataset import *
import os
from PIL import Image
import test
from test import *
import torch
from torchvision import transforms
from torchvision.utils import save_image



# In[2]:


###you don't need to five the following file name
def main():
#     for model_type in ['adain']:
#     size = read_video(f"./{eg}/{content_video_name}")
    for model_type in ['DDAdain', 'adain']:
        if model_type == 'adain':
            model = adain_model.StyleTranserNetwork()
        elif model_type == 'DDAdain':
            model = DDAdain_model.StyleTranserNetwork()
        else:
            raise WrongModelType

        svnc = f'styled_video_no_compare_{model_type}'
        svc = f'styled_video_compare_{model_type}'
        sv = f'styled_video_{model_type}'
        folders = [sv, svc, svnc]
    
        if model_type == 'DDAdain':
            style_transfer(f'./{eg}/video',f'./{eg}/{style_name}', 1, f'./result/model_state_{model_type}/40_epoch.pth', model, model_type)
        elif model_type == 'adain':
            style_transfer(f'./{eg}/video',f'./{eg}/{style_name}', 1, f'./result/model_state_{model_type}/70_epoch.pth', model, model_type)

        for name in folders: 
            create_video(f'./{eg}/{name}', f"./{eg}/{name}.avi", fps)


# In[3]:


###should be given!       
# eg = 'eg1'  
# content_video_name = 'content.mov'
# style_name = 'style.jpeg'
#fps=60
# main()

# eg = 'eg2'  
# content_video_name = 'content.mp4'
# style_name = 'style.jpg'
# fps=30
# main()

# eg = 'eg3'  
# content_video_name = 'content.MOV'
# style_name = 'style.jpg'
# fps=30
# main()

eg = 'eg4'  
content_video_name = 'content.mov'
style_name = 'style.jpeg'
fps=30
main()


# In[ ]:





# In[ ]:




