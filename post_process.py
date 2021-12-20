# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:04:41 2021

@author: 顾舜贤
"""


import numpy as np
import cv2 
from PIL import Image

load_image="finger.npy"
save_picture_name="finger.png"


image_array=np.load(load_image)
image=np.squeeze(image_array.transpose([2,3,1,0]))
image_post=np.maximum(np.minimum(np.uint8(image*255),255),0)
#image_post=cv2.GaussianBlur(image_post,(7,7), 1)
image_post=cv2.medianBlur(image_post,11)
HR= Image.fromarray(image_post)
HR.save(save_picture_name)