from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path

# Import fast.ai Library
from fastai import *
from fastai.vision import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


#load model
def load_model(model_path):
    path = Path(model_path)
    classes = ['Angry', 'Surprised','Happy', 'Sad']
    data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms()).normalize(imagene$
    learn = create_cnn(data2, models.resnet50)
    learn.load('stage-2-rn50')
    return learn

def model_predict(img_path, model_path):
    """
       model_predict will return the preprocessed image
    """
    learn = load_model(model_path)
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    return pred_class
