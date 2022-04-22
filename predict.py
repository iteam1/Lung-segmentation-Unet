import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import tensorflow as tf
import os 
import glob
import train 
import metrics 
import model
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope

def predict_img(path,model,savedir = './predictions/',show = True,savefig = True):
    '''
    Predict your image with trained model and save the prediction
    path --- path of image to predict
    model --- your model
    savedir ---where you save your prediction, default = './predictions/'
    show --- boolean option display your prediction, default = True
    savefig --- boolean option save your prediction, default = True
    Return y_pred (tensor array) shape [1,H,W,1]
    '''
    img_pred = train.read_image(path)
    img_name = path.split('/')[-1]
    img_name = img_name.split('.')[0]
    img_name = img_name + '_predicted'
    y_pred = model.predict(tf.expand_dims(img_pred,axis =0))
    if show:
        plt.title(img_name)
        plt.imshow(y_pred[0][:,:,0])
        # there is no image to save if you don't show anything
        if savefig:
            plt.savefig(savedir + img_name + '.png')
    return y_pred

if __name__ == '__main__':
    
    pred_path = input('Enter your image path: ')
    
    # create results floder
    train.create_dir('./predictions')
    
    # Set custom classes or functions in model with your custom definition
    # with CustomObjectScope({'iou':metrics.iou,'dice_coef':metrics.dice_coef,'dice_loss':metrics.dice_loss}):
    # model = tf.keras.models.load_model('/content/conbtent/MyDrive/model_lung_unet1.h5')
    model = tf.keras.models.load_model('/files/model_lung_unet1.h5')

    # predict
    predict_img(pred_path,model)