import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import tensorflow as tf
import os 
import glob
import train 
import predict as p
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope

def slow_evaluate(x,y1,y2,model,path = './results/',savefig = True):
    '''
    Evaluate your dataset
    x --- list of valid image path
    y1 --- list of valid left mask path
    y2 --- list of valid right mask path
    path --- path of evaluate result
    model --- your model
    savefig --- boolean option save your prediction, default = True
    Return mean loss of evaluation
    '''
    f_loss = tf.keras.losses.BinaryCrossentropy() # Loss function
    losses = []
    for i in tqdm(range(len(x))):
        # Predicting the mask
        y_pred = p.predict_img(x[i],model,savedir = path,savefig = savefig)
        ground_truth = train.read_mask(y1[i],y2[i])
        losses.append(f_loss(ground_truth,y_pred[0]).numpy()) # storage loss value of this image
    print(f"Evaluated Mean loss: {np.mean(losses)}")
    return np.mean(losses)

if __name__ == '__main__':
    
    # Seeding
    np.random.seed(42)
    np.random.seed(42)
    
    # load train valid test
    (x_train,y1_train,y2_train),(x_valid,y1_valid,y2_valid),(x_test,y1_test,y2_test) = train.load_data(path = './NLM-MontgomeryCXRSet\MontgomerySet')

    # create results floder
    train.create_dir('./results')
    
    # Set custom classes or functions in model with your custom definition
    # with CustomObjectScope({'iou':metrics.iou,'dice_coef':metrics.dice_coef,'dice_loss':metrics.dice_loss}):
    # model = tf.keras.models.load_model('/content/conbtent/MyDrive/model_lung_unet1.h5')
    model = tf.keras.models.load_model('/files/model_lung_unet1.h5')

    loss = slow_evaluate(x_valid,y1_valid,y2_valid,model)