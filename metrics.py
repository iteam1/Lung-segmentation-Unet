import numpy as np 
import tensorflow as tf
#from tensorflow.keras import backend as K

smooth = 1e-15

def iou(y_true,y_pred):
    '''
    IOU metric
    '''
    def f(y_true,y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - itersection
        x = (itersection + smooth)/(union + smooth)
        x = x.astype(np.float32)
        return x  
    return tf.numpy_function(f,[y_true,y_pred],tf.float32)

def dice_coef(y_true,y_pred):
    '''
    Dice coefficent metric
    '''
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection  =  tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth)/(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)+ smooth)

def dice_loss(y_true,y_pred):
    '''
    Loss of dice coefficient
    '''
    return 1.0 - dice_coef(y_true,y_pred)
