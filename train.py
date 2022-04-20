import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable show warning
import glob
import numpy as np
import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import Recall,Precision
from sklearn.model_selection import train_test_split
from model import build_unet
from metrics import dice_loss,dice_coef,iou

# Global parameters and Hyperparameters
H = 512 # image height
W = 512 # image width
batch_size = 8 # training image each path
lr = 1e-5 # learning rate
num_epochs = 10 # epochs
dataset_path = './NLM-MontgomeryCXRSet/MontgomerySet'
f_name = 'files' # the name of floder storing model training result
model_path = os.path.join(f_name,'model.h5')
csv_path  = os.path.join(f_name,'datalog.csv')

def create_dir(path):
    '''
    Create a directory if it not exist
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_data(path= 'NLM-MontgomeryCXRSet\MontgomerySet',test_size = 0.2):
    '''
    Create train-valid-test
    path --- dataset directory default = NLM-MontgomeryCXRSet\MontgomerySet
    test_size --- test size default = 0.2
    Return train-set,valid-set,test-set 
    '''
    images = sorted(glob.glob(path + '/CXR_png/' + '*.png')) # lung Xray image
    mask_1 = sorted(glob.glob(path + '/ManualMask/leftMask/' + '*.png')) # lung left mask
    mask_2 = sorted(glob.glob(path + '/ManualMask/rightMask/' + '*.png')) # lung right mask
    # split train-valid
    X_train,X_valid = train_test_split(images,test_size = test_size,random_state = 42) # split list of image
    y1_train,y1_valid = train_test_split(mask_1,test_size = test_size,random_state = 42)
    y2_train,y2_valid = train_test_split(mask_2,test_size = test_size,random_state = 42)
    # split train-test
    X_train,X_test = train_test_split(X_train,test_size = test_size,random_state=  42)
    y1_train,y1_test = train_test_split(y1_train,test_size = test_size,random_state = 42)
    y2_train,y2_test = train_test_split(y2_train,test_size = test_size,random_state = 42)
    
    return (X_train,y1_train,y2_train),(X_valid,y1_valid,y2_valid),(X_test,y1_test,y2_test)

def read_image(path,target_size = (512,512)):
    '''
    Read image from path ,Scale into [0:1] return tensor array
    path --- image path
    target_size --- resize the image into this size
    Return tensor array [None,H,W,3]
    '''
    image = tf.keras.preprocessing.image.load_img(path,target_size = target_size)
    image_tensor = tf.keras.preprocessing.image.img_to_array(image)
    image_tensor = tf.expand_dims(image_tensor,axis = 0) 
    return image_tensor

def read_mask(path1,path2):
    '''
    Read left mask  and right mask then add them into 1 picture
    path1 --- left image path
    path2 --- right image path
    target_size --- resize the image into this size
    '''
    x1 = tf.keras.preprocessing.image.load_img(path1,target_size = (512,512))
    x1 = tf.keras.preprocessing.image.img_to_array(x1)
    x2 = tf.keras.preprocessing.image.load_img(path2,target_size = (512,512))
    x2 = tf.keras.preprocessing.image.img_to_array(x2)
    x = x1 + x2
    x = x/255. # Scale it
    x = (x > 0.5) * 255.
    x = tf.expand_dims(x,aixs = 0)
    return x
        
if __name__ == '__main__':
    
    # Seeding
    np.random.seed(42)
    np.random.set_seed(42)
    
    # Directory for storing training files
    create_dir(f_name)
    
    # Load dataset


