import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable show warning
import glob
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import Recall,Precision
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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
    image_tensor = image_tensor /255.
    #image_tensor = tf.expand_dims(image_tensor,axis = 0) 
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
    x = x.sum(axis =2)
    x = x/255. # Scale it
    x = (x > 0.5) * 1
    x = x.reshape((512,512,1))
    return x

def make_data(x,y1,y2):
    X  =np.zeros(shape = (len(x),512,512,3))
    y = np.zeros(shape = (len(x),512,512,1))
    for i in tqdm(range(len(x))):
        X[i] = read_image(x[i],target_size = (512,512))
        y[i] = read_mask(y1[i],y2[i])
        
    return X,y
       
if __name__ == '__main__':
    
    # Seeding
    np.random.seed(42)
    np.random.seed(42)
    
    # Directory for storing training files
    create_dir(f_name)
    
    # Load dataset
    (train_x,train_y1,train_y2),(valid_x,valid_y1,valid_y2),(test_x,test_y1,test_y2) = load_data()
    print(f"Train: {len(train_x)} | {len(train_y1)} | {len(train_y2)} |")
    X_train,y_train = make_data(x_train,y1_train,y2_train)
    print(f"Valid: {len(valid_x)} | {len(valid_y1)} | {len(valid_y2)} |")
    X_valid,y_valid = make_data(x_valid,y1_valid,y2_valid)
    print(f"Test: {len(test_x)} | {len(test_y1)} | {len(test_y2)} |")
    X_test,y_test = make_data(x_test,y1_test,y2_test)
    
    # Build model
    model1 = build_unet(input_shape = (512,512,3))
    
    # Compile model
    # model1.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #           optimizer = tf.keras.optimizers.Adam(),
    #           metrics = ['accuracy']
    #           )
    metrics = [dice_coef,iou,Recall(),Precision()]
    model.compile(loss = dice_loss,optimizer = Adam(learning_rate=lr))
    
    # Train model
    history = model1.fit(X_train,y_train,
            epochs = 10,
            batch_size = 2, # OOM problem
            validation_data = (X_valid,y_valid),
            validation_steps = int(0.1*len(y_valid)),
            )
    
    # Save model
    pd.Dataframe(history.history).plot()
    plt.savefig()

    
    
