from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,\
    MaxPooling2D,Conv2DTranspose,Concatenate,Input

from tensorflow.keras.models import Model

from tensorflow.keras.utils import plot_model

def conv_block(input,num_filters):
    '''
    Create a block of convolutional layers conv-conv
    input --- input layer
    num_filters --- (int) number of filter in covolutional layer
    Return block of conv layer
     '''
    x = Conv2D(num_filters,3,padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
     
    x = Conv2D(num_filters,3,padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
     
    return x
 
def encoder_block(input,num_filters):
    '''
    Create encoder block
    input --- input layer
    num_filters --- (int) number of filter in covolutional layer
    Return feature layer and maxpooling layer
    '''
    f = conv_block(input,num_filters) # features
    p = MaxPooling2D(pool_size =(2,2))(f)
    return f,p
 
def decoder_block(input,skip_features,num_filters):
    '''
    Create decoder block
    input --- input layer
    skip_features --- the feature layer from encoder block for concatenate to decoder block
    num_filters --- (int) number of filter in covolutional layer
    Return decoder block
    '''
    x = Conv2DTranspose(num_filters,(2,2),strides = 2,padding = 'same')(input)
    x =  Concatenate()([x,skip_features])
    x = conv_block(x,num_filters)
    return x 
 
def build_unet(input_shape):
    inputs = Input(input_shape)
    # encode
    f1,p1 = encoder_block(inputs,64)
    f2,p2 = encoder_block(p1,128) 
    f3,p3 = encoder_block(p2,256) 
    f4,p4 = encoder_block(p3,512)
    
    b1 = conv_block(p4,1024)
    
    #decode
    d1 = decoder_block(b1,f4,512)
    d2 = decoder_block(d1,f3,512)
    d3 = decoder_block(d2,f2,512)
    d4 = decoder_block(d3,f1,512)
    
    outputs = Conv2D(1,1,padding ='same',activation = 'sigmoid')(d4)
    
    model = Model(inputs,outputs,name = 'U-Net')
    return model
 
if __name__ == '__main__':
    input_shape = (512,512,3)
    model = build_unet(input_shape)
    model.summary()
    plot_model(model,show_shapes = True)
     
    