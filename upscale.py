#UPSCALE.PY

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers, Model
import tensorlayer as tl
import time
import pathlib
from matplotlib import pyplot as plt
from skimage import measure

str_data_dir = './dataset/train'
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_dir = pathlib.Path(str_data_dir)
image_count = len(list(data_dir.glob('*.jpg')))

BATCH_SIZE = 1
HR_IMG_HEIGHT = 96
HR_IMG_WIDTH = 96
DOWNSAMPLING_FACTOR = 4

LR_IMG_HEIGHT = HR_IMG_HEIGHT/DOWNSAMPLING_FACTOR
LR_IMG_WIDTH = HR_IMG_WIDTH/DOWNSAMPLING_FACTOR

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
NUM_CHANNELS = 3
B = 4

from subpixel import SubpixelConv2D, Subpixel

def make_sr_generator_model():
    lr_img = layers.Input(shape=(None, None, NUM_CHANNELS))
    #lr_img = layers.Input(shape=(1080, 1080, NUM_CHANNELS))  
    ################################################################################                                                                                      
    ## Now that we have a low res image, we can start the actual generator ResNet ##                                                                                      
    ################################################################################                                                                                      
    
    x = layers.Convolution2D(64, (9,9), (1,1), padding='same')(lr_img)
    x = layers.ReLU()(x)

    b_prev = x

    #####################                                                                                                                                                 
    ## Residual Blocks ##                                                                                                                                                 
    #####################                                                                                                                                                 

    for i in range(B):
      b_curr = layers.Convolution2D(64, (3,3), (1,1), padding='same')(b_prev)
      b_curr = layers.BatchNormalization()(b_curr)
      b_curr = layers.ReLU()(b_curr)
      b_curr = layers.Convolution2D(64, (3,3), (1,1), padding='same')(b_curr)
      b_curr = layers.BatchNormalization()(b_curr)
      b_curr = layers.Add()([b_prev, b_curr]) #skip connection                                                                                                            

      b_prev = b_curr

    res_out = b_curr # Output of residual blocks                                                                                                                          

    x2 = layers.Convolution2D(64, (3,3), (1,1), padding='same')(res_out)
    x2 = layers.BatchNormalization()(x2)
    x = layers.Add()([x, x2]) #skip connection      


    #######################################################                                                                                                               
    ## Resolution-enhancing sub-pixel convolution layers ##                                                                                                               
    #######################################################                                                                                                               

    # Layer 1 (Half of the upsampling)                                                                                                                                    
    x = layers.Convolution2D(256, (3,3), (1,1), padding='same')(res_out)
    x = SubpixelConv2D(input_shape=(None, None, None, NUM_CHANNELS), scale=DOWNSAMPLING_FACTOR/2, idx=0)(x)
    #x = Subpixel(256, kernel_size=(3,3), r=DOWNSAMPLING_FACTOR/2, padding='same', strides=(1,1))                                                                         
    x = layers.ReLU()(x)

    # Layer 2 (Second half of the upsampling)                                                                                                                             
    x = layers.Convolution2D(256, (3,3), (1,1), padding='same')(x)
    x = SubpixelConv2D(input_shape=(None, None, None, NUM_CHANNELS/((DOWNSAMPLING_FACTOR/2) ** 2)), scale=(DOWNSAMPLING_FACTOR/2), idx=1)(x)
    #x = Subpixel(256, kernel_size=(3,3), r=DOWNSAMPLING_FACTOR/2, padding='same', strides=(1,1))                                                                         
    x = layers.ReLU()(x)

    generated_sr_image = layers.Convolution2D(3, (9,9), (1,1), padding='same')(x)
    output_shape = generated_sr_image.get_shape().as_list()
    #assert output_shape == [None, HR_IMG_HEIGHT, HR_IMG_WIDTH, NUM_CHANNELS]
    return Model(inputs=lr_img, outputs=generated_sr_image, name='generator')

def make_sr_discriminator_model():
    inputs = layers.Input(shape=(HR_IMG_HEIGHT, HR_IMG_WIDTH, NUM_CHANNELS))
    # k3n64s1                                                                                                                                                             
    x = layers.Convolution2D(64, (3,3), (1,1), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #################                                                                                                                                                     
    ## Conv Blocks ##                                                                                                                                                     
    #################                                                                                                                                                     

    # k3n64s2                                                                                                                                                             
    x = layers.Convolution2D(64, (3,3), (2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n128s1                                                                                                                                                            
    x = layers.Convolution2D(128, (3,3), (1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n128s2                                                                                                                                                            
    x = layers.Convolution2D(128, (3,3), (2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n256s1                                                                                                                                                            
    x = layers.Convolution2D(256, (3,3), (1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n256s2                                                                                                                                                            
    x = layers.Convolution2D(256, (3,3), (2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n512s1                                                                                                                                                            
    x = layers.Convolution2D(512, (3,3), (1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # k3n512s2                                                                                                                                                            
    x = layers.Convolution2D(512, (3,3), (2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    ################                                                                                                                                                      
    ## Dense Tail ##                                                                                                                                                      
    ################                      
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs, name='discriminator')


discriminator = make_sr_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator = make_sr_generator_model()
#generator.summary()

checkpoint_dir = './vggan_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)
#status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#generator.summary()
#print(generator.layers[0])
#input_layer = tf.keras.layers.InputLayer(input_shape=(1080,1080,3), name = "input-1")
#generator.layers[0] = input_layer
#print(generator.layers[28])
#print(generator.layers[31])

#generator.layers[28] = SubpixelConv2D(input_shape=(None, 1080, 1080, NUM_CHANNELS), scale=DOWNSAMPLING_FACTOR/2, idx=0)
#generator.layers[31] = SubpixelConv2D(input_shape=(None, 1080*(DOWNSAMPLING_FACTOR/2), 1080*(DOWNSAMPLING_FACTOR/2), NUM_CHANNELS/((DOWNSAMPLING_FACTOR/2) ** 2)), scale=(DOWNSAMPLING_FACTOR/2), idx=1)
    
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor                                                                                                                                                   
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.                                                                                                                                   
  img = tf.image.convert_image_dtype(img, tf.float32)
  return img
  # resize the image to the desired size.                                                                                                                                                                
  #return tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=0, target_height=IMG_HEIGHT, target_width=IMG_WIDTH)                                                                         
 # return tf.image.random_crop(img,[270,270,3])


def process_path(file_path):
  # load the raw data from the file as a string                                                                                                                                                          
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img


def normalize(t):
  return (t - tf.reduce_min(t))/(tf.reduce_max(t)-tf.reduce_min(t))

#tf.print("Generating low res images")
#lr_images = process_path('./dataset/dev/Blur13_1/1_gb13.jpg')

def generate_gaussian_kernel(shape=(3,3),sigma=0.8):
    """                                                                                                                                                                                                     
    2D gaussian mask - should give the same result as MATLAB's                                                                                                                                              
    fspecial('gaussian',[shape],[sigma])                                                                                                                                                                    
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# Gaussian Blur Setup
BLUR_KERNEL_SIZE = 3
kernel_weights = generate_gaussian_kernel()

# Size compatibility code
kernel_weights = np.expand_dims(kernel_weights, axis=-1)
kernel_weights = np.repeat(kernel_weights, NUM_CHANNELS, axis=-1) # apply the same filter on all the input channels
kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons

# Blur
blur_layer = layers.DepthwiseConv2D(BLUR_KERNEL_SIZE, use_bias=False, padding='same')

# Downsample
downsample_layer = layers.AveragePooling2D(pool_size=(DOWNSAMPLING_FACTOR, DOWNSAMPLING_FACTOR))

############################################# BLUR AND DOWNSAMPLE LAYERS #############################################


################################################### MODEL CREATION ##################################################

def make_downsampler_model():
        hr_img = layers.Input(shape=(None, None, NUM_CHANNELS))
        lr_img = blur_layer(hr_img)
        lr_img = downsample_layer(lr_img)
        return Model(inputs=hr_img, outputs=lr_img, name='downsampler')

downsampler = make_downsampler_model()
blur_layer.set_weights([kernel_weights])
blur_layer.trainable = False  # the weights should not change during training     



path = 'dataset/dev/Orig/5_orig.jpg'

lr_images = process_path(path)
lr_images_input = tf.expand_dims(lr_images, 0)
lr_images_input =  tf.image.crop_to_bounding_box(lr_images_input, 0, 0, 1080, 1080)

print(lr_images_input)

lr_images = downsampler(lr_images_input, training = False)
print(lr_images)

#PIL.Image.fromarray(np.asarray(lr_images)).show()


#plt.imsave("lr_images2.png", np.array(lr_images, dtype=np.float32))
#uint8

print("Generating image")

#lr_images_input = tf.expand_dims(lr_images, 0)
generated_images = generator(lr_images, training=False)
generated_images = normalize(generated_images)

print("Image Generated")

plt.imsave("original_cropped.png", np.array(lr_images_input[0], dtype=np.float32))
#print(generated_images)
plt.imsave("downsample_cropped.png", np.array(lr_images[0], dtype=np.float32))
ds_size = tf.image.resize(lr_images, (1080, 1080))
print(ds_size.shape)
plt.imsave("downsample_resized.png", np.array(ds_size[0], dtype=np.float32))
generated_images = tf.image.adjust_saturation(generated_images, 0.5)
plt.imsave("gen_cropped.png", np.array(generated_images[0], dtype=np.float32))

print(generated_images.shape, lr_images_input.shape)
s = measure.compare_ssim(np.array(lr_images_input[0], dtype=np.float32), np.array(generated_images[0], dtype=np.float32), multichannel = True)
d = measure.compare_ssim(np.array(ds_size[0], dtype=np.float32), np.array(generated_images[0], dtype=np.float32), multichannel = True)
print("structural similarity between input and output of model is:" + str(s))
print("structural similarity between input and output of model is:" + str(d))
