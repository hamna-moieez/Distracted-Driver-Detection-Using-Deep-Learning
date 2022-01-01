
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image

import numpy as np
import model

def vgg(input_shape, num_classes, mode):
    vgg = VGG16(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=input_shape, pooling=None, classes=num_classes,
        classifier_activation='softmax'
        )
    if mode=='feature-extraction':
        vgg.trainable = False
    if mode=='fine-tune':
        vgg.trainable = True
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(vgg.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 10

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in vgg.layers[:fine_tune_at]:
            layer.trainable =  False
    print(vgg.summary())
    return vgg
