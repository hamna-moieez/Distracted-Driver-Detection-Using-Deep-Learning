
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image

import numpy as np
import model

def vgg(input_shape, num_classes):
    vgg = VGG16(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=input_shape, pooling=None, classes=num_classes,
        classifier_activation='softmax'
        )
    print(vgg.summary())
    return vgg


