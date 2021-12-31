import utils
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

def finetune(base_model, input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = utils.augment_data(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs)
    return model

