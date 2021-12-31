import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import image_dataset_from_directory


def view_image(img_path):
        img = load_img(img_path)  # this is a PIL image
        plt.imshow(img)
        plt.show()
        

DATASET_PATH = '/Users/hamnamoieez/Desktop/PRDL/Distracted-Driver-Detection-Using-Deep-Learning/imgs/'
data_path = os.path.join(DATASET_PATH, 'train')


BATCH_SIZE = 16
IMG_SIZE = (224, 224)

def prepare_data(subset):
        dataset = image_dataset_from_directory(data_path,
                                                label_mode='categorical',
                                                color_mode='rgb',
                                                shuffle=True,
                                                seed=231,
                                                validation_split=0.2,
                                                subset=subset,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE)

        return dataset
def get_data():
        train = prepare_data('training')
        validation = prepare_data('validation')
        class_names = train.class_names
        print("List of Classes: ",class_names)


        for image_batch, labels_batch in train:
                print("Images batch shape: ", image_batch.shape)
                print("Labels batch shape: ", labels_batch.shape)
                break
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train.cache().prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation.cache().prefetch(buffer_size=AUTOTUNE)

        return train_dataset, validation_dataset

def augment_data(input):
        x = tf.keras.layers.RandomFlip('horizontal')(input)
        data_augmentation = tf.keras.layers.RandomRotation(0.2)(x)
        return data_augmentation
