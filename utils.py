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
        

BASE_PATH = '/Users/hamnamoieez/Desktop/PRDL/Distracted-Driver-Detection-Using-Deep-Learning/'
data_path = os.path.join(BASE_PATH, 'imgs/train')

BATCH_SIZE = 8
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

def plot_metrics(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # Save accuracy figure
        plt.savefig('accuracy.png')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # Save loss figure
        plt.savefig('loss.png')
        plt.show()
        return acc, val_acc, loss, val_loss

def plot_finetuned(history_fine, initial_epochs, acc, val_acc, loss, val_loss):
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([initial_epochs-1,initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([initial_epochs-1,initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
        plt.savefig('metrics_finetuned.png')

