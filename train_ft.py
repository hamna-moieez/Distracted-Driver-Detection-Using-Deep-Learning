
import utils
import model
import vgg
import transferlearn
import os
import tensorflow as tf
import pickle

NUM_CLASSES = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
INPUT_SHAPE = (224,224,3)
weights_path = os.path.join(utils.BASE_PATH, 'weights/')
history_path = os.path.join(utils.BASE_PATH, 'history/')
def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    print(model.summary())
    return model


def fit_model(model, initial_epochs, training, valid):
    history = model.fit(training,
                    epochs=initial_epochs,
                    validation_data=valid)
    return history

def save_weights(model, name):
    model.save(weights_path+name+".h5")

def save_history(history, file_name):
    with open(file_name, 'wb') as file_p:
        pickle.dump(history.history, file_p)

train_data, val_data = utils.get_data()


'''
TRAIN KERAS BUILTIN VGG16 WITH FINETUNED LAYERS

One way to increase performance even further is to train (or "fine-tune") 
the weights of the top layers of the pre-trained model alongside the training 
of the classifier you added. The training process will force the weights to be 
tuned from generic feature maps to features associated specifically with the dataset.

Also, fine-tune a small number of top layers rather than the whole base model. 
In most convolutional networks, the higher up a layer is, the more specialized it is. 
The first few layers learn simple & generic features that generalize to almost all types of images. 
As you go higher up, the features are increasingly more specific to the dataset on which the model was trained. 
The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.
'''

fine_tune_epochs = 10
total_epochs =  NUM_EPOCHS + fine_tune_epochs
vgg16_fine = vgg.vgg(INPUT_SHAPE, NUM_CLASSES, 'fine-tune')
fine_model = compile_model(transferlearn.transfer_learn(vgg16_fine, INPUT_SHAPE, NUM_CLASSES), LEARNING_RATE)
fine_history = fit_model(fine_model, total_epochs, train_data, val_data)
save_weights(fine_model, 'fineTune')
save_history(fine_history, history_path+'fineTune')
