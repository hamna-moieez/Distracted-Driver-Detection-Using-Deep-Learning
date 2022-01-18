
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
TRAIN KERAS BUILTIN VGG16 WITH FEATURE EXTRACTION

Only training a few layers on top of the base model. 
The weights of the pre-trained network were not updated during training.
'''
vgg16 = vgg.vgg(INPUT_SHAPE, NUM_CLASSES, 'feature-extraction')
fe_model = compile_model(transferlearn.transfer_learn(vgg16, INPUT_SHAPE, NUM_CLASSES), LEARNING_RATE)
fe_history = fit_model(fe_model, NUM_EPOCHS, train_data, val_data)
save_weights(fe_model, 'featureExtractor')
save_history(fe_history, history_path+'featureExtractor')
