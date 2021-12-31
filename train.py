
import utils
import model
import vgg
import finetune

import tensorflow as tf

NUM_CLASSES = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
INPUT_SHAPE = (224,224,3)


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

train_data, val_data = utils.get_data()

'''
TRAIN CUSTOM MODEL
'''
# custom_model = model.create_model(INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE)
# history = fit_model(custom_model, NUM_EPOCHS, train_data, val_data)

'''
TRAIN KERAS BUILTIN VGG16 WITH FINETUNED TOP
'''
vgg16 = vgg.vgg(INPUT_SHAPE, NUM_CLASSES)
finetuned_model = compile_model(finetune.finetune(vgg16, INPUT_SHAPE, NUM_CLASSES), LEARNING_RATE)
fit_model(finetuned_model, NUM_EPOCHS, train_data, val_data)

#TODO: Transfer learning and fine tuning over other networks
#TODO: Test on Deep Learning Architectures other than CNN's
#TODO: TCAV, Saliency maps