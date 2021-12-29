
import utils
import model
import vgg

NUM_CLASSES = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
INPUT_SHAPE = (224,224,3)

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
TRAIN KERAS BUILTIN VGG16 (IF REQUIRED)
'''
vgg16 = vgg.vgg(INPUT_SHAPE, NUM_CLASSES)
# model.fit_model(vgg(), model.NUM_EPOCHS, train, val)

#TODO: Transfer learning and fine tuning over VGG16
#TODO: Test on Deep Learning Architectures other than CNN's
