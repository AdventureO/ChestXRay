#from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetMobile
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os

def set_keras_backend(backend):

    if k.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(k)
        assert k.backend() == backend

set_keras_backend("tensorflow")

train_data_dir = '/gpfs/rocket/dmytro/xray/data/train'
validation_data_dir = '/gpfs/rocket/dmytro/xray/data/validation'
nb_validation_samples = 10341 + 12082
nb_train_samples = 112121 - nb_validation_samples
epochs = 20
batch_size = 200
img_width, img_height = 224, 224

model = NASNetMobile(weights = "imagenet", include_top=False, input_shape = (img_width, img_height,3))
# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers:
    layer.trainable = False


#Adding custom Layers 
x = model.output
x = Flatten()(x)
#x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "binary_crossentropy", 
                    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), 
                    metrics=["accuracy"]
                   )


checkpoint = ModelCheckpoint("ResNet50.h5", monitor='val_acc', verbose=1, 
                             save_best_only=True, save_weights_only=False, 
                             mode='auto', period=1
                            )
early = EarlyStopping(monitor='val_acc', 
                      min_delta=0, patience=10, 
                      verbose=1, mode='auto'
                     )


test_datagen = ImageDataGenerator()
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
                                                    train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                                                        validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

model_final.fit_generator(
                            train_generator,
                            samples_per_epoch = 500,#nb_train_samples,
                            epochs = epochs,
                            validation_data = validation_generator,
                            nb_val_samples = 100,
                            callbacks = [checkpoint, early]
                            )
