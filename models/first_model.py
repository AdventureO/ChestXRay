import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D, Convolution2D,ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D, MaxPooling3D
from keras.layers import AveragePooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, Conv1D
from keras.layers import Convolution2D,MaxPooling2D
from keras.optimizers import SGD, Adam,Adagrad,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.np_utils import to_categorical
K.set_image_dim_ordering('th')
seed = 7
numpy.random.seed(seed)



def net_builder(num_classes,input_shape):
    """
    Returns convolutional net.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3,input_shape, input_shape), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    #model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(AveragePooling2D(pool_size=(2,2),strides=1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def simple_net(nb_classes, img_shape):
    model = Sequential()
    model.add(Conv2D(32,(3,3), input_shape = (3,64,64), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(16,(5,5), activation='relu')) # used tanh activation as in the original
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(120,(5,5), activation='relu'))

    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 

    model.compile(loss="binary_crossentropy", optimizer=opt,metrics=['accuracy'])

    return model

img_width, img_height = 64, 64

train_data_dir = '/gpfs/rocket/dmytro/xray/data/train'
validation_data_dir = '/gpfs/rocket/dmytro/xray/data/validation'
nb_validation_samples = 10341 + 12082
nb_train_samples = 112121 - nb_validation_samples
epochs = 20
batch_size = 100

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

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
model = simple_net(nb_classes=1, img_shape=64)

model.fit_generator(
    train_generator,
    steps_per_epoch=150, #nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
validation_steps=nb_validation_samples // batch_size)

model.save_weights('/gpfs/hpchome/dobko/xray/saved_models/first_try.h5')
