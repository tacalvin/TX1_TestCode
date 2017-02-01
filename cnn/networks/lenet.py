from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class lenet:
    """The LeNet Class"""
    def build(width, height, depth, classes, weightPath=None):
        """Builds the LeNet with paramters"""
        #The width of our input images.
        #The height of our input images.
        #The depth (i.e., number of channels) of our input images.
        #And the number of classes (i.e., unique number of class labels) in our dataset.
        model = Sequential()

        #first set of CONV => RELU => POOL
        model.add(Convolution2D(20,5,5,border_mode="same",input_shape=(depth,height,width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        #FOR CNN you typically increase the CONV filter sizes as you go further
        #second set of CONV => RELU => POOL
        model.add(Convolution2D(50,5,5,border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        #FullyConnected Layers (FC) => RELU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        #Softmax classifer
        model.add(Dense(classes))
        model.add(Activation("softmax"))


        #If a weightsPath are provided is supplied use those weights
        if weightPath is not None:
            model.load_weights(weightPath)

        #return model
        return model


