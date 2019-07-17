import numpy as np
import keras

class Network():

    def __init__(self):
        
        self.x_train, self.y_train, self.x_test, self.y_test = self.getData()
        self.cnnModel = None
        self.dffModel = None

    def createCnnModel(self):

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(6, 2, input_shape=(28, 28, 1)))
        model.add(keras.layers.Activation(keras.activations.relu))
        model.add(keras.layers.MaxPool2D(2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Activation(keras.activations.softmax))

        model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(), 
                metrics=['acc'])

        model.fit(self.x_train, self.y_train, batch_size=32, epochs=10, validation_data=(self.x_test,self.y_test))

        self.cnnModel = model

    def createDffModel(self):
    
        model = keras.models.Sequential()

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation=keras.activations.relu))
        model.add(keras.layers.Dense(128, activation=keras.activations.relu))
        model.add(keras.layers.Dense(64, activation=keras.activations.relu))
        model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

        model.compile(optimizer=keras.optimizers.Adam(), 
                    loss=keras.losses.sparse_categorical_crossentropy,
                    metrics=['acc'])
        
        model.fit(self.x_train, self.y_train, batch_size=32, epochs=10, validation_data=(self.x_test,self.y_test))
       
        self.dffModel = model

    '''def train(self, model):
    
        model.fit(self.x_train, self.y_train, batch_size=32, epochs=10, validation_data=(self.x_test,self.y_test))
    '''
    def save(self, model, path:str):

        model.save(path)

    def getData(self):

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = keras.utils.normalize(x_train, axis=1)
        x_test = keras.utils.normalize(x_test, axis=1)
        x_train = np.expand_dims(x_train,-1)
        x_test = np.expand_dims(x_test,-1)

        return (x_train, y_train, x_test, y_test)

if __name__ == '__main__':

    nn = Network()
    nn.createCnnModel()
    nn.createDffModel()
    nn.save(nn.cnnModel,'cnn_test.model')
    nn.save(nn.dffModel,'dff_test.model')

