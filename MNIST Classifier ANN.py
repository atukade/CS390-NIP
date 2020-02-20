#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

# Disable some troublesome logging.
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        s = self.__sigmoid(x)
        return s*(1-s)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 5, minibatches = True, mbs = 100):
        X = xVals.reshape(xVals.shape[0], -1)
        Y = yVals      
        
        for i in range(epochs):
            batch_x = self.__batchGenerator(X, mbs)
            batch_y = self.__batchGenerator(Y, mbs)
            
            for j in range(0, xVals.shape[0], mbs):               
                X_mini = next(batch_x)
                Y_mini = next(batch_y)
                A1, A2 = self.__forward(X_mini)


                L2e = (Y_mini - A2)
                L2d = L2e * self.__sigmoidDerivative(A2)
                L1e = np.dot(L2d, self.W2.T)
                L1d = L1e * self.__sigmoidDerivative(A1)
                L1a = np.dot(X_mini.T, L1d)*self.lr
                L2a = np.dot(A1.T, L2d)*self.lr
            
                self.W1 += L1a
                self.W2 +=L2a
            

            print("Epoch", i)            
        

    # Forward pass.
    def __forward(self, input):
        Z1 = np.dot(input, self.W1)
        layer1 = self.__sigmoid(Z1)
        Z2 = np.dot(layer1, self.W2)
        layer2 = self.__sigmoid(Z2)
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2
        
    
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

def buildANN():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_dim=784))    
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metric=['accuracy'])
    return model


def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xTrain, xTest = xTrain/255.0, xTest/255.0
    xTrain = xTrain.reshape((-1, 784))
    xTest = xTest.reshape((-1,784))
    
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")       
        model = NeuralNetwork_3Layer(IMAGE_SIZE, NUM_CLASSES, 256)
        model.train(xTrain, yTrain)
        return model
    
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = buildANN()
        model.fit(xTrain, yTrain, epochs = 5)
        return model

    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        preds = model.predict(data)
        b = np.zeros_like(preds)
        b[np.arange(len(preds)), preds.argmax(1)] = 1.
        return b
        
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        b = np.zeros_like(preds)
        b[np.arange(len(preds)), preds.argmax(1)] = 1.
        return b
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0

    normal = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    a = np.asarray(yTest)
    b = np.asarray(preds)
    a = np.dot(a, normal.T)
    b = np.dot(b, normal.T)
    a = a.astype(int)
    b = b.astype(int)
    
    
    matrix = np.zeros(shape=(10, 10)).astype(int)
    precision = np.zeros(10)
    recall = np.zeros(10)
    F1 = np.zeros(10)
    
    
    import pandas
    
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        for j in range(10):
            for k in range(10):
                if (a[i] == j and b[i] == k):
                    matrix[j][k] += 1

    for i in range(10):
        prec = 0
        rec = 0
        for j in range(10):
            prec += matrix[j][i]
            rec += matrix[i][j]
        precision[i] = matrix[i][i]/prec
        recall[i] = matrix[i][i]/rec

    for i in range(10):
        F1[i] = 2*(precision[i]*recall[i])/(precision[i] + recall[i])
    
        
    row_labels = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7', 'a_8', 'a_9']
    column_labels = ['p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8', 'p_9']
    print("a = Actual, p = Predicted")
    df = pandas.DataFrame(matrix, columns=column_labels, index=row_labels)
    index = ['F1 Score']
    f1_df = pandas.DataFrame(F1, columns=index)
    print(df)
    print(f1_df)
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


class NeuralNetwork_3Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer)
        self.W3 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        
    # Activation function.
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        s = self.__sigmoid(x)
        return s*(1-s)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n] 
            
    def train(self, xVals, yVals, epochs = 5, minibatches = True, mbs = 100):
        X = xVals.reshape(xVals.shape[0], -1)
        Y = yVals      
        
        L1a = 0
        L2a = 0
        L3a = 0
        cost = 0
        
        for i in range(epochs):
            batch_x = self.__batchGenerator(X, mbs)
            batch_y = self.__batchGenerator(Y, mbs)
            
            for j in range(0, xVals.shape[0], mbs):               
                X_mini = next(batch_x)
                Y_mini = next(batch_y)
                A1, A2, A3 = self.__forward(X_mini)

                L3e = (Y_mini - A3)
                #L2e = cost
                L3d = L3e * self.__sigmoidDerivative(A3)
                L2e = np.dot(L3d, self.W3.T)
                L2d = L2e * self.__sigmoidDerivative(A2)
                L1e = np.dot(L2d, self.W2.T)
                L1d = L1e * self.__sigmoidDerivative(A1)
                
                
                L1a = np.dot(X_mini.T, L1d)*self.lr
                L2a = np.dot(A1.T, L2d)*self.lr
                L3a = np.dot(A2.T, L3d)*self.lr
            
                self.W1 += L1a
                self.W2 +=L2a
                self.W3 += L3a
            

            print("Epoch", i)            
        

    # Forward pass.
    def __forward(self, input):
        Z1 = np.dot(input, self.W1)
        layer1 = self.__sigmoid(Z1)
        Z2 = np.dot(layer1, self.W2)
        layer2 = self.__sigmoid(Z2)
        Z3 = np.dot(layer2, self.W3)
        layer3 = self.__sigmoid(Z3)
        return layer1, layer2, layer3

    # Predict.
    def predict(self, xVals):
        _, _, layer3 = self.__forward(xVals)
        return layer3


def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    #data[1][0]
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()





