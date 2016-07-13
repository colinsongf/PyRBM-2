# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 23:18:37 2016

@author: Rony

My first Feedforward Neural Network
"""


import numpy as np


# Global settings
np.random.seed(14)


class PyRBM():

    def __init__(self, numOfVisibleUnits, numOfHiddenUnits, learningRate = 0.1):

        self.XTrain = np.array([])
        self.yTrain = np.array([])
        self.XValid = np.array([])
        self.yValid = np.array([])
        self.XTest  = np.array([])
        self.yTest  = np.array([])

        self.numOfVisibleUnits  = numOfVisibleUnits
        self.numOfHiddenUnits   = numOfHiddenUnits
        self.learningRate       = learningRate

        # Initialize a weight matrix, of dimensions (numOfVisibleUnits x numOfHiddenUnits), using a Gaussian
        # distribution with mean 0 and standard deviation 0.1
        self.weights = 0.01 * np.random.randn(self.numOfVisibleUnits, self.numOfHiddenUnits)
        # Insert weights for the bias units into the first row and first column
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def readTrainingSetIn(self):

        # Read Training Data in
        with open('/Users/Rony/Documents/PyFFNN/Training Data.csv') as trainingDataFile:
            trainingData = trainingDataFile.read().splitlines()

        XTemp = []
        X = []
        y = []

        for line in trainingData:
            lineMod = str(line)
            XTemp.append(lineMod.split(',')[:5])
            y.append(int(lineMod.split(',')[5]))

        for arr in XTemp:
            newArr = []
            for elem in arr:
                newArr.append(float(elem))
            X.append(newArr)

        # Convert the arrays into numpy arrays
        self.XTrain = np.array(X)
        self.yTrain = np.array(y)
    def readValidationSetIn(self):

        # Read Validation Data in
        with open('/Users/Rony/Documents/PyFFNN/Validation Data.csv') as validationDataFile:
            validationData = validationDataFile.read().splitlines()

        XTemp = []
        X = []
        y = []

        for line in validationData:
            lineMod = str(line)
            XTemp.append(lineMod.split(',')[:5])
            y.append(int(lineMod.split(',')[5]))

        for arr in XTemp:
            newArr = []
            for elem in arr:
                newArr.append(float(elem))
            X.append(newArr)

        # Convert the arrays into numpy arrays
        self.XValid = np.array(X)
        self.yValid = np.array(y)
    def readTestSetIn(self):

        # Read Testing Data in
        with open('/Users/Rony/Documents/PyFFNN/Test Data.csv') as testDataFile:
            testData = testDataFile.read().splitlines()

        XTemp = []
        X = []
        y = []

        for line in testData:
            lineMod = str(line)
            XTemp.append(lineMod.split(',')[:5])
            y.append(int(lineMod.split(',')[5]))

        for arr in XTemp:
            newArr = []
            for elem in arr:
                newArr.append(float(elem))
            X.append(newArr)

        # Convert the arrays into numpy arrays
        self.XTest = np.array(X)
        self.yTest = np.array(y)

    def logisticActivationFunc(self, x):

        return 1.0 / (1 + np.exp(-x))

    def train(self, numOfEpochs = 1000):

        numOfTrainingExamples = len(self.XTrain)

        # Insert bias units of 1 into the first column
        self.XTrain = np.insert(self.XTrain, 0, 1, axis = 1)

        for epoch in range(numOfEpochs):
            # Clamp to the data and sample from the hidden units
            # (This is the "positive CD phase", aka the reality phase.)
            posHiddenActivations    = np.dot(self.XTrain, self.weights)
            posHiddenProbs          = self.logisticActivationFunc(posHiddenActivations)
            posHiddenStates         = posHiddenProbs > np.random.randn(numOfTrainingExamples, self.numOfHiddenUnits + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            posAssociations         = np.dot(self.XTrain.T, posHiddenProbs)

            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            negVisibleActivations   = np.dot(posHiddenStates, self.weights.T)
            negVisibleProbs         = self.logisticActivationFunc(negVisibleActivations)
            negVisibleProbs[:, 0]   = 1  # Fix the bias unit
            negHiddenActivations    = np.dot(negVisibleProbs, self.weights)
            negHiddenProbs          = self.logisticActivationFunc(negHiddenActivations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states
            # themselves
            negAssociations         = np.dot(negVisibleProbs.T, negHiddenProbs)

            # Update weights
            self.weights            += self.learningRate * ((posAssociations - negAssociations) / numOfTrainingExamples)

            error                   = np.sum((self.XTrain - negVisibleProbs) ** 2)
            print("Epoch %s: error is %s" % (epoch, error))

        print self.XTrain
    def validate(self):

        self.readValidationSetIn()
        return self.runVisibleUnits(self.XValid)
    def runVisibleUnits(self, data):

            # Assuming the RBM has been trained (so that weights for the network have been learned), run the network
            # on a set of visible units, to get a sample of the hidden units
            # Returns hidden states in the form of a matrix where each row consists of the hidden units activated from
            # the visible units in the data matrix passed in

            numOfExamples       = len(data)

            # Create a matrix, where each row is to be the hidden units (plus a bias unit)
            # sampled from a training example
            hiddenStates        = np.ones((numOfExamples, self.numOfHiddenUnits + 1))

            # Insert bias units of 1 into the first column of data
            data                = np.insert(data, 0, 1, axis = 1)

            # Compute the activations of the hidden units
            hiddenActivations   = np.dot(data, self.weights)
            # Compute the probabilities of turning the hidden units on
            hiddenProbs         = self.logisticActivationFunc(hiddenActivations)
            # Turn the hidden units on with their specified probabilities
            hiddenStates[:, :]  = hiddenProbs > np.random.randn(numOfExamples, self.numOfHiddenUnits + 1)
            #Ignore the bias units
            hiddenStates        = hiddenStates[:, 1:]

            return hiddenStates

myRBM = PyRBM(numOfVisibleUnits = 5, numOfHiddenUnits = 3)
myRBM.readTrainingSetIn()
myRBM.train()
myRBM.validate()