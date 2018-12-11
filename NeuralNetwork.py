import numpy as np
from ctypes.wintypes import DOUBLE

RAND_MAX = 32767

class OutputWeights():
    __name__ = ""
    outputWeights = []
    
    def __init__(self):
        self.__name__ = ""
        
    def setName(self, name):
        self.__name__ = name

class Connection():
    __name__ = ""
    weight = 0.0
    deltaWeight = 0.0
        
    def __init__(self, name):
        self.__name = name
        
    def getWeight(self):
        return self.weight
    
    def setWeight(self, weight):
        self.weight = weight
        
    def getDeltaWeight(self):
        return self.getDeltaWeight()
    
    def setDeltaWeight(self, deltaWeight):
        self.deltaWeight = deltaWeight
            
class Neuron():
    __name__ = ""
    Cnxn = ""
    alpha = DOUBLE(0)
    eta = DOUBLE(0)
    gradient = DOUBLE(0)
    idx = DOUBLE(0)
    outputValue = DOUBLE(0)
    ow = OutputWeights()
    outputWeights = ow.__init__()
    
    def __init__(self, name):
        self.__name__ = name
        cx = Connection()
        self.Cnxn = cx.__init__(self.__name__)
        self.outputWeights.setName(self.__name__)
        
    def getOutputValue(self):
        return self.outputValue
    
    def setOutputValue(self, desiredValue):
        self.outputValue = desiredValue

    def randomWeight(self):
        return np.random.rand() / DOUBLE(RAND_MAX)

    def sumDOW(self, nextLayer):
        ttl = DOUBLE(0)
        for item in nextLayer:
            ttl += self.outputWeights[item].weight * nextLayer[item].gradient
        return ttl

    def transferFunction (self, x):
        return np.tanh(x)

    def updateInputWeights(self, prevLayer):
        for item in prevLayer:
            neuron = item
            oldDeltaWeight = neuron.outputWeights[item].deltaWeight
            newDeltaWeight = self.eta * neuron.getOutputValue() * self.gradient + self.alpha * oldDeltaWeight
            neuron.outputWeights[item].deltaWeight = newDeltaWeight
            neuron.outputWeights[item].weight += newDeltaWeight

    def feedForward(self, prevLayer):
        ttl = 0.0
        for item in prevLayer:
            ttl += item.getOutputValue()
    
    def calcOutputGradient(self, targetValue):
        delta = targetValue - self.outputValue
        gradient = delta * self.tranferFunctionDerivative(self.outputValue)
    
    def calcHiddenGradients(self, nextLayer):
        self.dow = self.sumDOW(nextLayer)
        self.gradient = self.dow * self.transferFunctionDerivative(self.outputValue)
    
    def main(self, numOutputs, desiredIndex):
        for output in numOutputs:
            output.push_back(Connection())
            outputWeights.back().weight = randomWeight()
        index = desiredIndex
            
