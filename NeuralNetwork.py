import numpy as np
from ctypes.wintypes import DOUBLE

RAND_MAX = 32767

class OutputWeights():
    name = ""
    outputWeights = []
    
    def __init__(self, name, outputWeight):
        self.name = name
        self.outputWeights.append(outputWeight)
        
class Connection():
    name = ""
    weight = 0.0
    deltaWeight = 0.0
        
    def __init__(self, name):
        self.name = name
        
    def getWeight(self):
        return self.weight
    
    def setWeight(self, weight):
        self.weight = weight
        
    def getDeltaWeight(self):
        return self.getDeltaWeight()
    
    def setDeltaWeight(self, deltaWeight):
        self.deltaWeight = deltaWeight
            
class Neuron():
    name = ""
    numberOfOutputs = 0
    connections = []
    alpha = DOUBLE(0)
    eta = DOUBLE(0)
    gradient = DOUBLE(0)
    idx = DOUBLE(0)
    outputValue = DOUBLE(0)
    outputWeights = []
    
    def __init__(self, name, numberOfOutputs):
        self.name = name
        self.idx = self.randomWeight()
        i = 0
        while i < int(numberOfOutputs):
            self.connections.append(Connection("{0}.cnxn{1}".format(self.name, i)))
            self.outputWeights.append(OutputWeights("{0}.ow{1}".format(self.name, i), self.idx))
            i = i + 1
        #print(self.connection.name)
        #self.outputWeights.setName(self.__name__)
        
    def getOutputValue(self):
        return self.outputValue
    
    def setOutputValue(self, desiredValue):
        self.outputValue = desiredValue

    def randomWeight(self):
        return np.random.uniform(0.0, 1.0)

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
        self.gradient = delta * self.tranferFunctionDerivative(self.outputValue)
    
    def calcHiddenGradients(self, nextLayer):
        self.dow = self.sumDOW(nextLayer)
        self.gradient = self.dow * self.transferFunctionDerivative(self.outputValue)
        
    def printAttributes(self):
        cnxns = ""
        for item in self.connections:
            cnxns = "{0}\n\t\tName: {1}\n\t\tWeight: {2}\n\t\tdeltaWeight: {3}\n".format(cnxns, item.name, item.weight, item.deltaWeight)
        i = 0
        for weight in self.outputWeights:
            ows = "Name: {0}\n\t\tWeights: {1}".format(weight.name, weight.outputWeights[i])
            i = i + 1
        print("Neuron: {0}\n\tConnections:\n\t\t{1}\n\toutputWeights:\n\t\t{2}\n\teta: {3}\n\tgradient: {4}\n\tidx: {5}\n".format(self.name, cnxns, ows, self.eta, self.gradient, self.idx))
    
    def main(self, numOutputs, desiredIndex):
        for output in numOutputs:
            output.push_back(Connection())
            output.back().weight = self.randomWeight()
        self.idx = desiredIndex

Layer = np.vectorize(Neuron)

class NeuralNetwork():
    __name__ = ""
    __trainingSet__ = []
    __topology__ = []
    __numInputs__ = 0
    __inputLayer__ = []
    __numHidden__ = 0
    __hiddenLayer__ = []
    __numOutputs__ = 0
    __outputLayer__ = []
    __numberOfLayers__ = 0
    __layers__ = []
    error = DOUBLE(0)
    recentAverageError = DOUBLE(0)
    recentAverageSmoothingFactor = DOUBLE(0)
    
    def __init__(self, name, topology, trainingSet):
        self.__name__ = name
        self.__topology__ = topology
        self.__trainingSet__ = trainingSet
        self.__numberOfLayers__ = len(self.__topology__)
        i = 0
        for item in self.__topology__:
            nrons = 0
            if i == 0:
                self.__numInputs__ = item
                while nrons < int(self.__numInputs__):
                    n = Neuron("Input{0}".format(nrons + 1), self.__numHidden__)
                    self.__inputLayer__.append(n)
                    nrons = nrons + 1
                self.__layers__.append(self.__inputLayer__)
            if i == 1:
                self.__numHidden__ = item
                while nrons < int(self.__numHidden__):
                    n = Neuron("Hidden{0}".format(nrons + 1), self.__numOutputs__)
                    self.__hiddenLayer__.append(n)
                    nrons = nrons + 1
                self.__layers__.append(self.__hiddenLayer__)
            if i == 2:
                self.__numOutputs__ = item
                while nrons < int(self.__numOutputs__):
                    n = Neuron("Output{0}".format(nrons + 1), self.__numHidden__)
                    self.__outputLayer__.append(n)
                    nrons = nrons + 1
                self.__layers__.append(self.__outputLayer__)
            i = i + 1
        x = 0
        while x < self.__layers__.__len__():
            for nron in self.__layers__[x]:
                nron.printAttributes()
            x = x + 1
        #lyr = 1
        #for nron in self.__topology__:
        #    i = 0
        #    while i < nron:
        #        n = Neuron("Layer{0}.Neuron{1}".format(lyr, i + 1))
        #        self.__layers__.append(n)
        #        i = i + 1
        #    lyr = lyr + 1
        #    print(nron)
        #for nron in self.__layers__:
        #    print(nron.name)
        #for item in self.topology:
        #    self.layers.push_back(Layer())
        #    numOutputs = len(self.topology[item])
        #j=0
        #for item in self.topology:
        #    self.layers.back().push_back(Neuron(numOutputs, j))
        #    j = j + 1
            
    def backPropogate(self, targetValues):
        outputLayer = self.__outputLayer__.back()
        for item in self.__outputLayer__:
            delta = targetValues[item] - self.__outputLayer__[item].getOutputValue()
            self.error = pow(delta, 2)
        self.error /= len(outputLayer) - 1
        self.error = np.sqrt(self.error)
        #RecentAverageError
        self.recentAverageError = (self.recentAverageError * self.recentAverageSmoothingFactor + self.error) / (self.recentAverageSmoothingFactor + 1.0)
        #Hidden Layer Gradient
        i = 0
        for item in self.layers:
            currentLayer = item
            if i > 0:
                prevLayer = currentLayer
                currentLayer = item
                for nron in currentLayer:
                    nron.updateInputWeights(prevLayer)
            i = i + 1
        
    def feedForward(self, inputValues):
        # Network input values
        elements = inputValues.split(",")
        input1 = int(elements[0])
        input2 = int(elements[1])
        outPut = int(elements[2])
        for nron in self.__inputLayer__:
            nron.feedForward(input1)
            nron.feedForward(input2)
        #self.layers.setOutputValue(inPut)
        # Network forward propogate
        i = 0
        for layer in self.layers:
            curlayer = layer
            if i > 0:
                prevlayer = curlayer
                curlayer = layer
                self.layers.feedForward(prevlayer)

    def getRecentAverageError(self):
        return self.recentAverageError
    
    def getResults(self, resultValues):
        resultValues = []
        for layer in self.layers.back().size():
            resultValues.push_back()[layer].getOutputValue()
            
    def getTopology(self):
        return self.__topology__
            

class XOR():
    __name__ = ""
    
    def __init(self, name):
        self.name = name
    
    def getBool(self):
        return np.random.randint(0,2)
    
    def getXORresult(self, a, b):
        if int(a) == int(b):
            retval = 0
        if int(a) != int(b):
            retval = 1
        return retval
        
class TrainingSet:
    __xor__ = XOR()
    __name__ = ""
    __setLength__ = 0
    __trainingSet__ = []
    
    def __init__(self, name, trainLength):
        self.__name__ = name
        self.__setLength__ = int(trainLength)
        n = 0
        while n < self.__setLength__:
            a = self.__xor__.getBool()
            b = self.__xor__.getBool()
            c = self.__xor__.getXORresult(a, b)
            #print("{0},{1},{2}".format(a,b,c))
            self.__trainingSet__.append("{0},{1},{2}".format(a,b,c))
            n = n + 1
    
    def getTrainingSet(self):
        return self.__trainingSet__
        
i = 0
topology = []
#N = []
while i < 3:
    if i == 0:
        lyrType = "input"
    if i == 1:
        lyrType = "hidden"
    if i == 2:
        lyrType = "output"
    nrons = input("Enter the number of neurons for {0} layer: ".format(lyrType))
    topology.append(nrons)
    x = 0
    i = i + 1

response2 = input("Please provide training set length: ")
tset = TrainingSet("tset{0}".format(response2), response2)
train = tset.getTrainingSet()
N = NeuralNetwork("N", topology, train)
for x in train:
    print("{0}".format(x))
    N.feedForward(x)


print("Topology: {0}".format(N.getTopology()))
print("Training Set: {0}".format(tset.__name__))

print("-30-30-30-")