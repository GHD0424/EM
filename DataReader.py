import numpy as np

class DataReader():

    def __init__(self):
        pass

    def readDataX(self, filename):
        X = []
        with open(filename, "r", encoding="utf-8") as inputFile:
            for line in inputFile.readlines():
                line = line.strip('\n').split(' ')
                line = [eval(x) for x in line]
                X.append(line)
        X = np.array(X)
        return X

    def sampleSplit(self, data, ratio=0.2):
        np.random.shuffle(data)
        n = np.int(data.shape[0]*ratio)
        testData, trainData = data[:n], data[n:]
        return trainData, testData
