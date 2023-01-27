import DataReader, GMM
import matplotlib.pyplot as plt
from matplotlib import animation


if __name__ == "__main__":
    file_name = input('Please input dataset name(A, B, C or Z):') + '.txt'
    dataReader = DataReader.DataReader()
    data = dataReader.readDataX(file_name)
    #print(data)
    if file_name == 'Z.txt':
       model = GMM.GMM(n_components=4, n=8)
       model.train(data)

       model.plotLogLikelihood()
       plt.savefig("./img/LogLikelihood.png")
       plt.close('all')

       labels = model.classify(data)
       model.plotAndSave(data, labels, "./img/DataCluster.png")
    else: 
       model = GMM.GMM(n_components=4, n=2)
       model.train(data)

       model.plotLogLikelihood()
       plt.savefig("./img/LogLikelihood.png")
       plt.close('all')

       labels = model.classify(data)
       model.plotAndSave(data, labels, "./img/DataCluster.png")
       


