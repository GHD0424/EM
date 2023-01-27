import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


class GMM():
    def __init__(self, n_components=1, n=1, eps=1e-5):
        self.n_components = n_components
        self.n = n
        self.mu = np.zeros((self.n_components, self.n))
        self.class_prior = np.zeros(self.n_components)
        self.covariance = np.zeros((self.n_components, self.n, self.n))
        self.eps = eps
        self.LogLikelihood = []
        self.iterations = 0

    def train(self, X):
        m = X.shape[0]
        self._paramInitialization(X, m)
        try:
            self._EM(X, m)
        except:
            print(
                "error in EM algorithm. Didn't get a successful model. Need to try again."
            )
            raise

    def classify(self, X):
        m = X.shape[0]
        pdfs = np.zeros((m, self.n_components))
        for k in range(self.n_components):
            pdfs[:, k] = self._gaussianPDF(X, self.mu[k], self.covariance[k])
        labels = np.argmax(pdfs, axis=1)
        return labels.reshape(-1, 1)

    def plot(self, X, labels=None):
        if labels is not None:
            plt.scatter(X[:, 0], X[:, 1], s=8, c=labels, alpha=0.5)
        else:
            plt.scatter(X[:, 0], X[:, 1], s=8, c="grey", alpha=0.5)
            #for k in range(self.n_components):
              #self._plotGaussianContour(self.mu[k], self.covariance[k], X)

    def plotAndSave(self, X, labels=None, savePath='./img/picture.png'):
        self.plot(X, labels)
        plt.savefig(savePath)
        plt.close('all')

    def plotLogLikelihood(self):
        length = len(self.LogLikelihood)
        X = np.linspace(0, length, length)
        Y = self.LogLikelihood
        plt.xlabel('iteration')
        plt.ylabel('LogLikelihood')
        plt.plot(X, Y, color='red')

    def _paramInitialization(self, X, m):
        estimator = KMeans(n_clusters=self.n_components)
        estimator.fit(X)
        centroids = estimator.cluster_centers_
        self.mu = centroids

        dist = np.tile(
            np.sum(X * X, axis=1).reshape(-1, 1),
            (1, self.n_components)) + np.tile(
                np.sum(self.mu * self.mu, axis=1).reshape(1, -1),
                (m, 1)) - 2 * np.dot(X, self.mu.T)
        labels = np.argmin(dist, axis=1)
        for k in range(self.n_components):
            cluster = X[labels == k, :]
            self.class_prior[k] = cluster.shape[0] / m
            self.covariance[k, :, :] = np.cov(cluster.T)
    def _EM(self, X, m):
        pdfs = np.zeros((m, self.n_components))
        gamma = np.zeros((m, self.n_components))
        self.iterations = totalIteration = 100
        preLogLikelihood = LogLikelihood = 0
        for nowIter in range(totalIteration):
            for k in range(self.n_components):
                try:
                    pdfs[:, k] = self.class_prior[k] * self._gaussianPDF(
                        X, self.mu[k], self.covariance[k])
                except:
                    print(
                        "covariance matrix {} is singular matrix.\n something wrong in producing probabilities. iteration {}"
                        .format(k + 1, nowIter))
                    raise
            gamma = pdfs / np.sum(pdfs, axis=1).reshape(-1, 1)
            class_prior = np.sum(gamma, axis=0) / np.sum(gamma)
            mu = np.zeros((self.n_components, self.n))
            covariance = np.zeros((self.n_components, self.n, self.n))
            for k in range(self.n_components):
                mu[k] = np.average(X, axis=0, weights=gamma[:, k])
                cov = np.zeros((self.n, self.n))
                for i in range(m):
                    tmp = (X[i] - mu[k]).reshape(-1, 1)
                    cov += gamma[i, k] * np.dot(tmp, tmp.T)
                covariance[k, :, :] = cov / np.sum(gamma[:, k])
            self.class_prior = class_prior
            self.mu = mu
            self.covariance = covariance

            savePath = "./img/iterations/iteration" + str(nowIter) + ".png"
            self.plotAndSave(X, savePath=savePath)
            try:
                LogLikelihood = self._computeLogLikelihood(X)
            except:
                print(
                    "something wrong in computing LogLikelihood. iteration {}".
                    format(nowIter + 1))
                raise

            self.LogLikelihood.append(LogLikelihood)
            if abs(preLogLikelihood - LogLikelihood) < self.eps:
                self.iterations = nowIter
                break
            preLogLikelihood = LogLikelihood
    def _computeLogLikelihood(self, X):
        m = X.shape[0]
        pdfs = np.zeros((m, self.n_components))
        for k in range(self.n_components):
            try:
                pdfs[:, k] = self.class_prior[k] * self._gaussianPDF(
                    X, self.mu[k], self.covariance[k])
            except:
                print("covariance matrix {} is singular matrix".format(k + 1))
                raise
        return np.sum(np.log(np.sum(pdfs, axis=1)))

    def _plotGaussianContour(self, mu, covariance, X):
        xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
        ymin, ymax = np.min(X[:, 1]), np.max(X[:, 0])
        M = 200
        X, Y = np.meshgrid(np.linspace(xmin - 1, xmax + 1, M),
                           np.linspace(ymin - 1, ymax + 1, M))
        d = np.dstack((X, Y))
        Z = multivariate_normal.pdf(d, mu, covariance)
        plt.contour(X, Y, Z, levels=7)

        plt.scatter(mu[0], mu[1], s=100, marker='x', c='red')

    def _gaussianPDF(self, X, mu, covariance):
        covdet = np.linalg.det(covariance + np.eye(self.n) * 0.01)
        covinv = np.linalg.inv(covariance + np.eye(self.n) * 0.01)
        xdiff = X - mu
        prob = 1.0 / np.power(2 * np.pi, self.n / 2) / np.sqrt(
            np.abs(covdet)) * np.exp(
                -1.0 / 2 * np.diag(np.dot(np.dot(xdiff, covinv), xdiff.T)))
        return prob

