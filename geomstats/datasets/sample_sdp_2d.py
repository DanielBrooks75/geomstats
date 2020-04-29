"""
Helper data classes for the MDM illustration example on SPD matrices
"""

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class DatasetSPD_2D():

    def __init__(self, n_samples=100, n_features=2, n_classes=3):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.data_helper = DataHelper()

    def generate_sample_dataset(self):
        X, Y = self.make_data()
        X, Y = self.data_helper.shuffle(X, Y)
        return X, Y

    # returns N random rotation matrices close of var_rot around R
    def random_rotations(self, R, var_rot):
        rots = gs.zeros((self.n_samples, self.n_features, self.n_features))
        for i in range(self.n_samples):
            # rot = self.random_rotation(self.n_features)
            rot = SpecialOrthogonal(self.n_features).random_uniform()
            rots[i] = gs.matmul(
                R, self.divide_angle_of_cov2(rot, var_rot))
        return rots

    def angle_of_rot2(self, r):
        return gs.arctan(r[0][1] / r[0][0])

    def divide_angle_of_cov2(self, r, alpha):
        angle = self.angle_of_rot2(r) * alpha
        c, s = gs.cos(angle), gs.sin(angle)
        return gs.array([[c, -s], [s, c]])

    def make_data(self):

        spd=SPDMatrices(n=self.n_features)
        so=SpecialOrthogonal(n=self.n_features)

        # hyperparams
        # get self.n_classes mean mean vectors
        M = gs.random.uniform(-5, 5, (self.n_classes, self.n_features))
        # get self.n_classes mean diagonal covariances
        S = gs.random.uniform(0.1, 5., (self.n_classes, self.n_features))
        if(self.n_features == 2):
            # get self.n_classes mean rotations
            # R = self.random_rotations(gs.eye(self.n_features), 1)
            R = so.random_gaussian(gs.eye(self.n_features), 1, n_samples=self.n_samples)
        var_mean = gs.eye(self.n_features) * 0.05  # class variance in mean
        var_cov = gs.eye(self.n_features) * 0.05  # in covariance
        if(self.n_features == 2):
            var_rot = 0.01  # in rotation

        # data
        mu = gs.zeros((self.n_classes * self.n_samples, self.n_features))
        cov = gs.zeros(
            (self.n_classes *
             self.n_samples,
             self.n_features,
             self.n_features))
        Y = gs.zeros((self.n_classes * self.n_samples, self.n_classes))
        for i in range(self.n_classes):
            means = gs.random.multivariate_normal(
                M[i], var_mean, self.n_samples)
            covs = gs.random.multivariate_normal(
                S[i], var_cov, self.n_samples)
            # rots = self.random_rotations(R[i], var_rot)
            rots = so.random_gaussian(R[i], var_rot, n_samples=self.n_samples)
            mu[i * self.n_samples:(i + 1) * self.n_samples] = means
            for j in range(self.n_samples):
                c = gs.diag(gs.abs(covs[j]))
                c = gs.dot(rots[j], gs.dot(c, rots[j].T))
                cov[i * self.n_samples + j] = c
            Y[i * self.n_samples:(i + 1) * self.n_samples, i] = 1
        return cov, Y


class DataHelper():
    """
    DataHelper provides simple functions to handle data.

    Data is assumed of the following shape:
    X: Data, shape=[n_samples, ...]
    Y: Labels, shape=[n_samples, n_classes] (one-hot encoding)
    """

    def shuffle(self, X, Y):
        tmp = list(zip(X, Y))
        gs.random.shuffle(tmp)
        X, Y = zip(*tmp)
        X = gs.array(X)
        Y = gs.array(Y)
        return X, Y

    def get_label_at_index(self, i, labels):
        return gs.where(labels[i])[0][0]

    def data_with_label(self, data, labels, c):
        return data[gs.where(gs.where(labels)[1] == c)]
