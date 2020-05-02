"""
Helper data classes for the MDM illustration example on SPD matrices
"""

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.spd_matrices import EigenSummary
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class DatasetSPD_2D():

    def __init__(self, n_samples=100, n_features=2, n_classes=3):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.data_helper = DataHelper()

    def generate_sample_dataset(self):
        X, Y = self.setup_data()
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

    def setup_data(self):

        mean_covariance_eigenvalues = gs.random.uniform(0.1, 5., (self.n_classes, self.n_features))
        base_rotations = SpecialOrthogonal(n=self.n_features).random_gaussian(
            gs.eye(self.n_features), 1, n_samples=self.n_classes)
        var_eigenvalues = gs.random.uniform(.04, .06, (self.n_classes, self.n_features))
        var_rotations = gs.random.uniform(.5, .75, (self.n_classes))
        # var_rotations = gs.random.uniform(.009, .011, (self.n_classes))

        # data
        cov = gs.zeros(
            (self.n_classes *
             self.n_samples,
             self.n_features,
             self.n_features))
        Y = gs.zeros((self.n_classes * self.n_samples, self.n_classes))
        for i in range(self.n_classes):
            # cov[i * self.n_samples:(i+1) * self.n_samples] = self.make_data_noisy(
            #     base_rotations[i],gs.diag(mean_covariance_eigenvalues[i]),var_rotations[i],var_eigenvalues[i])
            cov[i * self.n_samples:(i+1) * self.n_samples] = self.make_data(
                base_rotations[i],gs.diag(mean_covariance_eigenvalues[i]),var_rotations[i])
            Y[i * self.n_samples:(i + 1) * self.n_samples, i] = 1
        return cov, Y

    def make_data(self, eigenspace, eigenvalues, var):
        spd=SPDMatrices(n=self.n_features)
        spd.set_eigensummary(eigenspace,eigenvalues)
        spd_data = spd.random_gaussian(
            var_rotations=var, n_samples=self.n_samples)
        return spd_data

    def make_data_noisy(self,eigenspace, eigenvalues, var, var_eigenvalues):
        spd=SPDMatrices(n=self.n_features)
        spd.set_eigensummary(eigenspace,eigenvalues)
        spd_data = spd.random_gaussian_noisy(
            var_rotations=var, noise=var_eigenvalues, n_samples=self.n_samples)
        return spd_data


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
