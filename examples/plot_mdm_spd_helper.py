import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

fontsize = 15
matplotlib.rc('font', size=fontsize)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=fontsize)
matplotlib.rc('font',
              family='times',
              serif=['Computer Modern Roman'],
              monospace=['Computer Modern Typewriter'])

EPS = 1e-8


class DatasetSPD_2D():

    def __init__(self, n_samples, n_features, n_classes):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.data_helper = DataHelper()

    def generate_sample_dataset(self):
        X, Y = self.easy_gauss()
        X, Y = self.data_helper.shuffle(X, Y)
        return X, Y

    def mat2vec(self, mat):
        n = mat.shape[-1]
        return mat[np.tril_indices(n)]

    # returns one random rotation of dimension n
    def random_rotation(self, n):
        if(self.n_features == 2):
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            s = np.array([[c, -s], [s, c]])
            return s
        z = (np.random.randn(n, n) + 1j *
             np.random.randn(n, n)) / np.sqrt(2.0)
        q, r = np.linalg.qr(z)
        d = np.diag(r)
        ph = np.diag(d / np.abs(d))
        # q=np.dot(q,np.dot(ph,q))
        s = np.dot(q, ph)
        return s

    # returns N random rotation matrices close of var_rot around R
    def random_rotations(self, R, var_rot):
        rots = np.zeros((self.n_samples, self.n_features, self.n_features))
        for i in range(self.n_samples):
            rots[i] = np.dot(
                R, self.divide_angle_of_cov2(
                    self.random_rotation(self.n_features), var_rot))
        return rots

    def angle_of_rot2(self, r):
        return np.arctan(r[0][1] / r[0][0])

    def divide_angle_of_cov2(self, r, alpha):
        angle = self.angle_of_rot2(r) * alpha
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def easy_gauss(self):

        # hyperparams
        # get self.n_classes mean mean vectors
        M = np.random.uniform(-5, 5, (self.n_classes, self.n_features))
        # get self.n_classes mean diagonal covariances
        S = np.random.uniform(0.1, 5., (self.n_classes, self.n_features))
        if(self.n_features == 2):
            # get self.n_classes mean rotations
            R = self.random_rotations(np.eye(self.n_features), 1)
        var_mean = np.eye(self.n_features) * 0.05  # class variance in mean
        var_cov = np.eye(self.n_features) * 0.05  # in covariance
        if(self.n_features == 2):
            var_rot = 0.01  # in rotation

        # data
        mu = np.zeros((self.n_classes * self.n_samples, self.n_features))
        cov = np.zeros(
            (self.n_classes *
             self.n_samples,
             self.n_features,
             self.n_features))
        Y = np.zeros((self.n_classes * self.n_samples, self.n_classes))
        for i in range(self.n_classes):
            means = np.random.multivariate_normal(
                M[i], var_mean, self.n_samples)
            covs = np.random.multivariate_normal(
                S[i], var_cov, self.n_samples)
            rots = self.random_rotations(R[i], var_rot)
            mu[i * self.n_samples:(i + 1) * self.n_samples] = means
            for j in range(self.n_samples):
                c = np.diag(np.abs(covs[j]))
                c = np.dot(rots[j], np.dot(c, rots[j].T))
                cov[i * self.n_samples + j] = c
            Y[i * self.n_samples:(i + 1) * self.n_samples, i] = 1
        return cov, Y


class PlotHelper():

    def __init__(self):
        self.fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=1, figure=self.fig)
        self.fig.add_subplot(spec[0, 0])
        self.colors = ['r', 'g', 'b', 'xkcd:camel', 'k']
        self.colors_alt = ['xkcd:burgundy', 'olive', 'cyan', 'xkcd:mud brown']

    def plot_ellipse(self, data_point, **kwargs):
        X, Y, x, y = self.ellipse(data_point)
        self.fig.axes[0].plot(X, Y, **kwargs)
        return x, y

    def angle_of_rot2(self, r):
        return np.arctan(r[0][1] / r[0][0])

    def ellipse(self, P):
        w, vr = np.linalg.eig(P)
        w = w.real + EPS
        Np = 100

        [e1, e2] = w
        x0, y0 = 0, 0
        angle = self.angle_of_rot2(vr)
        c, s = np.cos(angle), np.sin(angle)
        the = np.linspace(0, 2 * np.pi, Np)
        X = e1 * np.cos(the) * c - s * e2 * np.sin(the) + x0
        Y = e1 * np.cos(the) * s + c * e2 * np.sin(the) + y0
        return X, Y, X[Np // 4], Y[Np // 4]

    def plot_arrow(self, x_from, y_from, x_to, y_to):
        fig_trans = self.fig.transFigure.inverted()
        coord_from = fig_trans.transform(
            self.fig.axes[0].transData.transform(
                (x_from, y_from)))
        coord_to = fig_trans.transform(
            self.fig.axes[0].transData.transform(
                (x_to, y_to)))
        # d_coord=coord_to-coord_from
        arrow = matplotlib.patches.FancyArrowPatch(
            coord_from,
            coord_to,
            transform=self.fig.transFigure,
            fc="k",
            connectionstyle="arc3,rad=0.",
            arrowstyle='fancy',
            alpha=1.,
            mutation_scale=5.)
        # arrow=matplotlib.patches.Arrow(coord_from[0],coord_from[1],d_coord[0],d_coord[1],
        # transform=self.fig.transFigure,fc="k",width=1.)
        self.fig.patches.append(arrow)

    def plot_final(self):
        plt.legend(loc='best')
        self.fig.axes[0].set_title(
            'Example plot of the MDM classifier in dimension 2\n'
            '3-class fit and 3 test sample prediction\n'
            '(black arrows denote assignement)')
        plt.show()


class DataHelper():
    """
    DataHelper provides simple functions to handle data.

    Data is assumed of the following shape:
    X: Data, shape=[n_samples, ...]
    Y: Labels, shape=[n_samples, n_classes] (one-hot encoding)
    """

    def shuffle(self, X, Y):
        tmp = list(zip(X, Y))
        np.random.shuffle(tmp)
        X, Y = zip(*tmp)
        X = np.asarray(X)
        Y = np.asarray(Y)
        return X, Y

    def get_label_at_index(self, i, labels):
        return np.where(labels[i])[0][0]

    def data_with_label(self, data, labels, c):
        return data[np.where(np.where(labels)[1] == c)]
