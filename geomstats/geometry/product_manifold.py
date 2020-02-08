"""Product of manifolds."""

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold

# TODO(nina): get rid of for loops
# TODO(nina): unit tests


class ProductManifold(Manifold):
    """Class for a product of manifolds M_1 x ... x M_n.

    In contrast to the class Landmarks or DiscretizedCruves,
    the manifolds M_1, ..., M_n need not be the same, nor of
    same dimension, but the list of manifolds needs to be provided.
    """

    def __init__(self, manifolds, default_point_type='vector'):
        """Construct the ProductManifold object.

        By default, a point is represented by an array of shape:
            [n_samples, dim_1 + ... + dim_n_manifolds]
        where n_manifolds is the number of manifolds in the product.
        This type of representation is called 'vector'.

        Alternatively, a point can be represented by an array of shape:
            [n_samples, n_manifolds, dim]
        if the n_manifolds have same dimension dim.
        This type of representation is called `matrix`.

        Parameters
        ----------
        manifolds : list
            List of manifolds in the product.
        default_point_type : str, {'vector', 'matrix'}
            Default representation of points.
        """
        assert default_point_type in ['vector', 'matrix']
        self.default_point_type = default_point_type

        self.manifolds = manifolds
        self.n_manifolds = len(manifolds)

        dimensions = [manifold.dimension for manifold in manifolds]
        super(ProductManifold, self).__init__(
            dimension=sum(dimensions))

    def belongs(self, point, point_type=None):
        """Test if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
                           or shape=[n_samples, dim_2, dim_2]
            Point.
        point_type : str, {'vector', 'matrix'}
            Representation of point.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            Array of booleans evaluating if the corresponding points
            belong to the manifold.
        """
        if point_type is None:
            point_type = self.default_point_type
        assert point_type in ['vector', 'matrix']

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
        else:
            point = gs.to_ndarray(point, to_ndim=3)

        n_manifolds = self.n_manifolds
        belongs = gs.empty((point.shape[0], n_manifolds))
        cum_dim = 0
        # FIXME: this only works if the points are in intrinsic representation
        for i in range(n_manifolds):
            manifold_i = self.manifolds[i]
            cum_dim_next = cum_dim + manifold_i.dimension
            point_i = point[:, cum_dim:cum_dim_next]
            belongs_i = manifold_i.belongs(point_i)
            belongs[:, i] = belongs_i
            cum_dim = cum_dim_next

        belongs = gs.all(belongs, axis=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2)
        return belongs

    def regularize(self, point, point_type=None):
        """Regularize the point into the manifold's canonical representation.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
                           or shape=[n_samples, dim_2, dim_2]
            Point.
        point_type : str, {'vector', 'matrix'}
            Representation of point.

        Returns
        -------
        regularized_point : array-like, shape=[n_samples, dim]
                           or shape=[n_samples, dim_2, dim_2]
            Point in the manifold's canonical representation.
        """
        # TODO(nina): Vectorize.
        if point_type is None:
            point_type = self.default_point_type
        assert point_type in ['vector', 'matrix']

        regularized_point = [self.manifold[i].regularize(point[i])
                             for i in range(self.n_manifolds)]

        # TODO(nina): Put this in a decorator
        if point_type == 'vector':
            regularized_point = gs.hstack(regularized_point)
        elif point_type == 'matrix':
            regularized_point = gs.vstack(regularized_point)
        return gs.all(regularized_point)

        return regularized_point

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None,
                 point_type=None):
        """Compute the geodesic as a function of t.

        This geodesic is seen as the product of the geodesic on each space.

        Parameters
        ----------
        initial_point : array-like, shape=[n_samples, dim]
            Initial point of the geodesic.
        end_point : array-like, shape=[n_samples, dim], optional
            End point of the geodesic.
        initial_tangent_vec : array-like, shape=[n_samples, dim], optional
            Initial tangent vector of the geodesic.
        point_type : str, {'vector', 'matrix'}, optional
            Representation of point.

        Returns
        -------
        geodesics : list
            List of callables
        """
        if point_type is None:
            point_type = self.default_point_type
        assert point_type in ['vector', 'matrix']

        geodesics = gs.asarray([[self.manifold[i].metric.geodesic(
            initial_point,
            end_point=end_point,
            initial_tangent_vec=initial_tangent_vec,
            point_type=point_type)
            for i in range(self.n_manifolds)]])
        return geodesics
