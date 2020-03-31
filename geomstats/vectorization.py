"""
Decorator to handle vectorization.

This abstracts the backend type.
This assumes that functions are implemented to return vectorized outputs.
"""

import geomstats.backend as gs

POINT_TYPES_TO_NDIMS = {
    'scalar': 2,
    'vector': 2,
    'matrix': 3}


def squeeze_output_dim_0(initial_ndims, point_types):
    """Determine if the output needs to squeeze a singular dimension 0.

    The dimension 0 is squeezed iff all input parameters:
    - contain one sample,
    - have the corresponding dimension 0 squeezed,
    i.e. if all input parameters have ndim strictly less than the ndim
    corresponding to their vectorized shape.

    Parameters
    ----------
    initial_ndims : list
        Initial ndims of input parameters, as entered by the user.
    point_types : list
        Associated list of point_type of input parameters.

    Returns
    -------
    squeeze : bool
        Boolean deciding whether to squeeze dim 0 of the output.
    """
    for ndim, point_type in zip(initial_ndims, point_types):
        vect_ndim = POINT_TYPES_TO_NDIMS[point_type]
        assert ndim <= vect_ndim
        if ndim == vect_ndim:
            return False
    return True


def is_scalar(vect_array):
    """Test if an array represents a scalar."""
    has_ndim_2 = vect_array.ndim == 2
    has_singleton_dim_1 = vect_array.shape[1] == 1
    return has_ndim_2 and has_singleton_dim_1


def squeeze_output_dim_1(result, initial_shapes, point_types):
    """Determine if the output needs to squeeze a singular dimension 1.

    This happens if the user represents scalars as array of shapes:
    [n_samples,] instead of [n_samples, 1]

    Dimension 1 is squeezed by default if the return point type is a scalar.
    Dimension 1 is not squeezed if the user inputs at least one scalar with
    a singleton in dimension 1.

    Parameters
    ----------
    result: array-like
        Result output by the function, before reshaping.
    initial_shapes : list
        Initial shapes of input parameters, as entered by the user.
    point_types : list
        Associated list of point_type of input parameters.

    Returns
    -------
    squeeze : bool
        Boolean deciding whether to squeeze dim 1 of the output.
    """
    if not is_scalar(result):
        return False

    for shape, point_type in zip(initial_shapes, point_types):
        ndim = len(shape)
        if point_type == 'scalar':
            assert ndim <= 2
            if ndim == 2:
                return False
    return True


def decorator(param_names, point_types):
    """Vectorize geomstats functions.

    This decorator assumes that its function is coded using
    the following conventions:
    - all input parameters are fully-vectorized,
    - all outputs are fully-vectorized,

    where "fully-vectorized" means that:
    - one scalar has shape [1, 1],
    - n scalars have shape [n, 1],
    - on d-D vector has shape [1, d],
    - n d-D vectors have shape [n, d],
    etc.

    The decorator enables flexibility in the input shapes,
    and adapt the output shapes to match the users' expectations.

    Parameters
    ----------
    param_names : list
        Parameters names to be vectorized.
    point_types : list
        Associated list of their point_types.
    """
    def aux_decorator(function):
        def wrapper(*args, **kwargs):
            vect_args = []
            initial_shapes = []
            initial_ndims = []
            for arg, point_type in zip(args, point_types):
                if point_type == 'scalar':
                    arg = gs.array(arg)
                initial_shapes.append(arg.shape)
                initial_ndims.append(gs.ndim(arg))

                if point_type == 'scalar':
                    vect_arg = gs.to_ndarray(arg, to_ndim=1)
                    vect_arg = gs.to_ndarray(vect_arg, to_ndim=2, axis=1)
                else:
                    vect_arg = gs.to_ndarray(
                        arg, to_ndim=POINT_TYPES_TO_NDIMS[point_type])
                vect_args.append(vect_arg)
            result = function(*vect_args, **kwargs)

            if squeeze_output_dim_1(result, initial_shapes, point_types):
                result = gs.squeeze(result, axis=1)

            if squeeze_output_dim_0(initial_ndims, point_types):
                result = gs.squeeze(result, axis=0)
            return result
        return wrapper
    return aux_decorator
