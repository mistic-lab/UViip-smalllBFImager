import numpy as np
import filterFuncs as ff


def nan_helper(y):
    """
    Returns array of indices where NaN's exist as well as a function with
    signature indices= index(logical_indices), to convert logical indices of
    NaNs to 'equivalent' indices
    [https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array]

    Parameters
    ----------
    y : numpy array
        Array with NaNs
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def zero_to_nan(y):
    """
    Turns all complex 0's into NaNs. 0+0j -> np.NaN

    Parameters
    ----------
    y : numpy array of complex numbers
        Array with 0+0j values
    """

    y[y == 0+0j] = np.NaN
    return y


def lin_interp_zeros(y):
    """
    Takes array with 0+0j elements, turns those elements into NaNs, linearly
    interpolates over them. Returns array of same size.

    Parameters
    ----------
    y : numpy array of complex numbers
        Array with 0+0j values
    """

    y = zero_to_nan(y)
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def interleave_with_zeros(y):
    """
    Takes array with N elements, doubles the size of the array by inserting a
    zero between each original element. Returns new interleaved array.

    Parameters
    ----------
    y : numpy array
    """

    # zer = np.zeros((len(y), 1), dtype=y.dtype)
    zer = np.zeros((y.shape), dtype=y.dtype)
    # out = np.empty((y.size + zer.size, 1), dtype=y.dtype)
    out = np.empty((y.size + zer.size, y.shape[1]), dtype=y.dtype)
    out[0::2] = y
    out[1::2] = zer
    return out


def remove_bad_data_interpolate_twice(someDict):
    """
    For each someDict item, first linearly interpolates over 0+0j values, then
    interleaves the array with 0+0j and linearly interpolates over those as
    well. Returns dict of same size where each element is twice as long.

    Parameters
    ----------
    someDict : dictionary of numpy arrays
    """

    for key in someDict:
        someDict[key] = ff.lin_interp_zeros(someDict[key])
        someDict[key] = ff.spread_with_zeros(someDict[key])
        someDict[key] = ff.lin_interp_zeros(someDict[key])
    return someDict


def conjugate_array(someDict, key):
    newKey = 'ant'+key[-1]+key[-2]
    someDict[newKey] = np.conjugate(someDict[key])
    return someDict
