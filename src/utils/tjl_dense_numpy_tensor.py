import doctest
import bisect
import math
import numpy as np


## tensors of different degrees are equal it all terms in the higher degree tensor not present in the lower degree tensor are zero

## blob_size(x,N) gives the full size of the footprint of a tensor algebra element with x letters and developed to N layers
## level -1 is used for the empty or minimal zero tensor and points to just after the information that defines the shape of the data
## ie the beginning of the actual data for the tensor if any

def blob_size(width, max_degree_included_in_blob=-1):
    """ 
>>> [blob_size(x,y) for (x,y) in [(3,1),(3,0),(3,3),(2,6)]]
[5, 2, 41, 128]
>>> [blob_size(x) for x in [3,0]]
[1, 1]
>>> 
    """
    if max_degree_included_in_blob >= 0:
        if width == 0:
            return 2
        if width == 1:
            return max_degree_included_in_blob + 2
        return int(
            1 + int((-1 + width ** (1 + max_degree_included_in_blob)) / (-1 + width))
        )
    else:
        return int(1)


## gives the tuple that defines the shape of the tensor component at given degree
def tensor_shape(degree, width):
    """
>>> [tensor_shape(x,y) for (x,y) in [(3,5),(1,5),(0,5)]]
[(5, 5, 5), (5,), ()]
    """
    return tuple([width for i in range(degree)])


### the degree of the smallest tensor whose blob_size is at least blobsz
### layers(blobsz, width) := inf{k : blob_size(width, k) >= blobsz}
def layers(blobsz, width):
    """
>>> [layers(blob_size(x,y) + z,x) for x in [2] for y in [0,1,17] for z in [-1,0,1]]
[-1, 0, 1, 1, 1, 2, 17, 17, 18]
>>> [(z, layers(z,2), blob_size(2,z),layers(blob_size(2,z),2)) for z in range(9)]
[(0, -1, 2, 0), (1, -1, 4, 1), (2, 0, 8, 2), (3, 1, 16, 3), (4, 1, 32, 4), (5, 2, 64, 5), (6, 2, 128, 6), (7, 2, 256, 7), (8, 2, 512, 8)]
>>> 
    """
    return next((k for k in range(-1, blobsz) if blob_size(width, k) >= blobsz), None)


def blob_overflow(x, N):
    return layers(blob_size(x, N), N) != x


def blob_misssize(bs, N):
    return blob_size(layers(bs, N), N) != bs


def zero(width, depth=-1):
    """
>>> [zero(x,y) for x in range(3) for y in range(4)]
[array([0., 0.]), array([0., 0.]), array([0., 0.]), array([0., 0.]), array([1., 0.]), array([1., 0., 0.]), array([1., 0., 0., 0.]), array([1., 0., 0., 0., 0.]), array([2., 0.]), array([2., 0., 0., 0.]), array([2., 0., 0., 0., 0., 0., 0., 0.]), array([2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]
>>> 
    """
    ans = np.zeros(blob_size(width, depth), dtype=np.float64)
    ans[0:1] = [np.float64(width)]
    return ans


def one(width, depth=0):
    """
>>> [one(x,y) for x in range(3) for y in range(4)]
[array([0., 1.]), array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1., 1.]), array([1., 1., 0.]), array([1., 1., 0., 0.]), array([1., 1., 0., 0., 0.]), array([2., 1.]), array([2., 1., 0., 0.]), array([2., 1., 0., 0., 0., 0., 0., 0.]), array([2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]
>>> [one(x) for x in range(3) for y in range(4)]
[array([0., 1.]), array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1., 1.]), array([1., 1.]), array([1., 1.]), array([1., 1.]), array([2., 1.]), array([2., 1.]), array([2., 1.]), array([2., 1.])]
>>> 
    """
    ans = np.zeros(blob_size(width, depth), dtype=np.float64)
    ans[0:2] = np.array([np.float64(width), 1.0])
    return ans


## all ones not useful except for testing
def ones(width, depth=0):
    """
>>> [ones(x,y) for x in range(3) for y in range(4)]
[array([0., 1.]), array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1., 1.]), array([1., 1., 1.]), array([1., 1., 1., 1.]), array([1., 1., 1., 1., 1.]), array([2., 1.]), array([2., 1., 1., 1.]), array([2., 1., 1., 1., 1., 1., 1., 1.]), array([2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])]
>>> 
    """
    ans = np.ones(blob_size(width, depth), dtype=np.float64)
    ans[0:2] = np.array([np.float64(width), 1.0])
    return ans


## count entries not useful except for testing
def arange(width, depth=0):
    """
>>> [arange(x,y) for x in range(3) for y in range(4)]
[array([0., 1.]), array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1., 1.]), array([1., 1., 2.]), array([1., 1., 2., 3.]), array([1., 1., 2., 3., 4.]), array([2., 1.]), array([2., 1., 2., 3.]), array([2., 1., 2., 3., 4., 5., 6., 7.]), array([ 2.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15.])]
>>> [arange(x) for x in range(3) for y in range(4)]
[array([0., 1.]), array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1., 1.]), array([1., 1.]), array([1., 1.]), array([1., 1.]), array([2., 1.]), array([2., 1.]), array([2., 1.]), array([2., 1.])]
>>> 
    """
    ans = np.arange(blob_size(width, depth), dtype=np.float64)
    ans[0:2] = np.array([np.float64(width), 1.0])
    return ans


def tensor_add(lhs, rhs):
    """
>>> tensor_add(arange(3,2),arange(3,2))
array([ 3.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.,
       26.])
>>> tensor_add(arange(3,2),arange(4,2))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in tensor_add
ValueError: ('different width tensors cannot be added:', 3.0, '!=', 4.0)
>>> 
    """
    if int(rhs[0:1]) != int(lhs[0:1]):
        raise ValueError(
            "different width tensors cannot be added:", lhs[0], "!=", rhs[0]
        )
    if lhs.size >= rhs.size:
        ans = np.array(lhs)
        ans[1 : rhs.size] += rhs[1:]
    else:
        ans = np.array(rhs)
        ans[1 : lhs.size] += lhs[1:]
    return ans


def rescale(arg, factor, top=None):
    """
>>> rescale(arange(3,2), .5)
array([3. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ,
       6.5])
>>> 
    """
    if top is None:
        top = arg[: blob_size(int(arg[0]))]
    xx = np.tensordot(factor, arg, axes=0)
    xx[0 : top.size] = top[:]
    return xx


def tensor_sub(lhs, rhs):
    """
>>> tensor_sub(arange(3,2),ones(3,2))
array([ 3.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.,
       12.])
>>> 
    """
    return tensor_add(lhs, rescale(rhs, -1.0))


def tensor_multiply(lhs, rhs, depth):
    """
>>> print(tensor_multiply(arange(3,2),arange(3,2),2))
[ 3.  1.  4.  6.  8. 14. 18. 22. 22. 27. 32. 30. 36. 42.]
>>> print(tensor_multiply(arange(3,2),ones(3,2),2))
[ 3.  1.  3.  4.  5.  8.  9. 10. 12. 13. 14. 16. 17. 18.]
>>> print(tensor_multiply(arange(3,2),one(3,2),2))
[ 3.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13.]
>>> 
    """
    # lhs and rhs same width
    if int(rhs[0:1]) != int(lhs[0:1]):
        raise ValueError(
            "different width tensors cannot be combined:", lhs[0], "!=", rhs[0]
        )
    # extract width
    width = int(lhs[0])
    lhs_layers = layers(lhs.size, width)
    rhs_layers = layers(rhs.size, width)
    out_depth = min(depth, lhs_layers + rhs_layers)
    ans = zero(int(lhs[0]), depth)
    for i in range(
        min(out_depth, lhs_layers + rhs_layers) + 1
    ):  ## i is the total degree ## j is the degree of the rhs term
        for j in range(max(i - lhs_layers, 0), min(i, rhs_layers) + 1):
            ## nxt row the tensors must be shaped before multiplicaton and flattened before assignment
            ansb = blob_size(width, i - 1)
            anse = blob_size(width, i)
            lhsb = blob_size(width, (i - j) - 1)
            lhse = blob_size(width, (i - j))
            rhsb = blob_size(width, j - 1)
            rhse = blob_size(width, j)
            ans[ansb:anse] += np.tensordot(
                np.reshape(lhs[lhsb:lhse], tensor_shape(i - j, width)),
                np.reshape(rhs[rhsb:rhse], tensor_shape(j, width)),
                axes=0,
            ).flatten()
    return ans


def tensor_log(arg, depth, normalised=True):
    """
>>> d = 7
>>> s = stream2sigtensor(brownian(100,2),d)
>>> t = tensor_log(s,d)
>>> np.sum(tensor_sub(s, tensor_exp(tensor_log(s,d), d))[blob_size(2):]**2) < 1e-25
True
>>> 
    """

    """" 
    Computes the truncated log of arg up to degree depth.
    The coef. of the constant term (empty word in the monoid) of arg 
    is forced to 1.
    log(arg) = log(1+x) = x - x^2/2 + ... + (-1)^(n+1) x^n/n.
    arg must have a nonzero scalar term and depth must be > 0
        """
    width = int(arg[0])
    top = np.array(
        arg[0 : blob_size(width) + 1]
    )  # throw an error if there is no body to tensor as log zero not allowed
    if normalised:
        x = np.array(arg)
        x[blob_size(width)] = 0.0
        # x = (arg - 1)
        result = zero(width, -1)
        # result will grow as the computation grows
        for i in range(depth, 0, -1):
            top[blob_size(width)] = np.float64((2 * (i % 2) - 1)) / np.float64(i)
            result = tensor_add(result, top)
            result = tensor_multiply(result, x, depth)
        return result
    else:
        scalar = top[blob_size(width)]
        x = rescale(arg, np.float64(1.0 / scalar))
        result = tensor_log(x, depth, True)
        ans[blob_size(width)] += math.log(scalar)
        return result


def tensor_exp(arg, depth):
    """"
>>> d = 7
>>> s = stream2sigtensor(brownian(100,2),d)
>>> t = tensor_log(s,d)
>>> np.sum(tensor_sub(s, tensor_exp(tensor_log(s,d), d))[blob_size(2):]**2) < 1e-25
True
>>> 
>>> # Computes the truncated exponential of arg
>>> #     1 + arg + arg^2/2! + ... + arg^n/n! where n = depth
    """
    width = int(arg[0])
    result = np.array(one(width))
    if arg.size > blob_size(width):
        top = np.array(arg[0 : blob_size(width) + 1])
        scalar = top[-1]
        top[-1] = 0.0
        x = np.array(arg)
        x[blob_size(width)] = 0.0
        for i in range(depth, 0, -1):
            xx = rescale(
                arg, 1.0 / np.float64(i), top
            )  # top resets the shape and here is extended to set the scalar coefficient to zero
            result = tensor_multiply(result, xx, depth)
            result[blob_size(width)] += 1.0
        result = np.tensordot(math.exp(scalar), result, axes=0)
        result[: blob_size(width)] = top[: blob_size(width)]
    return result


## a vector to input for testing
## a vector to input for testing
def white(steps=int(100), width=int(1), time=1.0):
    """
>>> np.sum((np.sum(white(10000,3,2.)**2, axis=0) - 2)**2) < 0.01
True
>>> 
    """
    mu, sigma = 0, math.sqrt(time / steps)  # mean and standard deviation
    return np.random.normal(mu, sigma, (steps, width))


def brownian(steps=int(100), width=int(1), time=1.0):
    """
>>> brownian()[0]
array([0.])
>>> brownian(50,4).shape
(51, 4)
>>> 
    """
    path = np.zeros((steps + 1, width), dtype=np.float64)
    np.cumsum(white(steps, width, time), axis=0, out=path[1:, :])
    return path


def _stream2sigtensor(increments, depth):
    length, width = increments.shape
    if length > 1:
        lh = int(length / 2)
        return tensor_multiply(
            _stream2sigtensor(increments[:lh, :], depth),
            _stream2sigtensor(increments[lh:, :], depth),
            depth,
        )
    else:
        lie = zero(width, 1)
        lie[blob_size(width, 0) : blob_size(width, 1)] = increments[0, :]
        return tensor_exp(lie, depth)


def stream2sigtensor(stream, depth):
    """
>>> s=np.array([[0.],[1.]])
>>> (sum(stream2sigtensor(s,7)[1:]) - math.e)**2 < 1e-6
True
>>> s = brownian(100,2)
>>> t  = np.flip(s,0)
>>> d = 7
>>> np.sum(tensor_sub(tensor_multiply(stream2sigtensor(s,d),stream2sigtensor(t,d), d), one(2,d))[blob_size(2):]**2) < 1e-25
True
>>> 
    """
    increments = stream[1:, :] - stream[:-1, :]
    return _stream2sigtensor(increments, depth)


def stream2sig(stream, depth):
    """
>>> d=7
>>> s = brownian(100,2)
>>> stream2sig(s,d).shape[0] < stream2sigtensor(s,d).shape[0] 
True
>>> 
>>> from esig import tosig as ts
>>> s=brownian(200,2)
>>> np.sum((stream2sig(s,5) - ts.stream2sig(s,5))**2) < 1e-25
True
>>> 
    """
    return stream2sigtensor(stream, depth)[1:]


if __name__ == "__main__":
    ##np.set_printoptions(suppress=True,formatter={'float_kind':'{:16.3f}'.format}, linewidth=130)
    # string with the expression # np.array_repr(x, precision=6, suppress_small=True)
    doctest.testmod()
