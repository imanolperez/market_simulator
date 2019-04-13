import doctest
from esig import tosig as esig
from esig import tests as tests
import numpy as np
from . import tjl_dense_numpy_tensor as te
from . import tjl_hall_numpy_lie as li

#@decorators.accepts(int, int)
def logsigdim(signal_dimension, signature_degree):
    """Returns the length of the log-signature vector.

    Parameters
    ----------
    signal_dimension : int
        Dimension of the underlying vector space.
    signature_degree : int
        Degree of the log-signature.

    Returns
    -------
    int
        Length of the log-signature vector.
        
    """

    

    hall_set, degrees, degree_boundaries, reverse_map, width = li.hall_basis(
        signal_dimension, signature_degree
    )
    return hall_set.shape[0] - 1


#@decorators.accepts(int, int)
def logsigkeys(signal_dimension, signature_degree):
    """Finds keys associated to the log-signature.

    Parameters
    ----------
    signal_dimension : int
        Dimension of the underlying vector space.
    signature_degree : int
        Degree of the log-signature.

    Returns
    -------
    str
        A space separated ascii string containing the keys associated
        to the entries in the log-signature.

    """

    return li.logsigkeys(signal_dimension, signature_degree)


#@decorators.accepts(int, int)
def sigdim(signal_dimension, signature_degree):
    """Returns the length of the signature vector.

    Parameters
    ----------
    signal_dimension : int
        Dimension of the underlying vector space.
    signature_degree : int
        Degree of the signature.

    Returns
    -------
    int
        Length of the signature vector.
        
    """
    return te.blob_size(signal_dimension, signature_degree) - te.blob_size(
        signal_dimension
    )


#@decorators.accepts(int, int)
def sigkeys(signal_dimension, signature_degree):
    """Finds keys associated to the signature.

    Parameters
    ----------
    signal_dimension : int
        Dimension of the underlying vector space.
    signature_degree : int
        Degree of the signature.

    Returns
    -------
    str
        A space separated ascii string containing the keys associated
        to the entries in the signature.

    """
    return li.sigkeys(signal_dimension, signature_degree)


#@decorators.accepts((list, np.ndarray), int)
def stream2logsig(stream, signature_degree):
    """Computes the log-signature of a stream.

    Parameters
    ----------
    array : array of shape (length, 2)
        Stream whose log-signature will be computed.
    signature_degree : int
        Log-signature degree.

    Returns
    -------
    array
        Log-signature of the stream.

    """
    return li.sparse_to_dense(
        li.t2l(
            te.tensor_log(
                te.stream2sigtensor(stream, signature_degree), signature_degree
            ),
        ),
        stream.shape[1],
        signature_degree,
    )


#@decorators.accepts((list, np.ndarray), int)
def stream2sig(stream, signature_degree):
    """Computes the signature of a stream.

    Parameters
    ----------
    array : array of shape (length, 2)
        Stream whose signature will be computed.
    signature_degree : int
        Signature degree.

    Returns
    -------
    array
        Signature of the stream.

    Examples
    ---------
    >>> import tosig as ts
    >>> from tjl_dense_numpy_tensor import brownian
    >>> from esig import tosig as ets
    >>> width = 4
    >>> depth = 4
    >>> ts.sigkeys(width, depth)
    ' () (1) (2) (3) (4) (1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (2,3) (2,4) (3,1) (3,2) (3,3) (3,4) (4,1) (4,2) (4,3) (4,4) (1,1,1) (1,1,2) (1,1,3) (1,1,4) (1,2,1) (1,2,2) (1,2,3) (1,2,4) (1,3,1) (1,3,2) (1,3,3) (1,3,4) (1,4,1) (1,4,2) (1,4,3) (1,4,4) (2,1,1) (2,1,2) (2,1,3) (2,1,4) (2,2,1) (2,2,2) (2,2,3) (2,2,4) (2,3,1) (2,3,2) (2,3,3) (2,3,4) (2,4,1) (2,4,2) (2,4,3) (2,4,4) (3,1,1) (3,1,2) (3,1,3) (3,1,4) (3,2,1) (3,2,2) (3,2,3) (3,2,4) (3,3,1) (3,3,2) (3,3,3) (3,3,4) (3,4,1) (3,4,2) (3,4,3) (3,4,4) (4,1,1) (4,1,2) (4,1,3) (4,1,4) (4,2,1) (4,2,2) (4,2,3) (4,2,4) (4,3,1) (4,3,2) (4,3,3) (4,3,4) (4,4,1) (4,4,2) (4,4,3) (4,4,4) (1,1,1,1) (1,1,1,2) (1,1,1,3) (1,1,1,4) (1,1,2,1) (1,1,2,2) (1,1,2,3) (1,1,2,4) (1,1,3,1) (1,1,3,2) (1,1,3,3) (1,1,3,4) (1,1,4,1) (1,1,4,2) (1,1,4,3) (1,1,4,4) (1,2,1,1) (1,2,1,2) (1,2,1,3) (1,2,1,4) (1,2,2,1) (1,2,2,2) (1,2,2,3) (1,2,2,4) (1,2,3,1) (1,2,3,2) (1,2,3,3) (1,2,3,4) (1,2,4,1) (1,2,4,2) (1,2,4,3) (1,2,4,4) (1,3,1,1) (1,3,1,2) (1,3,1,3) (1,3,1,4) (1,3,2,1) (1,3,2,2) (1,3,2,3) (1,3,2,4) (1,3,3,1) (1,3,3,2) (1,3,3,3) (1,3,3,4) (1,3,4,1) (1,3,4,2) (1,3,4,3) (1,3,4,4) (1,4,1,1) (1,4,1,2) (1,4,1,3) (1,4,1,4) (1,4,2,1) (1,4,2,2) (1,4,2,3) (1,4,2,4) (1,4,3,1) (1,4,3,2) (1,4,3,3) (1,4,3,4) (1,4,4,1) (1,4,4,2) (1,4,4,3) (1,4,4,4) (2,1,1,1) (2,1,1,2) (2,1,1,3) (2,1,1,4) (2,1,2,1) (2,1,2,2) (2,1,2,3) (2,1,2,4) (2,1,3,1) (2,1,3,2) (2,1,3,3) (2,1,3,4) (2,1,4,1) (2,1,4,2) (2,1,4,3) (2,1,4,4) (2,2,1,1) (2,2,1,2) (2,2,1,3) (2,2,1,4) (2,2,2,1) (2,2,2,2) (2,2,2,3) (2,2,2,4) (2,2,3,1) (2,2,3,2) (2,2,3,3) (2,2,3,4) (2,2,4,1) (2,2,4,2) (2,2,4,3) (2,2,4,4) (2,3,1,1) (2,3,1,2) (2,3,1,3) (2,3,1,4) (2,3,2,1) (2,3,2,2) (2,3,2,3) (2,3,2,4) (2,3,3,1) (2,3,3,2) (2,3,3,3) (2,3,3,4) (2,3,4,1) (2,3,4,2) (2,3,4,3) (2,3,4,4) (2,4,1,1) (2,4,1,2) (2,4,1,3) (2,4,1,4) (2,4,2,1) (2,4,2,2) (2,4,2,3) (2,4,2,4) (2,4,3,1) (2,4,3,2) (2,4,3,3) (2,4,3,4) (2,4,4,1) (2,4,4,2) (2,4,4,3) (2,4,4,4) (3,1,1,1) (3,1,1,2) (3,1,1,3) (3,1,1,4) (3,1,2,1) (3,1,2,2) (3,1,2,3) (3,1,2,4) (3,1,3,1) (3,1,3,2) (3,1,3,3) (3,1,3,4) (3,1,4,1) (3,1,4,2) (3,1,4,3) (3,1,4,4) (3,2,1,1) (3,2,1,2) (3,2,1,3) (3,2,1,4) (3,2,2,1) (3,2,2,2) (3,2,2,3) (3,2,2,4) (3,2,3,1) (3,2,3,2) (3,2,3,3) (3,2,3,4) (3,2,4,1) (3,2,4,2) (3,2,4,3) (3,2,4,4) (3,3,1,1) (3,3,1,2) (3,3,1,3) (3,3,1,4) (3,3,2,1) (3,3,2,2) (3,3,2,3) (3,3,2,4) (3,3,3,1) (3,3,3,2) (3,3,3,3) (3,3,3,4) (3,3,4,1) (3,3,4,2) (3,3,4,3) (3,3,4,4) (3,4,1,1) (3,4,1,2) (3,4,1,3) (3,4,1,4) (3,4,2,1) (3,4,2,2) (3,4,2,3) (3,4,2,4) (3,4,3,1) (3,4,3,2) (3,4,3,3) (3,4,3,4) (3,4,4,1) (3,4,4,2) (3,4,4,3) (3,4,4,4) (4,1,1,1) (4,1,1,2) (4,1,1,3) (4,1,1,4) (4,1,2,1) (4,1,2,2) (4,1,2,3) (4,1,2,4) (4,1,3,1) (4,1,3,2) (4,1,3,3) (4,1,3,4) (4,1,4,1) (4,1,4,2) (4,1,4,3) (4,1,4,4) (4,2,1,1) (4,2,1,2) (4,2,1,3) (4,2,1,4) (4,2,2,1) (4,2,2,2) (4,2,2,3) (4,2,2,4) (4,2,3,1) (4,2,3,2) (4,2,3,3) (4,2,3,4) (4,2,4,1) (4,2,4,2) (4,2,4,3) (4,2,4,4) (4,3,1,1) (4,3,1,2) (4,3,1,3) (4,3,1,4) (4,3,2,1) (4,3,2,2) (4,3,2,3) (4,3,2,4) (4,3,3,1) (4,3,3,2) (4,3,3,3) (4,3,3,4) (4,3,4,1) (4,3,4,2) (4,3,4,3) (4,3,4,4) (4,4,1,1) (4,4,1,2) (4,4,1,3) (4,4,1,4) (4,4,2,1) (4,4,2,2) (4,4,2,3) (4,4,2,4) (4,4,3,1) (4,4,3,2) (4,4,3,3) (4,4,3,4) (4,4,4,1) (4,4,4,2) (4,4,4,3) (4,4,4,4)'
    >>> ts.logsigkeys(width, depth)
    ' 1 2 3 4 [1,2] [1,3] [1,4] [2,3] [2,4] [3,4] [1,[1,2]] [1,[1,3]] [1,[1,4]] [2,[1,2]] [2,[1,3]] [2,[1,4]] [2,[2,3]] [2,[2,4]] [3,[1,2]] [3,[1,3]] [3,[1,4]] [3,[2,3]] [3,[2,4]] [3,[3,4]] [4,[1,2]] [4,[1,3]] [4,[1,4]] [4,[2,3]] [4,[2,4]] [4,[3,4]] [1,[1,[1,2]]] [1,[1,[1,3]]] [1,[1,[1,4]]] [2,[1,[1,2]]] [2,[1,[1,3]]] [2,[1,[1,4]]] [2,[2,[1,2]]] [2,[2,[1,3]]] [2,[2,[1,4]]] [2,[2,[2,3]]] [2,[2,[2,4]]] [3,[1,[1,2]]] [3,[1,[1,3]]] [3,[1,[1,4]]] [3,[2,[1,2]]] [3,[2,[1,3]]] [3,[2,[1,4]]] [3,[2,[2,3]]] [3,[2,[2,4]]] [3,[3,[1,2]]] [3,[3,[1,3]]] [3,[3,[1,4]]] [3,[3,[2,3]]] [3,[3,[2,4]]] [3,[3,[3,4]]] [4,[1,[1,2]]] [4,[1,[1,3]]] [4,[1,[1,4]]] [4,[2,[1,2]]] [4,[2,[1,3]]] [4,[2,[1,4]]] [4,[2,[2,3]]] [4,[2,[2,4]]] [4,[3,[1,2]]] [4,[3,[1,3]]] [4,[3,[1,4]]] [4,[3,[2,3]]] [4,[3,[2,4]]] [4,[3,[3,4]]] [4,[4,[1,2]]] [4,[4,[1,3]]] [4,[4,[1,4]]] [4,[4,[2,3]]] [4,[4,[2,4]]] [4,[4,[3,4]]] [[1,2],[1,3]] [[1,2],[1,4]] [[1,2],[2,3]] [[1,2],[2,4]] [[1,2],[3,4]] [[1,3],[1,4]] [[1,3],[2,3]] [[1,3],[2,4]] [[1,3],[3,4]] [[1,4],[2,3]] [[1,4],[2,4]] [[1,4],[3,4]] [[2,3],[2,4]] [[2,3],[3,4]] [[2,4],[3,4]]'
    >>> stream = brownian(100, width)
    >>> print(np.max(np.abs(ets.stream2sig(stream,depth)-ts.stream2sig(stream,depth))) < 1e-12)
    True
    >>> print(np.max(np.abs(ets.stream2logsig(stream,depth)-ts.stream2logsig(stream,depth))) < 1e-12)
    True

    """
    width = stream.shape[1]
    return te.stream2sigtensor(stream, signature_degree)[te.blob_size(width) :]


#@decorators.accepts((list, np.ndarray), int)
def tensor_exp(tensor, truncation_level):
    """Computes the truncated exponential of tensor.
    
    The exponential is given by
    
    exp(tensor) := 1 + tensor + tensor^2/2! + ... + tensor^n/n!,
    
    where n = truncation_level.
    
    Parameters
    ----------
    tensor : array_like
        Tensor whose exponential will be computed. The scalar term,
        i.e. tensor[0], must be non-zero.
    truncation_level : int
        Truncation order.

    Returns
    -------
    array
        Exponential of tensor.

    Examples
    --------
    >>> d = 7
    >>> s = te.stream2sigtensor(te.brownian(100,2),d)
    >>> t = te.tensor_log(s,d)
    >>> np.sum(te.tensor_sub(s, tensor_exp(te.tensor_log(s,d), d))[te.blob_size(2):]**2) < 1e-16
    True

    """

    return te.tensor_exp(tensor, truncation_level)

#@decorators.accepts((list, np.ndarray), int)
def tensor_log(tensor, truncation_level):
    """Computes the logarithm of a tensor.

    Parameters
    ----------
    tensor : array_like
        Tensor whose logarithm will be computed. The scalar term,
        i.e. tensor[0], must be non-zero.
    truncation_level : int
        Truncation order.

    Returns
    -------
    array
        Logarithm of tensor.

    Examples
    --------
    >>> d = 7
    >>> s = te.stream2sigtensor(te.brownian(100,2),d)
    >>> t = tensor_log(s,d)
    >>> np.sum(te.tensor_sub(s, te.tensor_exp(tensor_log(s,d), d))[te.blob_size(2):]**2) < 1e-16
    True
    
    """

    return te.tensor_log(tensor, truncation_level)

#@decorators.accepts((list, np.ndarray), (list, np.ndarray), int)
def tensor_multiply(tensor1, tensor2, depth):
    """Multiplies two tensors.

    Parameters
    ----------
    tensor1 : array_like
        First argument.
    tensor2 : array_like
        Second argument.
    depth : int
        Dimension of the underlying vector space.

    Returns
    -------
    array
        Tensor product of tensor1 and tensor2.

    Examples
    --------
    >>> tensor_multiply(te.arange(3,2),te.arange(3,2),2).tolist()
    [3.0, 1.0, 4.0, 6.0, 8.0, 14.0, 18.0, 22.0, 22.0, 27.0, 32.0, 30.0, 36.0, 42.0]
    >>> tensor_multiply(te.arange(3,2),te.ones(3,2),2).tolist()
    [3.0, 1.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0]
    >>> tensor_multiply(te.arange(3,2),te.one(3,2),2).tolist()
    [3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    
    """

    return te.tensor_multiply(tensor1, tensor2, depth)

#@decorators.accepts(dict, int, int)
def lie2tensor(lie_element, width, depth):
    """Projects a Lie element in Hall basis form in tensor form.

    Parameters
    ----------
    lie_element : dict
        Lie element that will be changed of basis.
    width : int
        Dimension of the underlying vector space.
    depth : int
        Order of the truncated tensor algebra.

    Returns
    -------
    array
        The Lie element transformed into a tensor.

    Examples
    --------
    >>> width = 2
    >>> depth = 3
    >>> t = te.tensor_log(te.stream2sigtensor(te.brownian(100, width), depth), depth)
    >>> np.sum(te.tensor_sub(lie2tensor(li.t2l(t), width, depth), t)[2:]**2)  < 1e-16
    True
    
    """
    return li.l2t(lie_element, width, depth)

#@decorators.accepts((list, np.ndarray))
def tensor2lie(tensor):
    """Projects a Lie element in tensor form to a Lie in Hall basis form.

    Parameters
    ----------
    tensor : array_like
        Tensor.
    
    Returns
    -------
    array
        The tensor in terms of the Hall basis.

    Examples
    --------
    >>> width = 2
    >>> depth = 3
    >>> t = te.tensor_log(te.stream2sigtensor(te.brownian(100, width), depth), depth)
    >>> np.sum(te.tensor_sub(li.l2t(tensor2lie(t),width,depth), t)[2:]**2)  < 1e-16
    True
    
    """
    return li.t2l(tensor)


def logsig2sig(logsig, width, depth):
    L = {i + 1: l for i, l in enumerate(logsig)}

    tensor = lie2tensor(L, width, depth)[1:]

    return tensor_exp(np.r_[width, tensor], depth)[1:]