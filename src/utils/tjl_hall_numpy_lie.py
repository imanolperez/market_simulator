"""
Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurko and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)
"""
import doctest
from .tjl_dense_numpy_tensor import rescale
from .tjl_dense_numpy_tensor import tensor_add
import numpy as np
from collections import defaultdict
from . import tjl_dense_numpy_tensor



from .tjl_dense_numpy_tensor import blob_size

scalar_type = float


def hall_basis(width, desired_degree=0):
    """
hall_basis(1, 0)
(array([[0, 0]]), array([0]), array([1]), defaultdict(<class 'int'>, {}), 1)
hall_basis(1, 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".\tjl_hall_numpy_lie.py", line 41, in hall_basis
    if ((degrees[i] + degrees[j] == d) & (hall_set[j][0] <= i)):
IndexError: list index out of range
hall_basis(1, 1)
(array([[0, 0],
       [0, 1]]), array([0, 1]), array([1, 1]), defaultdict(<class 'int'>, {(0, 1): 1}), 1)
hall_basis(2,3)
(array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 2],
       [1, 3],
       [2, 3]]), array([0, 1, 1, 2, 3, 3]), array([1, 1, 3, 4]), defaultdict(<class 'int'>, {(0, 1): 1, (0, 2): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}), 2)

    """
    degrees = []
    hall_set = []
    degree_boundaries = []
    reverse_map = defaultdict(int)

    # the first entry in hall_set is not part of the basis but instead is
    # the nominal parent of all self parented elements (letters)
    # its values can be used for wider information about the lie element
    curr_degree = 0
    degrees.append(0)
    p = (0, 0)
    hall_set.append(p)
    degree_boundaries.append(1)
    if desired_degree > 0:
        # level 1 the first basis terms
        degree_boundaries.append(1)
        for i in range(1, width + 1):
            hall_set.append((0, i))
            degrees.append(1)
            reverse_map[(0, i)] = i
        curr_degree += 1
        for d in range(curr_degree + 1, desired_degree + 1):
            bound = len(hall_set)
            degree_boundaries.append(bound)
            for i in range(1, bound + 1):
                for j in range(i + 1, bound + 1):
                    if (degrees[i] + degrees[j] == d) & (hall_set[j][0] <= i):
                        hall_set.append((i, j))
                        degrees.append(d)
                        reverse_map[(i, j)] = len(hall_set) - 1
            curr_degree += 1
    return (
        np.array(hall_set, dtype=int),
        np.array(degrees, dtype=int),
        np.array(degree_boundaries, dtype=int),
        reverse_map,
        width,
    )


def hb_to_string(z, width, desired_degree):
    """
hb_to_string( 7 , 3, 6)
'[1,[1,2]]'

    """
    np_hall_set = hall_basis(width, desired_degree)[0]
    (n, m) = np_hall_set[z]
    if n:
        return (
            "["
            + hb_to_string(n, width, desired_degree)
            + ","
            + hb_to_string(m, width, desired_degree)
            + "]"
        )
    else:
        return str(m)


def logsigkeys(width, desired_degree):
    """
logsigkeys(3,6)
' 1 2 3 [1,2] [1,3] [2,3] [1,[1,2]] [1,[1,3]] [2,[1,2]] [2,[1,3]] [2,[2,3]] [3,[1,2]] [3,[1,3]] [3,[2,3]] [1,[1,[1,2]]] [1,[1,[1,3]]] [2,[1,[1,2]]] [2,[1,[1,3]]] [2,[2,[1,2]]] [2,[2,[1,3]]] [2,[2,[2,3]]] [3,[1,[1,2]]] [3,[1,[1,3]]] [3,[2,[1,2]]] [3,[2,[1,3]]] [3,[2,[2,3]]] [3,[3,[1,2]]] [3,[3,[1,3]]] [3,[3,[2,3]]] [[1,2],[1,3]] [[1,2],[2,3]] [[1,3],[2,3]] [1,[1,[1,[1,2]]]] [1,[1,[1,[1,3]]]] [2,[1,[1,[1,2]]]] [2,[1,[1,[1,3]]]] [2,[2,[1,[1,2]]]] [2,[2,[1,[1,3]]]] [2,[2,[2,[1,2]]]] [2,[2,[2,[1,3]]]] [2,[2,[2,[2,3]]]] [3,[1,[1,[1,2]]]] [3,[1,[1,[1,3]]]] [3,[2,[1,[1,2]]]] [3,[2,[1,[1,3]]]] [3,[2,[2,[1,2]]]] [3,[2,[2,[1,3]]]] [3,[2,[2,[2,3]]]] [3,[3,[1,[1,2]]]] [3,[3,[1,[1,3]]]] [3,[3,[2,[1,2]]]] [3,[3,[2,[1,3]]]] [3,[3,[2,[2,3]]]] [3,[3,[3,[1,2]]]] [3,[3,[3,[1,3]]]] [3,[3,[3,[2,3]]]] [[1,2],[1,[1,2]]] [[1,2],[1,[1,3]]] [[1,2],[2,[1,2]]] [[1,2],[2,[1,3]]] [[1,2],[2,[2,3]]] [[1,2],[3,[1,2]]] [[1,2],[3,[1,3]]] [[1,2],[3,[2,3]]] [[1,3],[1,[1,2]]] [[1,3],[1,[1,3]]] [[1,3],[2,[1,2]]] [[1,3],[2,[1,3]]] [[1,3],[2,[2,3]]] [[1,3],[3,[1,2]]] [[1,3],[3,[1,3]]] [[1,3],[3,[2,3]]] [[2,3],[1,[1,2]]] [[2,3],[1,[1,3]]] [[2,3],[2,[1,2]]] [[2,3],[2,[1,3]]] [[2,3],[2,[2,3]]] [[2,3],[3,[1,2]]] [[2,3],[3,[1,3]]] [[2,3],[3,[2,3]]] [1,[1,[1,[1,[1,2]]]]] [1,[1,[1,[1,[1,3]]]]] [2,[1,[1,[1,[1,2]]]]] [2,[1,[1,[1,[1,3]]]]] [2,[2,[1,[1,[1,2]]]]] [2,[2,[1,[1,[1,3]]]]] [2,[2,[2,[1,[1,2]]]]] [2,[2,[2,[1,[1,3]]]]] [2,[2,[2,[2,[1,2]]]]] [2,[2,[2,[2,[1,3]]]]] [2,[2,[2,[2,[2,3]]]]] [3,[1,[1,[1,[1,2]]]]] [3,[1,[1,[1,[1,3]]]]] [3,[2,[1,[1,[1,2]]]]] [3,[2,[1,[1,[1,3]]]]] [3,[2,[2,[1,[1,2]]]]] [3,[2,[2,[1,[1,3]]]]] [3,[2,[2,[2,[1,2]]]]] [3,[2,[2,[2,[1,3]]]]] [3,[2,[2,[2,[2,3]]]]] [3,[3,[1,[1,[1,2]]]]] [3,[3,[1,[1,[1,3]]]]] [3,[3,[2,[1,[1,2]]]]] [3,[3,[2,[1,[1,3]]]]] [3,[3,[2,[2,[1,2]]]]] [3,[3,[2,[2,[1,3]]]]] [3,[3,[2,[2,[2,3]]]]] [3,[3,[3,[1,[1,2]]]]] [3,[3,[3,[1,[1,3]]]]] [3,[3,[3,[2,[1,2]]]]] [3,[3,[3,[2,[1,3]]]]] [3,[3,[3,[2,[2,3]]]]] [3,[3,[3,[3,[1,2]]]]] [3,[3,[3,[3,[1,3]]]]] [3,[3,[3,[3,[2,3]]]]] [[1,2],[1,[1,[1,2]]]] [[1,2],[1,[1,[1,3]]]] [[1,2],[2,[1,[1,2]]]] [[1,2],[2,[1,[1,3]]]] [[1,2],[2,[2,[1,2]]]] [[1,2],[2,[2,[1,3]]]] [[1,2],[2,[2,[2,3]]]] [[1,2],[3,[1,[1,2]]]] [[1,2],[3,[1,[1,3]]]] [[1,2],[3,[2,[1,2]]]] [[1,2],[3,[2,[1,3]]]] [[1,2],[3,[2,[2,3]]]] [[1,2],[3,[3,[1,2]]]] [[1,2],[3,[3,[1,3]]]] [[1,2],[3,[3,[2,3]]]] [[1,2],[[1,2],[1,3]]] [[1,2],[[1,2],[2,3]]] [[1,3],[1,[1,[1,2]]]] [[1,3],[1,[1,[1,3]]]] [[1,3],[2,[1,[1,2]]]] [[1,3],[2,[1,[1,3]]]] [[1,3],[2,[2,[1,2]]]] [[1,3],[2,[2,[1,3]]]] [[1,3],[2,[2,[2,3]]]] [[1,3],[3,[1,[1,2]]]] [[1,3],[3,[1,[1,3]]]] [[1,3],[3,[2,[1,2]]]] [[1,3],[3,[2,[1,3]]]] [[1,3],[3,[2,[2,3]]]] [[1,3],[3,[3,[1,2]]]] [[1,3],[3,[3,[1,3]]]] [[1,3],[3,[3,[2,3]]]] [[1,3],[[1,2],[1,3]]] [[1,3],[[1,2],[2,3]]] [[1,3],[[1,3],[2,3]]] [[2,3],[1,[1,[1,2]]]] [[2,3],[1,[1,[1,3]]]] [[2,3],[2,[1,[1,2]]]] [[2,3],[2,[1,[1,3]]]] [[2,3],[2,[2,[1,2]]]] [[2,3],[2,[2,[1,3]]]] [[2,3],[2,[2,[2,3]]]] [[2,3],[3,[1,[1,2]]]] [[2,3],[3,[1,[1,3]]]] [[2,3],[3,[2,[1,2]]]] [[2,3],[3,[2,[1,3]]]] [[2,3],[3,[2,[2,3]]]] [[2,3],[3,[3,[1,2]]]] [[2,3],[3,[3,[1,3]]]] [[2,3],[3,[3,[2,3]]]] [[2,3],[[1,2],[1,3]]] [[2,3],[[1,2],[2,3]]] [[2,3],[[1,3],[2,3]]] [[1,[1,2]],[1,[1,3]]] [[1,[1,2]],[2,[1,2]]] [[1,[1,2]],[2,[1,3]]] [[1,[1,2]],[2,[2,3]]] [[1,[1,2]],[3,[1,2]]] [[1,[1,2]],[3,[1,3]]] [[1,[1,2]],[3,[2,3]]] [[1,[1,3]],[2,[1,2]]] [[1,[1,3]],[2,[1,3]]] [[1,[1,3]],[2,[2,3]]] [[1,[1,3]],[3,[1,2]]] [[1,[1,3]],[3,[1,3]]] [[1,[1,3]],[3,[2,3]]] [[2,[1,2]],[2,[1,3]]] [[2,[1,2]],[2,[2,3]]] [[2,[1,2]],[3,[1,2]]] [[2,[1,2]],[3,[1,3]]] [[2,[1,2]],[3,[2,3]]] [[2,[1,3]],[2,[2,3]]] [[2,[1,3]],[3,[1,2]]] [[2,[1,3]],[3,[1,3]]] [[2,[1,3]],[3,[2,3]]] [[2,[2,3]],[3,[1,2]]] [[2,[2,3]],[3,[1,3]]] [[2,[2,3]],[3,[2,3]]] [[3,[1,2]],[3,[1,3]]] [[3,[1,2]],[3,[2,3]]] [[3,[1,3]],[3,[2,3]]]'

    """
    np_hall_set, np_degrees, np_degree_boundaries, reverse_map, width = hall_basis(
        width, desired_degree
    )
    return " " + " ".join(
        [hb_to_string(z, width, desired_degree) for z in range(1, np_hall_set.shape[0])]
    )


def lie_to_string(li, width, depth):
    """
lie_to_string(prod(7, 6, 3, 6), 3, 6)
'-1.0 [[2,3],[1,[1,2]]]'

    """
    return " + ".join(
        [str(li[x]) + " " + hb_to_string(x, width, depth) for x in sorted(li.keys())]
    )


def key_to_sparse(k):
    """
>>> key_to_sparse(7)
defaultdict(<class 'float'>, {7: 1.0})
>>> 
    """
    ans = defaultdict(scalar_type)
    ans[k] = scalar_type(1)
    return ans


## add and subtract and scale sparse scalar_type vectors based on defaultdict class
"""
>>> lhs = key_to_sparse(3)
>>> rhs = key_to_sparse(5)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> subtract_into(lhs, rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 0.0})
>>> 
"""


def add_into(lhs, rhs):
    for k in rhs.keys():
        lhs[k] += rhs.get(k, scalar_type())
    return lhs


def subtract_into(lhs, rhs):
    for k in rhs.keys():
        lhs[k] -= rhs.get(k, scalar_type())
    return lhs


def scale_into(lhs, s):
    """
>>> rhs = key_to_sparse(5)
>>> lhs = key_to_sparse(3)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> scale_into(lhs, 3)
defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
>>> scale_into(lhs, 0)
defaultdict(<class 'float'>, {})
>>> 
    """
    if s:
        for k in lhs.keys():
            lhs[k] *= s
    else:
        lhs = defaultdict(scalar_type)
    return lhs


def sparsify(arg):
    """
>>> rhs = key_to_sparse(5)
>>> lhs = key_to_sparse(3)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> subtract_into(lhs, rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 0.0})
>>> sparsify(lhs)
defaultdict(<class 'float'>, {3: 1.0})
>>> 
    """
    empty_key_vals = list(k for k in arg.keys() if not arg[k])
    # an iterable would break
    for k in empty_key_vals:
        del arg[k]
    return arg


def multiply(lhs, rhs, func):
    ## WARNING assumes all multiplications are in range -
    ## if not then the product should use the coproduct and the max degree
    ans = defaultdict(scalar_type)
    for k1 in sorted(lhs.keys()):
        for k2 in sorted(rhs.keys()):
            add_into(ans, scale_into(func(k1, k2), lhs[k1] * rhs[k2]))
    return sparsify(ans)


def prod(k1, k2, width, depth):
    """
>>> rhs = key_to_sparse(5)
>>> lhs = key_to_sparse(3)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> scale_into(lhs, 3)
defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
>>> subtract_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 3.0, 5: 2.0})
>>> _prod = lambda kk1, kk2: prod(kk1, kk2, 3, 6)
>>> multiply(lhs,rhs,_prod)
defaultdict(<class 'float'>, {13: 3.0})
>>> multiply(rhs,lhs,_prod)
defaultdict(<class 'float'>, {13: -3.0})
>>> multiply(lhs,lhs,_prod)
defaultdict(<class 'float'>, {})
>>> 
    """
    _prod = lambda kk1, kk2: prod(kk1, kk2, width, depth)
    if k1 > k2:
        ans = _prod(k2, k1)
        scale_into(ans, -scalar_type(1))
        return ans
    ans = defaultdict(scalar_type)
    hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(width, depth)
    if k1 == k2:
        return ans
    if degrees[k1] + degrees[k2] > depth:
        return ans
    t = reverse_map.get((k1, k2), 0)
    if t:
        ans[t] = scalar_type(1)
    else:
        (k3, k4) = hall_set[k2]  ## (np.int32,np.int32)
        k3 = int(k3)
        k4 = int(k4)
        ### We use Jacobi: [k1,k2] = [k1,[k3,k4]]] = [[k1,k3],k4]-[[k1,k4],k3]
        wk13 = _prod(k1, k3)
        wk4 = key_to_sparse(k4)
        t1 = multiply(wk13, wk4, _prod)
        t2 = multiply(_prod(k1, k4), key_to_sparse(k3), _prod)
        ans = subtract_into(t1, t2)
    return ans


def sparse_to_dense(sparse, width, depth):
    """
>>> rhs = key_to_sparse(5)
>>> lhs = key_to_sparse(3)
>>> add_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 1.0, 5: 1.0})
>>> scale_into(lhs, 3)
defaultdict(<class 'float'>, {3: 3.0, 5: 3.0})
>>> subtract_into(lhs,rhs)
defaultdict(<class 'float'>, {3: 3.0, 5: 2.0})
>>> _prod = lambda kk1, kk2: prod(kk1, kk2, 3, 6)
>>> add_into(lhs , multiply(rhs,lhs,_prod))
defaultdict(<class 'float'>, {3: 3.0, 5: 2.0, 13: -3.0})
>>> sparse_to_dense(lhs,3,2)
array([0., 0., 3., 0., 2., 0.])
>>> 
    """
    ### is that last line correct??
    hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(width, depth)
    dense = np.zeros(len(hall_set), dtype=np.float64)
    for k in sparse.keys():
        if k < len(hall_set):
            dense[k] = sparse[k]
    return dense[1:]


def dense_to_sparse(dense, width, depth):
    """
>>> hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(2, 3)
>>> l = np.array( [i for i in range(1,len(hall_set))], dtype=np.float64)
>>> print(l," ",dense_to_sparse(l,2,3))
[1. 2. 3. 4. 5.]   defaultdict(<class 'float'>, {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0})
>>> hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(2, 3)
>>> l = np.array( [i for i in range(1,len(hall_set))], dtype=np.float64)
>>> print(l," ",dense_to_sparse(l,2,3))
[1. 2. 3. 4. 5.]   defaultdict(<class 'float'>, {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0})
>>> 
>>> sparse_to_dense(dense_to_sparse(l,2,3), 2,3) == l
array([ True,  True,  True,  True,  True])
>>> 
    """
    sparse = defaultdict(scalar_type)
    for k in range(len(dense)):
        if dense[k]:
            sparse[k + 1] = dense[k]
    return sparse


## expand is a map from hall basis keys to tensors
def expand(k, width, depth):
    _expand = lambda k: expand(k, width, depth)
    _tensor_multiply = lambda k1, k2: tjl_dense_numpy_tensor.tensor_multiply(
        k1, k2, depth
    )
    _tensor_sub = tjl_dense_numpy_tensor.tensor_sub
    if k:
        hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(
            width, depth
        )
        (k1, k2) = hall_set[k]
        if k1:
            return _tensor_sub(
                _tensor_multiply(_expand(k1), _expand(k2)),
                _tensor_multiply(_expand(k2), _expand(k1)),
            )
        else:
            ans = tjl_dense_numpy_tensor.zero(width, 1)
            ans[blob_size(width, 0) - 1 + k2] = scalar_type(
                1
            )  ## recall k2 will never be zero
            return ans
    return tjl_dense_numpy_tensor.zero(width)


def l2t(arg, width, depth):
    """
>>> from tjl_hall_numpy_lie import *
>>> from tjl_dense_numpy_tensor import *
>>> width = 2
>>> depth = 3
>>> t = tensor_log(stream2sigtensor(brownian(100, width), depth), depth)
>>> print(np.sum(tensor_sub(l2t(t2l(t),width,depth), t)[2:]**2)  < 1e-30)
True
>>> 
    """
    _expand = lambda k: expand(k, width, depth)
    _tensor_add = tjl_dense_numpy_tensor.tensor_add
    ans = tjl_dense_numpy_tensor.zero(width)
    for k in arg.keys():
        if k:
            ans = tjl_dense_numpy_tensor.tensor_add(
                ans, (tjl_dense_numpy_tensor.rescale(_expand(k), arg[k]))
            )
    return ans


## tuple a1,a2,...,an is converted into [a1,[a2,[...,an]]] as a LIE element recursively.
def rbraketing(tk, width, depth):
    _rbracketing = lambda t: rbraketing(t, width, depth)
    _prod = lambda x, y: prod(x, y, width, depth)
    _multiply = lambda k1, k2: multiply(k1, k2, _prod)
    hall_set, degrees, degree_boundaries, reverse_map, width = hall_basis(width, depth)
    if tk[1:]:
        return _multiply(_rbracketing(tk[:1]), _rbracketing(tk[1:]))
    else:
        ans = defaultdict(scalar_type)
        if tk:
            ans[tk[0]] = scalar_type(1)
        return ans


def index_to_tuple(i, width):
    # the shape of the tensor that contains
    # \sum t[k] index_to_tuple(k,width) is the tensor
    # None () (0) (1) ...(w-1) (0,0) ...(n1,...,nd) ...
    """
>>> from tjl_hall_numpy_lie import *
>>> from tjl_dense_numpy_tensor import *
>>> width = 2
>>> depth = 3
>>> t = arange(width, depth)
>>> print(t)
[ 2.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15.]
>>> for t1 in [t]:
...     for k, coeff in enumerate(t1):
...         print (coeff, index_to_tuple(k,width))
... 
2.0 None
1.0 ()
2.0 (1,)
3.0 (2,)
4.0 (1, 1)
5.0 (1, 2)
6.0 (2, 1)
7.0 (2, 2)
8.0 (1, 1, 1)
9.0 (1, 1, 2)
10.0 (1, 2, 1)
11.0 (1, 2, 2)
12.0 (2, 1, 1)
13.0 (2, 1, 2)
14.0 (2, 2, 1)
15.0 (2, 2, 2)
>>> 
    """

    _blob_size = lambda depth: tjl_dense_numpy_tensor.blob_size(width, depth)
    _layers = lambda bz: tjl_dense_numpy_tensor.layers(bz, width)
    bz = i + 1
    d = _layers(bz)  ## this index is in the d-tensors
    if _layers(bz) < 0:
        return
    j = bz - 1 - _blob_size(d - 1)
    ans = ()
    ## remove initial offset to compute the index
    if j >= 0:
        for jj in range(d):
            ans = (1 + (j % width),) + ans
            j = j // width
    return ans


def t_to_string(i, width):
    j = index_to_tuple(i, width)
    if index_to_tuple(i, width) == None:
        return " "
    return "(" + ",".join([str(k) for k in index_to_tuple(i, width)]) + ")"


def sigkeys(width, desired_degree):
    """
>>> from tjl_hall_numpy_lie import *
>>> from tjl_dense_numpy_tensor import *
>>> width = 2
>>> depth = 3
>>> from esig import tosig as ts
>>> ts.sigkeys(width , depth) == sigkeys(width , depth)
True
>>> sigkeys(width , depth)
' () (1) (2) (1,1) (1,2) (2,1) (2,2) (1,1,1) (1,1,2) (1,2,1) (1,2,2) (2,1,1) (2,1,2) (2,2,1) (2,2,2)'
>>> 
    """
    t_to_string(0, width)
    return " " + " ".join(
        [t_to_string(z, width) for z in range(1, blob_size(width, desired_degree))]
    )


def t2l(arg):
    # projects a lie element in tensor form to a lie in lie basis form
    """
>>> from tjl_hall_numpy_lie import *
>>> from tjl_dense_numpy_tensor import *
>>> width = 2
>>> depth = 3
>>> t = tensor_log(stream2sigtensor(brownian(100, width), depth), depth)
>>> print(np.sum(tensor_sub(l2t(t2l(t),width,depth), t)[2:]**2)  < 1e-30)
True
>>> 
    """
    width = int(arg[0])
    _layers = lambda bz: tjl_dense_numpy_tensor.layers(bz, width)
    _blob_size = lambda dep: tjl_dense_numpy_tensor.blob_size(width, dep)
    depth = _layers(len(arg))
    ans = defaultdict(scalar_type)
    ibe = _blob_size(0)  # just beyond the zero tensors
    ien = _blob_size(depth)
    for i in range(ibe, ien):
        t = index_to_tuple(i, width)
        if t:
            ## must normalise to get the dynkin projection to make it a projection
            add_into(
                ans,
                scale_into(rbraketing(t, width, depth), arg[i] / scalar_type(len(t))),
            )
    return ans
