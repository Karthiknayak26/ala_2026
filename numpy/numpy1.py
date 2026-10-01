import numpy as np
a=np.array([1,2,3])
assert(a.dtype==np.int64)
a.shape
assert a.shape==(3,)
assert(isinstance(a.shape,tuple))
a
a.T.shape