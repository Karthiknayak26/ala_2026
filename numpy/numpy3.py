import numpy as np
a = np.array(np.arange(1, 25, 1, np.int8))
assert (a == [x for x in range(1, 25)]).all()
b = a.reshape(2, 12)
assert(b.shape == (2, 12))
assert(b.size == 24)
assert(len(b) == 2)
assert(b.ndim == 2)