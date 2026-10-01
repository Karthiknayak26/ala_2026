import numpy as np

a = np.zeros(4)

# Correct ways to assert array equality:
assert (a == [0, 0, 0, 0]).all()
assert np.array_equal(a, [0, 0, 0, 0])
assert a.shape == (4,)
assert isinstance(a.shape, tuple)
