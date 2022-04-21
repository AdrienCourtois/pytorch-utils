from opt_einsum import oe
from functools import partial 

einsum = partial(oe.contract, backend="torch")
