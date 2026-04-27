"""Sequential Probabilistic PCA.

Classes
-------
PPCA
    Batch probabilistic PCA via the EM algorithm.
SequentialPPCA
    Online/mini-batch probabilistic PCA that updates incrementally.
"""

from .ppca import PPCA
from .sequential_ppca import SequentialPPCA

__all__ = ["PPCA", "SequentialPPCA"]
