"""Sequential (online) Probabilistic PCA.

Incrementally updates a PPCA model as data arrive in mini-batches,
without storing previously seen observations.

Algorithm
---------
The algorithm maintains running *sufficient statistics* that grow as new
data are presented, and performs one EM M-step after every mini-batch.

Let ``N`` be the total number of observations processed so far.

**Running statistics** (updated with each new batch)::

    sum_x    = sum_i  x_i                     shape (p,)
    sum_xx   = sum_i  ||x_i||^2               scalar
    sum_xEz  = sum_i  x_i * E[z_i]^T         shape (p, q)
    sum_Ez   = sum_i  E[z_i]                   shape (q,)
    sum_EzzT = sum_i  E[z_i z_i^T | x_i]     shape (q, q)

From these, the EM sufficient statistics are recovered exactly::

    mu   = sum_x / N
    S0   = sum_xx - N * ||mu||^2
    S1   = sum_xEz - outer(mu, sum_Ez)   # (p, q)
    S2   = sum_EzzT                       # (q, q)

The M-step updates are then::

    W_new    = S1 @ inv(S2)
    sigma2_new = (S0 - tr(W_new^T @ S1)) / (N * p)

The key approximation in the online setting is that ``E[z_i]`` for a
historical mini-batch was computed using the model parameters *at the time
that batch was processed*, rather than the current (latest) parameters.
This is the standard *online EM* approximation (Cappé & Moulines, 2009)
and is accurate when batches are large or when the model has converged.

References
----------
* Tipping, M. E. & Bishop, C. M. (1999). Probabilistic principal component
  analysis. *JRSS-B*, 61(3), 611-622.
* Cappé, O. & Moulines, E. (2009). On-line expectation–maximization
  algorithm for latent data models. *JRSS-B*, 71(3), 593-613.
"""

import numpy as np
from scipy import linalg


class SequentialPPCA:
    """Sequential (online) Probabilistic PCA.

    Incrementally updates a PPCA model as new data arrive in mini-batches,
    without storing previously seen observations.

    The generative model is identical to :class:`PPCA`::

        x = W z + mu + eps
        z   ~ N(0, I_q)
        eps ~ N(0, sigma2 * I_p)

    Parameters
    ----------
    n_components : int
        Number of latent dimensions *q*.
    n_features : int
        Dimensionality of each observation *p*.
    random_state : int or None, default=None
        Seed for the random number generator used to initialise *W*.

    Attributes
    ----------
    W_ : ndarray of shape (n_features, n_components)
        Current loading matrix estimate.
    sigma2_ : float
        Current noise-variance estimate.
    mean_ : ndarray of shape (n_features,)
        Running mean of all observations seen so far.
    n_samples_seen_ : int
        Total number of observations processed.
    """

    def __init__(self, n_components, n_features, random_state=None):
        self.n_components = n_components
        self.n_features = n_features
        self.random_state = random_state

        p, q = n_features, n_components
        rng = np.random.RandomState(random_state)

        # ---- model parameters (updated after each batch) -------------
        self.W_ = rng.standard_normal((p, q)) * 0.1
        self.sigma2_ = 1.0
        self.mean_ = np.zeros(p)

        # ---- running sufficient statistics ---------------------------
        self.n_samples_seen_ = 0
        self._sum_x = np.zeros(p)           # sum_i  x_i
        self._sum_xx = 0.0                  # sum_i  ||x_i||^2
        self._sum_xEz = np.zeros((p, q))    # sum_i  x_i E[z_i]^T
        self._sum_Ez = np.zeros(q)          # sum_i  E[z_i]
        self._sum_EzzT = np.zeros((q, q))   # sum_i  E[z_i z_i^T | x_i]

    # ------------------------------------------------------------------
    # Incremental fitting
    # ------------------------------------------------------------------

    def partial_fit(self, X):
        """Update the model with a new mini-batch of observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New mini-batch.

        Returns
        -------
        self : SequentialPPCA
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, p = X.shape
        q = self.n_components

        if p != self.n_features:
            raise ValueError(
                f"Expected n_features={self.n_features}, got {p}."
            )

        # ---- E-step with current parameters and current mean ---------
        # Use the mean estimated from all *previously* seen data.
        Xc = X - self.mean_

        W = self.W_
        s2 = self.sigma2_
        M = W.T @ W + s2 * np.eye(q)
        Minv = linalg.solve(M, np.eye(q), assume_a="pos")

        Ez = Xc @ W @ Minv.T               # (n, q): E[z_i | x_i]
        EzzT_sum = n * s2 * Minv + Ez.T @ Ez   # (q, q): sum_i E[z_i z_i^T]

        # ---- update running sufficient statistics --------------------
        self._sum_x += X.sum(axis=0)
        self._sum_xx += float(np.sum(X ** 2))
        self._sum_xEz += X.T @ Ez          # uses raw (un-centered) x
        self._sum_Ez += Ez.sum(axis=0)
        self._sum_EzzT += EzzT_sum
        self.n_samples_seen_ += n

        N = self.n_samples_seen_

        # ---- recover centered sufficient statistics ------------------
        # mean from ALL data seen so far (including this batch)
        self.mean_ = self._sum_x / N

        # S0 = sum ||x_i - mu||^2  (computed without centering history)
        S0 = self._sum_xx - N * float(np.dot(self.mean_, self.mean_))
        S0 = max(S0, 0.0)   # guard against tiny negative values from rounding

        # S1 = sum (x_i - mu) E[z_i]^T
        S1 = self._sum_xEz - np.outer(self.mean_, self._sum_Ez)   # (p, q)

        S2 = self._sum_EzzT   # (q, q)

        # ---- M-step --------------------------------------------------
        # Require at least q observations before attempting the M-step
        # so that S2 is likely to be non-singular.
        if N < q:
            return self

        try:
            # W_new = S1 @ inv(S2)
            W_new = linalg.solve(S2.T, S1.T, assume_a="pos").T

            # sigma2_new = (S0 - tr(W_new^T S1)) / (N * p)
            s2_new = (S0 - float(np.trace(W_new.T @ S1))) / (N * p)
            s2_new = max(s2_new, 1e-8)

            self.W_ = W_new
            self.sigma2_ = s2_new
        except linalg.LinAlgError:
            # S2 is (near-)singular; keep previous parameters
            pass

        return self

    def reset_statistics(self):
        """Clear the accumulated sufficient statistics.

        After calling this method the model *parameters* (``W_``,
        ``sigma2_``, ``mean_``) are preserved, but the running statistics
        are zeroed out and ``n_samples_seen_`` is reset to zero.

        This is useful for **epoch-based** training: at the start of each
        new pass over the dataset, call ``reset_statistics()`` so that only
        the current epoch's data inform the M-step, rather than a growing
        mixture of old and new E-step contributions.

        Example
        -------
        >>> for epoch in range(n_epochs):
        ...     model.reset_statistics()
        ...     for batch in batches:
        ...         model.partial_fit(batch)

        Returns
        -------
        self : SequentialPPCA
        """
        p, q = self.n_features, self.n_components
        self.n_samples_seen_ = 0
        self._sum_x = np.zeros(p)
        self._sum_xx = 0.0
        self._sum_xEz = np.zeros((p, q))
        self._sum_Ez = np.zeros(q)
        self._sum_EzzT = np.zeros((q, q))
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _posterior_gain(self):
        """Return ``M^{-1} W^T`` — the posterior gain matrix.

        Shape: (n_components, n_features).
        """
        q = self.n_components
        M = self.W_.T @ self.W_ + self.sigma2_ * np.eye(q)
        return linalg.solve(M, self.W_.T, assume_a="pos")   # (q, p)

    def transform(self, X):
        """Project observations to the latent space.

        Returns the posterior mean ``E[z | x]`` for each observation,
        computed under the current model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        Z : ndarray of shape (n_samples, n_components)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xc = X - self.mean_
        G = self._posterior_gain()   # (q, p)
        return Xc @ G.T              # (n, q)

    def inverse_transform(self, Z):
        """Reconstruct observations from latent coordinates.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components)

        Returns
        -------
        X_rec : ndarray of shape (n_samples, n_features)
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        return Z @ self.W_.T + self.mean_

    def score(self, X):
        """Average log-likelihood per sample under the current model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        score : float
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, p = X.shape
        Xc = X - self.mean_
        C = self.W_ @ self.W_.T + self.sigma2_ * np.eye(p)
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            return -np.inf
        Cinv_Xc = linalg.solve(C, Xc.T, assume_a="pos").T   # (n, p)
        ll = -0.5 * (n * (p * np.log(2.0 * np.pi) + logdet)
                     + np.sum(Xc * Cinv_Xc))
        return ll / n
