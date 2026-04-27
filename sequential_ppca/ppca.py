"""Batch Probabilistic PCA via the EM algorithm.

Reference
---------
Tipping, M. E. & Bishop, C. M. (1999). Probabilistic principal component
analysis. *Journal of the Royal Statistical Society: Series B*, 61(3),
611-622.
"""

import numpy as np
from scipy import linalg


class PPCA:
    """Probabilistic Principal Component Analysis (batch EM).

    Fits the generative model::

        x = W z + mu + eps
        z   ~ N(0, I_q)
        eps ~ N(0, sigma2 * I_p)

    where *W* is a ``(p, q)`` loading matrix, *mu* is the ``p``-dimensional
    mean, and *sigma2* is the isotropic noise variance.

    The maximum-likelihood solution is found by running the EM algorithm
    described in Tipping & Bishop (1999).

    Parameters
    ----------
    n_components : int
        Number of latent dimensions *q*.
    n_iter : int, default=200
        Maximum number of EM iterations.
    tol : float, default=1e-6
        Convergence threshold on the absolute change in log-likelihood
        between successive iterations.
    random_state : int or None, default=None
        Seed for the random number generator used to initialise *W*.

    Attributes
    ----------
    W_ : ndarray of shape (n_features, n_components)
        Fitted loading matrix.
    sigma2_ : float
        Fitted noise variance.
    mean_ : ndarray of shape (n_features,)
        Sample mean of the training data.
    log_likelihoods_ : list of float
        Log-likelihood evaluated after each EM iteration.
    n_iter_ : int
        Number of EM iterations actually performed.
    """

    def __init__(self, n_components, n_iter=200, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X):
        """Fit the PPCA model to training data *X* using the EM algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training observations.

        Returns
        -------
        self : PPCA
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, p = X.shape
        q = self.n_components

        if q >= p:
            raise ValueError(
                f"n_components ({q}) must be strictly less than "
                f"n_features ({p})."
            )

        rng = np.random.RandomState(self.random_state)

        # ---- initialise parameters -----------------------------------
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        self.W_ = rng.standard_normal((p, q)) * 0.1
        self.sigma2_ = float(np.var(Xc))

        prev_ll = -np.inf
        self.log_likelihoods_ = []

        for i in range(self.n_iter):
            W = self.W_
            s2 = self.sigma2_

            # ---- E-step ----------------------------------------------
            # M = W^T W + sigma2 I_q  (q x q)
            M = W.T @ W + s2 * np.eye(q)
            Minv = linalg.solve(M, np.eye(q), assume_a="pos")

            # Posterior means:  Ez[i] = M^{-1} W^T (x_i - mu)
            Ez = Xc @ W @ Minv.T               # (n, q)

            # Sum of posterior second moments: sum_i E[z_i z_i^T | x_i]
            S2 = n * s2 * Minv + Ez.T @ Ez     # (q, q)

            # ---- M-step ----------------------------------------------
            S1 = Xc.T @ Ez                     # (p, q)

            # W_new = S1 @ inv(S2)
            W_new = linalg.solve(S2.T, S1.T, assume_a="pos").T

            # sigma2_new = (1 / (n*p)) * (S0 - tr(W_new^T S1))
            # (uses the identity tr(S2 W_new^T W_new) = tr(W_new^T S1))
            S0 = float(np.sum(Xc ** 2))
            s2_new = (S0 - float(np.trace(W_new.T @ S1))) / (n * p)
            s2_new = max(s2_new, 1e-8)

            self.W_ = W_new
            self.sigma2_ = s2_new

            # ---- convergence check -----------------------------------
            ll = self._log_likelihood_centered(Xc)
            self.log_likelihoods_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                self.n_iter_ = i + 1
                return self
            prev_ll = ll

        self.n_iter_ = self.n_iter
        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _posterior_gain(self):
        """Return ``M^{-1} W^T`` — the gain matrix for the posterior mean.

        Shape: (n_components, n_features).
        """
        q = self.n_components
        M = self.W_.T @ self.W_ + self.sigma2_ * np.eye(q)
        return linalg.solve(M, self.W_.T, assume_a="pos")   # (q, p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, X):
        """Project observations *X* to the latent space.

        Returns the posterior mean ``E[z | x]`` for every observation.

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

    def fit_transform(self, X):
        """Fit the model and return the latent projections of *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        Z : ndarray of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

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

    def _log_likelihood_centered(self, Xc):
        """Total log-likelihood for zero-mean data *Xc*."""
        n, p = Xc.shape
        C = self.W_ @ self.W_.T + self.sigma2_ * np.eye(p)
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            return -np.inf
        Cinv_Xc = linalg.solve(C, Xc.T, assume_a="pos").T   # (n, p)
        return -0.5 * (n * (p * np.log(2.0 * np.pi) + logdet)
                       + np.sum(Xc * Cinv_Xc))

    def log_likelihood(self, X):
        """Total log-likelihood of *X* under the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ll : float
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._log_likelihood_centered(X - self.mean_)

    def score(self, X):
        """Average log-likelihood per sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        score : float
        """
        X = np.asarray(X, dtype=float)
        return self.log_likelihood(X) / len(X)
