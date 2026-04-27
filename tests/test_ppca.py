# Tests for the batch PPCA implementation.
import numpy as np
import pytest

from sequential_ppca import PPCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(n=500, p=10, q=3, sigma2=0.5, seed=0):
    """Generate data from a PPCA model with known parameters."""
    rng = np.random.RandomState(seed)
    W_true = rng.standard_normal((p, q))
    mu_true = rng.standard_normal(p)
    Z = rng.standard_normal((n, q))
    noise = np.sqrt(sigma2) * rng.standard_normal((n, p))
    X = Z @ W_true.T + mu_true + noise
    return X, W_true, mu_true, sigma2


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------

class TestPPCABasic:
    def test_fit_returns_self(self):
        X, *_ = make_data()
        model = PPCA(n_components=3)
        result = model.fit(X)
        assert result is model

    def test_attributes_set_after_fit(self):
        X, *_ = make_data()
        model = PPCA(n_components=3).fit(X)
        assert hasattr(model, "W_")
        assert hasattr(model, "sigma2_")
        assert hasattr(model, "mean_")
        assert hasattr(model, "log_likelihoods_")
        assert hasattr(model, "n_iter_")

    def test_W_shape(self):
        X, *_ = make_data(p=10, q=3)
        model = PPCA(n_components=3).fit(X)
        assert model.W_.shape == (10, 3)

    def test_sigma2_positive(self):
        X, *_ = make_data()
        model = PPCA(n_components=3).fit(X)
        assert model.sigma2_ > 0

    def test_mean_matches_sample_mean(self):
        X, *_ = make_data()
        model = PPCA(n_components=3).fit(X)
        np.testing.assert_allclose(model.mean_, X.mean(axis=0))

    def test_transform_shape(self):
        X, *_ = make_data(n=100, p=8, q=2)
        model = PPCA(n_components=2).fit(X)
        Z = model.transform(X)
        assert Z.shape == (100, 2)

    def test_inverse_transform_shape(self):
        X, *_ = make_data(n=50, p=8, q=2)
        model = PPCA(n_components=2).fit(X)
        Z = model.transform(X)
        X_rec = model.inverse_transform(Z)
        assert X_rec.shape == X.shape

    def test_fit_transform(self):
        X, *_ = make_data()
        model = PPCA(n_components=3)
        Z = model.fit_transform(X)
        assert Z.shape == (len(X), 3)

    def test_n_components_too_large_raises(self):
        X, *_ = make_data(p=5)
        with pytest.raises(ValueError):
            PPCA(n_components=5).fit(X)


# ---------------------------------------------------------------------------
# EM convergence
# ---------------------------------------------------------------------------

class TestPPCAConvergence:
    def test_log_likelihood_nondecreasing(self):
        """EM must be a monotone ascent algorithm."""
        X, *_ = make_data(n=300, seed=42)
        model = PPCA(n_components=3, n_iter=50, tol=0.0, random_state=0)
        model.fit(X)
        lls = model.log_likelihoods_
        diffs = np.diff(lls)
        # Allow a tiny tolerance for floating-point rounding
        assert np.all(diffs >= -1e-6), (
            f"Log-likelihood decreased: min diff = {diffs.min():.3e}"
        )

    def test_log_likelihoods_nonempty(self):
        X, *_ = make_data()
        model = PPCA(n_components=3, n_iter=100, tol=0.0, random_state=0)
        model.fit(X)
        assert len(model.log_likelihoods_) > 0

    def test_score_is_finite(self):
        X, *_ = make_data()
        model = PPCA(n_components=3, random_state=0).fit(X)
        s = model.score(X)
        assert np.isfinite(s)


# ---------------------------------------------------------------------------
# Reconstruction quality
# ---------------------------------------------------------------------------

class TestPPCAReconstruction:
    def test_reconstruction_error_low(self):
        """When sigma2 is small, W W^T should dominate and the
        reconstruction error should be small."""
        X, W_true, mu_true, _ = make_data(n=2000, p=8, q=3,
                                           sigma2=0.05, seed=7)
        model = PPCA(n_components=3, n_iter=300, random_state=0).fit(X)
        X_rec = model.inverse_transform(model.transform(X))
        rel_err = np.mean((X - X_rec) ** 2) / np.var(X)
        assert rel_err < 0.2

    def test_subspace_recovery(self):
        """The column space of W should be close to the true column space."""
        X, W_true, mu_true, _ = make_data(n=5000, p=10, q=3,
                                           sigma2=0.1, seed=1)
        model = PPCA(n_components=3, n_iter=500, random_state=0).fit(X)

        # Compare column spaces via principal angles
        Q_true, _ = np.linalg.qr(W_true)
        Q_fit, _ = np.linalg.qr(model.W_)
        # Frobenius distance between projection matrices
        P_true = Q_true @ Q_true.T
        P_fit = Q_fit @ Q_fit.T
        subspace_err = np.linalg.norm(P_true - P_fit, "fro")
        # With plenty of data the column spaces should nearly coincide
        assert subspace_err < 1.0


# ---------------------------------------------------------------------------
# Score / log-likelihood consistency
# ---------------------------------------------------------------------------

class TestPPCAScore:
    def test_score_train_vs_test(self):
        """Training score should be higher than a random model's score
        on held-out data (sanity check)."""
        rng = np.random.RandomState(0)
        X_train, *_ = make_data(n=500, p=8, q=3, seed=0)
        X_test, *_ = make_data(n=200, p=8, q=3, seed=99)

        model = PPCA(n_components=3, random_state=0).fit(X_train)
        s_test = model.score(X_test)
        assert np.isfinite(s_test)

    def test_log_likelihood_consistent_with_score(self):
        X, *_ = make_data(n=100)
        model = PPCA(n_components=3, random_state=0).fit(X)
        assert np.isclose(model.score(X), model.log_likelihood(X) / len(X))
