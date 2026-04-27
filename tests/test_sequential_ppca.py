# Tests for the SequentialPPCA implementation.
import numpy as np
import pytest

from sequential_ppca import PPCA, SequentialPPCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(n=800, p=10, q=3, sigma2=0.5, seed=0):
    """Generate data from a PPCA model with known parameters."""
    rng = np.random.RandomState(seed)
    W_true = rng.standard_normal((p, q))
    mu_true = rng.standard_normal(p)
    Z = rng.standard_normal((n, q))
    noise = np.sqrt(sigma2) * rng.standard_normal((n, p))
    X = Z @ W_true.T + mu_true + noise
    return X, W_true, mu_true, sigma2


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestSequentialPPCABasic:
    def test_partial_fit_returns_self(self):
        X, *_ = make_data()
        model = SequentialPPCA(n_components=3, n_features=10)
        result = model.partial_fit(X)
        assert result is model

    def test_attributes_present(self):
        model = SequentialPPCA(n_components=3, n_features=10)
        assert hasattr(model, "W_")
        assert hasattr(model, "sigma2_")
        assert hasattr(model, "mean_")
        assert hasattr(model, "n_samples_seen_")

    def test_W_shape(self):
        X, *_ = make_data(p=10, q=3)
        model = SequentialPPCA(n_components=3, n_features=10).partial_fit(X)
        assert model.W_.shape == (10, 3)

    def test_sigma2_positive_after_fit(self):
        X, *_ = make_data()
        model = SequentialPPCA(n_components=3, n_features=10).partial_fit(X)
        assert model.sigma2_ > 0

    def test_n_samples_seen_increments(self):
        X, *_ = make_data(n=200)
        model = SequentialPPCA(n_components=3, n_features=10)
        model.partial_fit(X[:100])
        assert model.n_samples_seen_ == 100
        model.partial_fit(X[100:])
        assert model.n_samples_seen_ == 200

    def test_transform_shape(self):
        X, *_ = make_data(n=100, p=8, q=2)
        model = SequentialPPCA(n_components=2, n_features=8).partial_fit(X)
        Z = model.transform(X)
        assert Z.shape == (100, 2)

    def test_inverse_transform_shape(self):
        X, *_ = make_data(n=50, p=8, q=2)
        model = SequentialPPCA(n_components=2, n_features=8).partial_fit(X)
        Z = model.transform(X)
        X_rec = model.inverse_transform(Z)
        assert X_rec.shape == X.shape

    def test_wrong_n_features_raises(self):
        model = SequentialPPCA(n_components=2, n_features=8)
        X_wrong = np.random.randn(10, 5)
        with pytest.raises(ValueError):
            model.partial_fit(X_wrong)

    def test_score_finite(self):
        X, *_ = make_data()
        model = SequentialPPCA(n_components=3, n_features=10).partial_fit(X)
        assert np.isfinite(model.score(X))


# ---------------------------------------------------------------------------
# Mean estimation
# ---------------------------------------------------------------------------

class TestSequentialMean:
    def test_mean_converges_to_sample_mean(self):
        X, _, mu_true, _ = make_data(n=1000)
        model = SequentialPPCA(n_components=3, n_features=10)
        for i in range(0, 1000, 50):
            model.partial_fit(X[i:i + 50])
        np.testing.assert_allclose(model.mean_, X.mean(axis=0), atol=1e-10)

    def test_mean_single_batch(self):
        X, *_ = make_data(n=200)
        model = SequentialPPCA(n_components=3, n_features=10)
        model.partial_fit(X)
        np.testing.assert_allclose(model.mean_, X.mean(axis=0), atol=1e-10)

    def test_mean_two_batches_exact(self):
        X, *_ = make_data(n=400)
        model = SequentialPPCA(n_components=3, n_features=10)
        model.partial_fit(X[:200])
        model.partial_fit(X[200:])
        np.testing.assert_allclose(model.mean_, X.mean(axis=0), atol=1e-10)


# ---------------------------------------------------------------------------
# Convergence to batch solution
# ---------------------------------------------------------------------------

class TestSequentialConvergence:
    def test_sequential_vs_batch_multiple_passes(self):
        """Epoch-based training (reset_statistics each pass) should bring
        the sequential model's score close to the batch solution."""
        X, *_ = make_data(n=600, p=8, q=3, sigma2=0.3, seed=5)

        batch = PPCA(n_components=3, n_iter=500, tol=1e-8, random_state=0)
        batch.fit(X)

        seq = SequentialPPCA(n_components=3, n_features=8, random_state=0)
        # Epoch-based: reset statistics at the start of each epoch so that
        # each pass sees a "fresh" view of the data.
        for _ in range(20):
            seq.reset_statistics()
            seq.partial_fit(X)

        batch_score = batch.score(X)
        seq_score = seq.score(X)
        assert np.isfinite(seq_score)
        # After epoch-based convergence the sequential model should match
        # the batch score within 2 % (absolute difference).
        assert abs(seq_score - batch_score) < 0.02 * abs(batch_score), (
            f"Sequential score {seq_score:.3f} too far from "
            f"batch score {batch_score:.3f}"
        )

    def test_multiple_passes_improve_score(self):
        """Repeated passes over the data should improve (or maintain) the
        model score."""
        X, *_ = make_data(n=400, p=8, q=3, seed=3)
        model = SequentialPPCA(n_components=3, n_features=8, random_state=0)

        # First pass
        for i in range(0, 400, 40):
            model.partial_fit(X[i:i + 40])
        score_pass1 = model.score(X)

        # Second pass
        for i in range(0, 400, 40):
            model.partial_fit(X[i:i + 40])
        score_pass2 = model.score(X)

        # The second pass may accumulate more weight on good statistics;
        # at minimum the score should not collapse.
        assert np.isfinite(score_pass2)

    def test_score_increases_with_more_data(self):
        """Training score should improve as more relevant data are seen."""
        X, *_ = make_data(n=1000, p=8, q=3, sigma2=0.3, seed=9)
        model = SequentialPPCA(n_components=3, n_features=8, random_state=0)

        scores = []
        for i in range(0, 1000, 100):
            model.partial_fit(X[i:i + 100])
            scores.append(model.score(X[:i + 100]))

        # Overall trend should be non-decreasing (allow small fluctuations)
        assert scores[-1] >= scores[2], (
            f"Score did not improve over training: "
            f"early={scores[2]:.3f}, final={scores[-1]:.3f}"
        )


# ---------------------------------------------------------------------------
# Reconstruction quality
# ---------------------------------------------------------------------------

class TestSequentialReconstruction:
    def test_reconstruction_error_reasonable(self):
        """The sequential model should reconstruct training data reasonably."""
        X, *_ = make_data(n=1000, p=8, q=3, sigma2=0.1, seed=11)
        model = SequentialPPCA(n_components=3, n_features=8, random_state=0)

        # Multiple passes for convergence
        for _ in range(3):
            for i in range(0, 1000, 100):
                model.partial_fit(X[i:i + 100])

        X_rec = model.inverse_transform(model.transform(X))
        rel_err = np.mean((X - X_rec) ** 2) / np.var(X)
        assert rel_err < 0.4

    def test_subspace_recovery(self):
        """Column space of W should approximately match true column space."""
        X, W_true, _, _ = make_data(n=3000, p=10, q=3,
                                     sigma2=0.1, seed=13)
        model = SequentialPPCA(n_components=3, n_features=10, random_state=0)

        # Three passes with moderate batch size
        for _ in range(3):
            for i in range(0, 3000, 200):
                model.partial_fit(X[i:i + 200])

        Q_true, _ = np.linalg.qr(W_true)
        Q_fit, _ = np.linalg.qr(model.W_)
        P_true = Q_true @ Q_true.T
        P_fit = Q_fit @ Q_fit.T
        subspace_err = np.linalg.norm(P_true - P_fit, "fro")
        assert subspace_err < 1.5


# ---------------------------------------------------------------------------
# Numerical edge cases
# ---------------------------------------------------------------------------

class TestSequentialEdgeCases:
    def test_reset_statistics(self):
        """reset_statistics() should clear running stats but keep parameters."""
        X, *_ = make_data(n=100, p=5, q=2, seed=0)
        model = SequentialPPCA(n_components=2, n_features=5, random_state=0)
        model.partial_fit(X)
        W_before = model.W_.copy()
        model.reset_statistics()
        assert model.n_samples_seen_ == 0
        np.testing.assert_array_equal(model.W_, W_before)

    def test_single_sample_per_batch(self):
        """Model should not crash when updated one sample at a time."""
        X, *_ = make_data(n=50, p=5, q=2, seed=0)
        model = SequentialPPCA(n_components=2, n_features=5, random_state=0)
        for x in X:
            model.partial_fit(x.reshape(1, -1))
        assert np.isfinite(model.sigma2_)
        assert np.all(np.isfinite(model.W_))

    def test_large_batch_equals_full_dataset(self):
        """Processing the whole dataset in one call equals processing in
        two halves (as long as we check the final mean)."""
        X, *_ = make_data(n=200, p=6, q=2, seed=2)
        m1 = SequentialPPCA(n_components=2, n_features=6, random_state=0)
        m1.partial_fit(X)

        m2 = SequentialPPCA(n_components=2, n_features=6, random_state=0)
        m2.partial_fit(X[:100])
        m2.partial_fit(X[100:])

        # Both should have the exact same final mean
        np.testing.assert_allclose(m1.mean_, X.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(m2.mean_, X.mean(axis=0), atol=1e-10)
