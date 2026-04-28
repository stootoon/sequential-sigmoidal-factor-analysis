import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

class SequentialSigmoidalFactorAnalysis: 
    """
    x[t+1] = L x[t] + b + eta[t]
    z[t]   = sigmoid(x[t])
    y[t]   = C z[t] + d + eps[t]

    p = [a, s1, s2, ..., s_{K-1}]

    L is lower-triangular Toeplitz:

        [[a,   0,   0,   0],
         [s1,  a,   0,   0],
         [s2, s1,   a,   0],
         [s3, s2,  s1,   a]]

    i.e. L[i, j] = p[i-j] for i >= j.
    """

    def __init__(
        self,
        p,
        b,
        C,
        x0=None,
        d=None,
        latent_noise=0.0,
        obs_noise=0.1,
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)

        self.p = np.asarray(p, float)
        self.b = np.asarray(b, float)
        self.C = np.asarray(C, float)

        self.N, self.K = self.C.shape

        assert self.p.shape == (self.K,)
        assert self.b.shape == (self.K,)

        self.L = self.make_L(self.p)

        self.x0 = np.zeros(self.K) if x0 is None else np.asarray(x0, float)
        self.d = np.zeros(self.N) if d is None else np.asarray(d, float)

        assert self.x0.shape == (self.K,)
        assert self.d.shape == (self.N,)

        self.latent_noise = self._as_vec(latent_noise, self.K)
        self.obs_noise = self._as_vec(obs_noise, self.N)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _as_vec(x, n):
        x = np.asarray(x, float)
        if x.ndim == 0:
            return np.full(n, float(x))
        assert x.shape == (n,)
        return x

    @staticmethod
    def make_L(p):
        """
        Build lower-triangular Toeplitz L from p.

        p[0] = diagonal self-persistence
        p[1] = first subdiagonal
        p[2] = second subdiagonal
        ...
        """
        p = np.asarray(p, float)
        K = len(p)

        L = np.zeros((K, K))
        for d in range(K):
            L += np.diag(np.full(K - d, p[d]), k=-d)

        return L

    def simulate_dynamics(self, T):
        x = np.zeros((T, self.K))
        z = np.zeros((T, self.K))

        x[0] = self.x0
        z[0] = self.sigmoid(x[0])

        for t in range(T - 1):
            eta = self.rng.normal(0, self.latent_noise, size=self.K)
            x[t + 1] = self.L @ x[t] + self.b + eta
            z[t + 1] = self.sigmoid(x[t + 1])

        return x, z

    def simulate_onsets(self, T, onsets, slopes, latent_noise=0.0):
        onsets = np.asarray(onsets, float)
        slopes = np.asarray(slopes, float)

        assert onsets.shape == (self.K,)
        assert slopes.shape == (self.K,)

        noise = self._as_vec(latent_noise, self.K)

        t = np.arange(T)[:, None]
        x = slopes[None, :] * (t - onsets[None, :])
        x += self.rng.normal(0, noise, size=x.shape)

        z = self.sigmoid(x)
        return x, z

    def simulate_observations(self, z):
        assert z.shape[1] == self.K

        eps = self.rng.normal(0, self.obs_noise, size=(len(z), self.N))
        y = z @ self.C.T + self.d[None, :] + eps

        return y

    def generate(self, T, mode="dynamics", onsets=None, slopes=None, latent_noise=0.0):
        if mode == "dynamics":
            x, z = self.simulate_dynamics(T)
        elif mode == "onsets":
            x, z = self.simulate_onsets(T, onsets, slopes, latent_noise)
        else:
            raise ValueError("mode must be 'dynamics' or 'onsets'")

        y = self.simulate_observations(z)

        return {
            "x": x,
            "z": z,
            "y": y,
            "L": self.L,
            "p": self.p,
            "b": self.b,
            "C": self.C,
            "d": self.d,
            "x0": self.x0,
            "latent_noise": self.latent_noise,
            "obs_noise": self.obs_noise,
        }

    @staticmethod
    def onset_times(x, threshold=0.0):
        onsets = []
        for i in range(x.shape[1]):
            idx = np.where(x[:, i] >= threshold)[0]
            onsets.append(None if len(idx) == 0 else idx[0])
        return onsets

    @staticmethod
    def plot_latents(x, z):
        T, K = z.shape
        t = np.arange(T)

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)

        for i in range(K):
            ax[0].plot(t, x[:, i], label=f"$x_{i+1}$")
            ax[1].plot(t, z[:, i], label=f"$z_{i+1}$")

        ax[0].axhline(0, linestyle="--", linewidth=1)
        ax[1].axhline(0.5, linestyle="--", linewidth=1)

        ax[0].set_title("latent drive")
        ax[1].set_title("activation")
        ax[0].set_xlabel("time")
        ax[1].set_xlabel("time")
        ax[0].set_ylabel("$x$")
        ax[1].set_ylabel("$z$")

        for a in ax:
            a.legend(frameon=False)

        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_observations(y):
        plt.figure(figsize=(7, 4))
        plt.imshow(y.T, aspect="auto", origin="lower")
        plt.xlabel("time")
        plt.ylabel("observed channel")
        plt.colorbar(label="$y$")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def initial_sigmoid_latents(T, K, onset_frac=(0.15, 0.85), slope=0.08):
        onsets = np.linspace(onset_frac[0], onset_frac[1], K) * T
        slopes = slope * np.ones(K)
        t = np.arange(T)[:, None]
        x = slopes[None, :] * (t - onsets[None, :])
        z = SequentialSigmoidalFactorAnalysis.sigmoid(x)
        return x, z, onsets, slopes

    @staticmethod
    def fit_C_d(y, z):
        T = y.shape[0]
        Z = np.column_stack([z, np.ones(T)])

        B, *_ = np.linalg.lstsq(Z, y, rcond=None)
        C = B[:-1].T
        d = B[-1]
        resid = y - Z @ B
        return C, d, resid

    @staticmethod
    def unpack_theta(theta, K):
        p = theta[:K]
        b = theta[K : 2 * K]
        x0 = theta[2 * K : 3 * K]
        return p, b, x0

    @staticmethod
    def pack_theta(p, b, x0):
        return np.concatenate([p, b, x0])

    @classmethod
    def fit(
        cls,
        y,
        K,
        p0=None,
        b0=None,
        x00=None,
        obs_noise_floor = 1e-6,
        max_nfev=2000,
        seed=None,
        verbose=1
    ):
        y = np.asarray(y, float)
        T, N = y.shape
        rng = np.random.default_rng(seed)

        if p0 is None:
            p0 = np.zeros(K)
            p0[0] = 0.2 + 0.05 * rng.standard_normal()
            p0[1:] = 0.0

        if b0 is None:
            b0 = 0.01 * rng.standard_normal(K)

        if x00 is None:
            x00 = -3.0 - 0.5 * np.arange(K)

        theta0 = cls.pack_theta(p0, b0, x00)

        lower = np.concatenate([
            np.array([0.0]),       # a
            -np.ones(K-1),         # s
            -np.inf * np.ones(K),  # b
            -np.inf * np.ones(K),  # x0
        ])

        upper = np.concatenate([
            np.array([0.999]),     # a
            np.ones(K-1),          # s
            np.inf * np.ones(K),   # b
            np.inf * np.ones(K),   # x0
        ])

        def residual_fun(theta):
            p, b, x0 = cls.unpack_theta(theta, K)
            C_dummy = np.zeros((N, K))
            model = cls(
                p=p,
                b=b,
                C=C_dummy,
                x0=x0,
                obs_noise=1.0,
                latent_noise=0.,
                seed = seed,
            )

            x, z = model.simulate_dynamics(T)
            C, d, resid = cls.fit_C_d(y, z)

            return resid.ravel()

        result = least_squares(
            residual_fun,
            theta0,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            verbose=verbose,
        )

        p_hat, b_hat, x0_hat = cls.unpack_theta(result.x, K)

        C_dummy = np.zeros((N, K))
        tmp = cls(
            p=p_hat,
            b=b_hat,
            C=C_dummy,
            x0=x0_hat,
            obs_noise=1.0,
            latent_noise=0.,
            seed=seed,
        )

        x_hat, z_hat = tmp.simulate_dynamics(T)
        C_hat, d_hat, resid = cls.fit_C_d(y, z_hat)

        obs_noise_hat = np.sqrt(np.mean(resid**2, axis=0) + obs_noise_floor)

        fitted = cls(
            p=p_hat,
            b=b_hat,
            C=C_hat,
            x0=x0_hat,
            d=d_hat,
            obs_noise=obs_noise_hat,
            latent_noise=0.0,
            seed=seed,
        )

        return {
            "model": fitted,
            "x": x_hat,
            "z": z_hat,
            "yhat": z_hat @ C_hat.T + d_hat[None, :],
            "residuals": resid,
            "obs_noise": obs_noise_hat,
            "result": result,
        }
