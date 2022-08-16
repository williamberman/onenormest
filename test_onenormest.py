from onenormest import onenormest, _onenormest
import jax
import jax.numpy as jnp
import scipy
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose

# TESTS:

# Equivalency with nonest
# Equivalency with scipy est
# The aggregate statistic like tests from scipy
# {CPU, GPU} Performance comparison tests

# TODO does jax have a special representation of a sparse matrix

class TestOnenormest:
    def test_onenormest_table_3_t_2(self):
        key = jax.random.PRNGKey(0)
        t = 2
        n = 100
        itmax = 5
        nsamples = 5000
        observed = []
        expected = []
        nmult_list = []
        nresample_list = []
        for key in jax.random.split(key, nsamples):
            key, subkey = jax.random.split(key)
            A = jnp.linalg.inv(jax.random.normal(subkey, (n, n)))
            est, v, w, nmults, nresamples = _onenormest(key, A, t, itmax)
            observed.append(est)
            expected.append(jnp.linalg.norm(A, ord=1))
            nmult_list.append(nmults)
            nresample_list.append(nresamples)
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        relative_errors = np.abs(observed - expected) / expected

        # check the mean underestimation ratio
        underestimation_ratio = observed / expected
        assert_(0.99 < np.mean(underestimation_ratio) < 1.0)

        # check the max and mean required column resamples
        assert_equal(np.max(nresample_list), 2)
        assert_(0.05 < np.mean(nresample_list) < 0.2)

        # check the proportion of norms computed exactly correctly
        nexact = np.count_nonzero(relative_errors < 1e-14)
        proportion_exact = nexact / float(nsamples)
        assert_(0.9 < proportion_exact < 0.95)

        # check the average number of matrix*vector multiplications
        assert_(3.5 < np.mean(nmult_list) < 4.5)

    def test_onenormest_table_4_t_7(self):
        key = jax.random.PRNGKey(0)
        t = 7
        n = 100
        itmax = 5
        nsamples = 5000
        observed = []
        expected = []
        nmult_list = []
        nresample_list = []
        for key in jax.random.split(key, nsamples):
            A = np.random.randint(-1, 2, size=(n, n))
            est, v, w, nmults, nresamples = _onenormest(key, A, t, itmax)
            observed.append(est)
            expected.append(jnp.linalg.norm(A, 1))
            nmult_list.append(nmults)
            nresample_list.append(nresamples)
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        relative_errors = np.abs(observed - expected) / expected

        # check the mean underestimation ratio
        underestimation_ratio = observed / expected
        assert_(0.90 < np.mean(underestimation_ratio) < 0.99)

        # check the required column resamples
        assert_equal(np.max(nresample_list), 0)

        # check the proportion of norms computed exactly correctly
        nexact = np.count_nonzero(relative_errors < 1e-14)
        proportion_exact = nexact / float(nsamples)
        assert_(0.15 < proportion_exact < 0.25)

        # check the average number of matrix*vector multiplications
        assert_(3.5 < np.mean(nmult_list) < 4.5)

    def test_onenormest_table_5_t_1(self):
        key = jax.random.PRNGKey(0)
        t = 1
        n = 100
        itmax = 5
        alpha = 1 - 1e-6
        A = -scipy.linalg.inv(np.identity(n) + alpha*np.eye(n, k=1))
        first_col = np.array([1] + [0]*(n-1))
        first_row = np.array([(-alpha)**i for i in range(n)])
        B = -scipy.linalg.toeplitz(first_col, first_row)
        assert_allclose(A, B)
        B = jnp.array(B)
        est, v, w, nmults, nresamples = _onenormest(key, B, t, itmax)
        exact_value = jnp.linalg.norm(B, 1)
        underest_ratio = est / exact_value
        assert_allclose(underest_ratio, 0.05, rtol=1e-4)
        assert_equal(int(nmults), 11)
        assert_equal(int(nresamples), 0)
