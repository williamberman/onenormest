from onenormest import onenormest, _onenormest
import jax
import jax.numpy as jnp
import scipy
import numpy as np
from numpy.testing import assert_, assert_equal

# TESTS:

# Equivalency with nonest
# Equivalency with scipy est
# The aggregate statistic like tests from scipy
# {CPU, GPU} Performance comparison tests

# TODO does jax have a special representation of a sparse matrix

# def test_nonest_equivalence():
#     key = jax.random.PRNGKey(0)
#     n = 50

#     num_tests = 10

#     for key in jax.random.split(key, num_tests):
#         key, subkey = jax.random.split(key)

#         A = jax.random.normal(subkey, (n, n))

#         est = onenormest(key, A)
#         scipy_est = scipy.sparse.linalg.onenormest(np.array(A))
#         actual = jnp.linalg.norm(A, ord=1)

#         print('*****')
#         print(est)
#         print(scipy_est)
#         print(actual)
#         print('*****')


class TestOnenormest:

    # @pytest.mark.xslow
    def test_onenormest_table_3_t_2(self):
        t = 2
        n = 100
        itmax = 5
        nsamples = 5000

        key = jax.random.PRNGKey(0)

        observed = []
        expected = []
        nmult_list = []
        nresample_list = []

        for key in jax.random.split(key, nsamples):
            key, subkey = jax.random.split(key)

            # TODO(will) - not sure why they invert the matrix here
            # A = scipy.linalg.inv(np.random.randn(n, n))
            A = jax.random.normal(subkey, (n, n))
            est, v, w, nmults, nresamples = _onenormest(key, A, t, itmax)
            observed.append(est)
            expected.append(jnp.linalg.norm(A, ord=1))
            nmult_list.append(nmults)
            nresample_list.append(nresamples)

        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        relative_errors = np.abs(observed - expected) / expected

        # check the mean underestimation ratio
        underestimation_ratio = np.mean(observed / expected)
        print(underestimation_ratio)
        assert_(0.99 < underestimation_ratio < 1.0)

        # check the max and mean required column resamples
        assert_equal(np.max(nresample_list), 2)
        assert_(0.05 < np.mean(nresample_list) < 0.2)

        # check the proportion of norms computed exactly correctly
        nexact = np.count_nonzero(relative_errors < 1e-14)
        proportion_exact = nexact / float(nsamples)
        assert_(0.9 < proportion_exact < 0.95)

        # check the average number of matrix*vector multiplications
        assert_(3.5 < np.mean(nmult_list) < 4.5)