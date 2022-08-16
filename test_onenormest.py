from onenormest import onenormest, _onenormest
import jax
import jax.numpy as jnp
import scipy
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
import pytest
import random

random.seed(0)

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
            key, subkey = jax.random.split(key, 2)
            A = jax.random.randint(subkey, (n, n), -1, 2)
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

    # TODO(will) - this causes a type error in jax
    def test_onenormest_table_6_t_1(self):
        key = jax.random.PRNGKey(0)
        t = 1
        n = 100
        itmax = 5
        nsamples = 5000
        observed = []
        expected = []
        nmult_list = []
        nresample_list = []
        for key in jax.random.split(key, nsamples):
            key, key1, key2 = jax.random.split(key, 3)
            A_inv = jax.random.normal(key1, (n, n)) + 1j * jax.random.normal(key2, (n, n))
            A = jnp.linalg.inv(A_inv)
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
        underestimation_ratio_mean = np.mean(underestimation_ratio)
        assert_(0.90 < underestimation_ratio_mean < 0.99)

        # check the required column resamples
        max_nresamples = np.max(nresample_list)
        assert_equal(max_nresamples, 0)

        # NOTE(will) - relaxed from scipy
        # check the proportion of norms computed exactly correctly
        nexact = np.count_nonzero(relative_errors < 1e-5)
        proportion_exact = nexact / float(nsamples)
        assert_(0.5 < proportion_exact < 0.8)

        # NOTE(will) - relaxed from scipy
        # check the average number of matrix*vector multiplications
        mean_nmult = np.mean(nmult_list)
        assert_(4 < mean_nmult < 6)

    
    def test_returns(self):
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        A = jnp.linalg.inv(jax.random.normal(subkey, (50, 50)))

        s0 = jnp.linalg.norm(A, 1)
        s1, v = onenormest(key, A, compute_v=True)
        s2, w = onenormest(key, A, compute_w=True)
        s3, v2, w2 = onenormest(key, A, compute_w=True, compute_v=True)

        assert_allclose(s1, s0, rtol=1e-5)
        assert_allclose(np.linalg.norm(A.dot(v), 1), s0*np.linalg.norm(v, 1), rtol=1e-5)
        assert_allclose(A.dot(v), w, rtol=1e-5)

        assert((s1 == s2).all())
        assert((s2 == s3).all())
        assert((v == v2).all())
        assert((w == w2).all())

class TestBenchmark:
    pass

test_cases = [
    # JAX cpu
    {
        't': 2,
        'n': 100,
        'itmax': 5,
        'device': 'cpu',
        'lib': 'jax',
    },
    {
        't': 7,
        'n': 100,
        'itmax': 5,
        'device': 'cpu',
        'lib': 'jax',
    },
    {
        't': 16,
        'n': 2048,
        'itmax': 8,
        'device': 'cpu',
        'lib': 'jax',
    },
    {
        't': 128,
        'n': 2048,
        'itmax': 32,
        'device': 'cpu',
        'lib': 'jax',
    },
    {
        't': 128,
        'n': 4096,
        'itmax': 32,
        'device': 'cpu',
        'lib': 'jax',
    },

    {
        't': 2,
        'n': 100,
        'itmax': 5,
        'device': 'gpu',
        'lib': 'jax',
    },
    {
        't': 7,
        'n': 100,
        'itmax': 5,
        'device': 'gpu',
        'lib': 'jax',
    },
    {
        't': 16,
        'n': 2048,
        'itmax': 8,
        'device': 'gpu',
        'lib': 'jax',
    },
    {
        't': 128,
        'n': 2048,
        'itmax': 32,
        'device': 'gpu',
        'lib': 'jax',
    },
    {
        't': 128,
        'n': 4096,
        'itmax': 32,
        'device': 'gpu',
        'lib': 'jax',
    },

    # scipy
    {
        't': 2,
        'n': 100,
        'itmax': 5,
        'lib': 'scipy',
    },
    {
        't': 7,
        'n': 100,
        'itmax': 5,
        'lib': 'scipy',
    },
    {
        't': 16,
        'n': 2048,
        'itmax': 8,
        'lib': 'scipy',
    },
    {
        't': 128,
        'n': 2048,
        'itmax': 32,
        'lib': 'scipy',
    },
    {
        't': 128,
        'n': 4096,
        'itmax': 32,
        'lib': 'scipy',
    },
]

# do not set device when running on scipy
def make_test_case(*args, t, n, itmax, lib, device=None):
    name = f"test_onenormest_n_{n}_t_{t}_itmax_{itmax}_{lib}"

    if device:
        name = f"{name}_{device}"

    def args():
        key = jax.random.PRNGKey(random.randint(0, 2*n))
        key, subkey = jax.random.split(key)
        A = jax.random.normal(subkey, (n, n))
        return key, A
        
    if lib == 'jax':
        if device == 'gpu' and jax.devices()[0].platform != 'gpu':
            @pytest.mark.skip(reason="gpu not available")
            def func(_cls, _benchmark):
                pass
        else:
            def func_to_benchmark():
                key, A = args()
                est, *_ = _onenormest(key, A, t, itmax)
                est.block_until_ready()

            def func(cls, benchmark):
                with jax.default_device(jax.devices(device)[0]):
                    # Compile the function
                    key, A = args()
                    est, *_ = _onenormest(key, A, t, itmax)
                    est.block_until_ready()

                    benchmark(func_to_benchmark)
    elif lib == 'scipy':
        def func_to_benchmark():
            _key, A = args()
            A = np.array(A)
            scipy.sparse.linalg.onenormest(A)

        def func(cls, benchmark):
            benchmark(func_to_benchmark)
    else:
        assert(False)

    return name, func

for test_case in test_cases:
    name, func = make_test_case(**test_case)
    setattr(TestBenchmark, name, func)

        