import jax
from jax import lax, vmap, jit
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple

def onenormest(key, A, t=2, itmax=5, compute_v=False, compute_w=False):
    """
    https://github.com/gnu-octave/octave/blob/eff42b5a8c617f62a0ee1ddc2b70c246bbf32cb3/scripts/linear-algebra/normest1.m
    https://github.com/scipy/scipy/blob/59dac8a9fa9ea856f4a50521d295a3497d648faa/scipy/sparse/linalg/_onenormest.py
    http://eprints.maths.manchester.ac.uk/321/1/35608.pdf
    """
    # TODO add type checking
    # TODO add shape checking
    # TODO add delegate to non estimate method
    # TODO add checks for arguments that ask to compute certain things
    est, v, w, nmults, nresamples = _onenormest(key, A, t, itmax)

    # Report the norm estimate along with some certificates of the estimate.
    if compute_v or compute_w:
        result = (est,)
        if compute_v:
            result += (v,)
        if compute_w:
            result += (w,)
        return result
    else:
        return est

class _OnenormestLoopState(NamedTuple):
    # Number of loop iterations (starts at 1)
    k: int
    
    # Flag to indicate break out of loop. We can't directly break out of the
    # loop because jax requires us to return a value from the loop body that
    # is then checked in the cond fun.
    break_flag: bool
        
    # "ind_hist = [ ] % Integer vector recording indices of used unit vectors e_j"
    #
    # row vector of size n+1 (should be size n but additional value is to allow for sentinel values in ind). 
    # ind_hist[j] == 0 if e_j has not been used
    # ind_hist[j] == 1 if e_j has been used  
    #
    # in the scipy implementation and Higham, this is a growable array that stores
    # indices of the unit vectors. In the octave implementation, this is a fixed
    # sized array that writes 1 into index j when e_j is used.
    # We use the fixed sized array so we can jit compile the function.
    ind_hist: jnp.ndarray
        
    # The previous estimate of the one norm
    est_old: float
        
    # The current estimate of the one norm
    est: float
    
    # ind is a row vector of shape (t,)
    #
    # ind tracks which elementary vectors are stored in the column vectors of
    # x. i.e. e_{ind[j]} = X[:, j]
    # 
    # ind is shape (n,) in Higham but only the first t values
    # are read out of it. The first t values are read for writing to ind_hist.
    # It is also read out of with column indices of Y and Y is shape (n, t).
    #
    # because we only test elementary vectors a single time, it is not guaranteed
    # we'll have t elementary vectors to test on each loop. We handle this by filling
    # non-used elements of ind with a sentinel value "n". This sentinel value requires
    # extending ind_hist's size by one to n+1
    ind: jnp.ndarray
        
    # TODO
    S: jnp.ndarray
        
    # TODO
    X: jnp.ndarray
        
    nresamples: int
    
    nmults: int
        
    # The column of Y that produces the best estimate for the 1 norm of A
    # as a row vector (n,)
    w: jnp.ndarray
        
    # v == The unit vector e_{ind_best}
    ind_best: int
        
    # key for resampling
    key: jax.random.PRNGKey
        

@partial(jit, static_argnums=[2])
def _onenormest(key, A, t, itmax):
    AT = A.T
    n = A.shape[0]
    key, subkey = jax.random.split(key)
    X, nresamples = _onenormest_build_starting_matrix(subkey, n, t)
    
    nmults = 0
    
    # size is set to n+1 so ind can write to its sentintel empty value of n 
    ind_hist = jnp.zeros((n+1,))
    
    est_old = float(0)
    
    est = float(0)
    
    ind = jnp.zeros((t,), dtype=int)
    
    S = jnp.zeros((n, t))
    
    w = jnp.zeros((n,))
    
    ind_best = 0
    
    k = 1
    
    init_loop_state = _OnenormestLoopState(
        k=k,
        break_flag=False,
        ind_hist=ind_hist,   
        est_old=est_old,
        est=est,
        ind=ind,
        S=S,
        X=X,
        nmults=nmults,
        nresamples=nresamples,
        w=w,
        ind_best=ind_best,
        key=key
    )
    
    # Continue while the break flag has not been set
    def cond_fun(loop_state: _OnenormestLoopState) -> bool:
        return jnp.logical_not(loop_state.break_flag)
    
    def body_fun(loop_state: _OnenormestLoopState) -> _OnenormestLoopState:
        k = loop_state.k
        ind_hist = loop_state.ind_hist
        est_old = loop_state.est_old
        est = loop_state.est
        ind = loop_state.ind
        S = loop_state.S
        X = loop_state.X
        nmults = loop_state.nmults
        nresamples = loop_state.nresamples
        w = loop_state.w
        ind_best = loop_state.ind_best
        key = loop_state.key

        Y = A @ X
        nmults += 1
        
        # "est = max{ ||Y (: , j)||_1 : j = 1:t }"
        # The estimate is the max 1 norm of the column vectors of Y
        potential_estimates = jnp.sum(jnp.abs(Y), axis=0)
        # The column in Y of the best esimate
        best_j = jnp.argmax(potential_estimates)
        est = potential_estimates[best_j]
        
        # if est > est_old or k = 2
        #     ind_best = ind_j where est = |Y (: , j)|_1
        #     w = Y(: , ind best)
        # end
        w = lax.cond(
            jnp.logical_or(est > est_old, k == 2), 
            lambda: Y[:, best_j], 
            lambda: w
        )
        # TODO(will)
        # Note the additional k >= 2 in the condition. This comes from scipy. 
        # Not sure why it's there
        ind_best = lax.cond(
            jnp.logical_and(jnp.logical_or(est > est_old, k == 2), k >= 2), 
            lambda: ind[best_j], 
            lambda: ind_best
        )
        
        est = lax.cond(
            jnp.logical_and(k >= 2, est <= est_old),
            lambda: est_old,
            lambda: est,
        )
        
        def cont1():
            est_old = est
            S_old = S
            
            def cont2():
                S = _onenormest_sign_round_up(Y)
                
                def cont3():
                    def resample_S_helper():
                        key_, subkey = jax.random.split(key)
                        S_, inc_nresamples = _onenormest_resample_S(subkey, S, S_old, n, t)
                        return S_, inc_nresamples, key_
                    
                    S_, inc_nresamples, key_ = lax.cond(t > 1, resample_S_helper, lambda: (S, 0, key))
                    nresamples_ = nresamples + inc_nresamples
                    
                    Z = AT @ S_
                    nmults_ = nmults + 1
                    
                    # "h_i = |Z(i, :)|∞, ind_i = i, i = 1:n"
                    # h is the max norms of the row vectors of Z
                    # ind will end up being the indices of h in sorted descending order
                    # we can just set ind after we argsort h
                    h = jnp.linalg.norm(Z, ord=jnp.inf, axis=1)
                    
                    def cont4():
                        # "Sort h so that h1 ≥···≥ hn and re-order ind correspondingly"
                        # We don't actually need to sort h because we only need its indices
                        # ind_tmp is ind before it's sorted and capped to size t
                        ind_tmp = jnp.argsort(h)[::-1]
                        
                        def cont5():
                            ind = lax.cond(
                                t > 1, 
                                lambda: _onenormest_ind_not_in_ind_hist(ind_tmp, ind_hist, n, t),
                                # hist is empty so we don't have to check hist, but we
                                # do have to take the first t elements of ind_tmp
                                lambda: ind_tmp[:t]
                            )
                            
                            # "X(: , j) = e_{ind[j]} , j = 1:t"
                            #
                            # Set the first t column vectors of X to the elementary
                            # vectors with the dimension specified in ind
                            #
                            # Since X has t columns, we can just set a new X
                            # instead of writing into the old one. 
                            #
                            # Note that it's ok for ind to contain the sentinel n
                            # because `elementary_vector`
                            # will return a zero vector instead of an elementary vector.
                            # This will cause X to have zero vectors which is ok because the
                            # zero vectors will cause norm estimations of 0 which are always
                            # a correct underestimate of the one norm.
                            X = _onenormest_elementary_vectors_with_lookup(n, jnp.arange(t), ind)
                            
                            # "ind_hist = [ind_hist ind(1:t)]"
                            # Note that we cannot concatenate to ind_hist because our ind_hist is
                            # a fixed size where we set flags at indices for used unit vectors
                            ind_hist_ = ind_hist.at[ind].set(1)
                            
                            return _OnenormestLoopState(break_flag=False, k=k+1, ind_hist=ind_hist_, est_old=est_old, est=est, ind=ind, S=S_, X=X, nmults=nmults_, nresamples=nresamples_, w=w, ind_best=ind_best, key=key_)
                        
                        
                        return lax.cond(
                            jnp.logical_and(t > 1, _onenormest_check_ind_in_ind_hist(ind_tmp, ind_hist, t)),
                            lambda: _OnenormestLoopState(break_flag=True, k=k, ind_hist=ind_hist, est_old=est_old, est=est, ind=ind, S=S_, X=X, nmults=nmults_, nresamples=nresamples_, w=w, ind_best=ind_best, key=key_),
                            cont5
                        )

                    
                    # "if k ≥ 2 and max(h_i)) = h_{ind_best}
                    #     goto (6) (break)
                    # end"
                    return lax.cond(
                        jnp.logical_and(k >= 2, jnp.max(h) == h[ind_best]),
                        lambda: _OnenormestLoopState(break_flag=True, k=k, ind_hist=ind_hist, est_old=est_old, est=est, ind=ind, S=S_, X=X, nmults=nmults_, nresamples=nresamples_, w=w, ind_best=ind_best, key=key_),
                        cont4
                    )
                                        
                # If every column of S is parallel to a column of S_old, 
                #     goto (6) (break)
                # end
                return lax.cond(
                    _onenormest_every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old, n),
                    lambda: _OnenormestLoopState(break_flag=True, k=k, ind_hist=ind_hist, est_old=est_old, est=est, ind=ind, S=S, X=X, nmults=nmults, nresamples=nresamples, w=w, ind_best=ind_best, key=key),
                    cont3
                )
            
            return lax.cond(
                k > itmax,
                lambda: _OnenormestLoopState(break_flag=True, k=k, ind_hist=ind_hist, est_old=est_old, est=est, ind=ind, S=S, X=X, nmults=nmults, nresamples=nresamples, w=w, ind_best=ind_best, key=key),
                cont2
            )
        
        return lax.cond(
            jnp.logical_and(k >= 2, est <= est_old),
            lambda: _OnenormestLoopState(break_flag=True, k=k, ind_hist=ind_hist, est_old=est_old, est=est, ind=ind, S=S, X=X, nmults=nmults, nresamples=nresamples, w=w, ind_best=ind_best, key=key),
            cont1
        )
                
    
    final_loop_state = lax.while_loop(
        cond_fun,
        body_fun,
        init_loop_state
    )
    
    est = final_loop_state.est
    ind_best = final_loop_state.ind_best
    w = final_loop_state.w
    nmults = final_loop_state.nmults
    nresamples = final_loop_state.nresamples
    
    v = _onenormest_elementary_vector(n, ind_best)
    return est, v, w, nmults, nresamples


def _onenormest_build_starting_matrix(key, n, t):
    """
    "Choose starting matrix X ∈ Rn×t with columns of unit 1-norm."
    
    "We now explain our choice of starting matrix. We take the first column of X to
    be the vector of 1s, which is the starting vector used in Algorithm 2.1. This has the
    advantage that for a matrix with nonnegative elements the algorithm converges with
    an exact estimate on the second iteration, and such matrices arise in applications,
    for example as a stochastic matrix or as the inverse of an M-matrix. The remaining
    columns are chosen as rand{−1, 1}, with a check for and correction of parallel columns,
    exactly as for S in the body of the algorithm."
    """
    
    # Initializing the matrix to all zeroes is important for the parallel checks
    # that will be done during resampling
    X = jnp.zeros((n, t))
    
    # "We take the first column of X to be the vector of 1s"
    X = X.at[:,0].set(jnp.ones((n,), dtype=float))
    
    # NOTE(will) - We could alternatively sample all columns of the matrix at once up front.
    # We could still do a parallel check by dot producting the individual column against
    # the matrix and checking there is only a single column (itself) that it is parallel
    # with. Since we would still have to loop over the columns of the matrix to do the check, 
    # I'm assuming that sampling one at a time in the loop is more efficient.
    
    # "The remaining columns are chosen as rand{−1, 1}""
    # 1 key for re-sampling parallel columns
    # t-1 subkeys for sampling the remaining column vectors
    
    # "with a check for and correction of parallel columns, exactly as for S in the body of the algorithm"
    def sample_body_fun(i, args):
        key, X, nresamples = args
        
        def resample_while_loop_body_fun(args):
            key, v, nresamples = args
            
            key, subkey = jax.random.split(key)
            v = _onenormest_sample_row(subkey, n)
            nresamples += 1
            
            return key, v, nresamples
            
        key, subkey = jax.random.split(key)
        starting_vector = _onenormest_sample_row(subkey, n)
        
        key, orthogonal_v, nresamples = lax.while_loop(
            lambda args: _onenormest_X_needs_resampling(n, X, args[1]), # args[1] is current vector
            resample_while_loop_body_fun,
            (key, starting_vector, nresamples)
        )
        
        X = X.at[:,i].set(orthogonal_v)
        
        return key, X, nresamples
    
    
    (_key, X, nresamples) = lax.fori_loop(
        # Start at column index 1. column i is checked if parallel against column's [0, i)
        # so there are no prior columns to check column 0 against.
        1,
        t,
        sample_body_fun,
        # key for resampling, X, nresamples
        (key, X, 0)
    )
    
    # "columns of unit 1-norm"
    X = X / float(n)
    
    return X, nresamples

def _onenormest_sample_row(key, n):
    """
    random row vector of size n of {-1, 1}s
    """
    return jax.random.randint(key, minval=0, maxval=2, shape=(n,))*2 - 1

# Sample column vectors instead of single row vector
_onenormest_sample_col_vectors = vmap(_onenormest_sample_row, in_axes=(0, None), out_axes=1)

def _onenormest_X_needs_resampling(n, X, v):
    # v needs resampling if it is parallel to any column of X.
    #
    # All not sampled columns of X are zero and will not be parallel
    # with v
    #
    # v is a row vector so we take a vector-matrix product to dot product
    # v with all columns of X
    #
    # SciPy implementation:
    # "Columns are considered parallel when they are equal or negative.
    # Entries are required to be in {-1, 1},
    # which guarantees that the magnitudes of the vectors are identical."
    return jnp.any(v @ X == n)


# NOTE(will) - taken from scipy -- How to attribute?
def _onenormest_sign_round_up(X):
    """
    This should do the right thing for both real and complex matrices.
    From Higham and Tisseur:
    "Everything in this section remains valid for complex matrices
    provided that sign(A) is redefined as the matrix (aij / |aij|)
    (and sign(0) = 1) transposes are replaced by conjugate transposes."
    """
    X = jnp.where(X != 0, X, jnp.ones_like(X))
    X = X / jnp.abs(X)
    return X


def _onenormest_every_col_of_X_is_parallel_to_a_col_of_Y(X, Y, n):
    # if any of (Y.T @ X)[:, i] == n, then X[:, i] is parallel to a col in Y
    
    # Z[:, i] is the dot product of X[:, i] with Y's col vectors
    Z = Y.T @ X
    
    # Using the same `== n` check as in _onenormest_X_needs_resampling,
    # if any of Z[:, i] == n, then X[:, i] is parallel to at least one
    # of Y's column vectors
    X_parallel_column_vectors = jnp.any(Z == n, axis=0)
    
    every_col_is_parallel = jnp.all(X_parallel_column_vectors)
    
    return every_col_is_parallel


def _onenormest_resample_S(key, S, S_old, n, t) -> Tuple[jnp.ndarray, int]:
    """
    "Ensure that no column of S is parallel to another column of S
    or to a column of S_old by replacing columns of S by rand{−1, 1}."
    """
    def sample_body_fun(i, args):
        key, S, nresamples = args
        
        def resample_while_loop_cond_fun(args):
            _key, S, _nresamples = args
            v = S[:, i]
            return _onenormest_S_needs_resampling(v, S, S_old, n)
        
        def resample_while_loop_body_fun(args):
            key, S, nresamples = args
            
            key, subkey = jax.random.split(key)
            v = _onenormest_sample_row(subkey, n)
            S = S.at[:, i].set(v)
            nresamples += 1
            
            return key, S, nresamples
        
        key, S, nresamples = lax.while_loop(
            resample_while_loop_cond_fun,
            resample_while_loop_body_fun,
            (key, S, nresamples)
        )
        
        return key, S, nresamples
        
        
    _key, S, nresamples = lax.fori_loop(
        0,
        t,
        sample_body_fun,
        # key for resampling, S, nresamples
        (key, S, 0)
    )
    
    return S, nresamples

def _onenormest_S_needs_resampling(v, S, S_old, n):
    """
    v is a column vector of S
    v must be parallel to no columns in S or S_old
    v will be parallel to itself which is in S, so we ensure there
    is a single parallel vector in S
    
    we use the same `== n` check as in _onenormest_X_needs_resampling
    """
    S_parallel_columns = (S.T @ v) == n
    num_S_parallel = jnp.count_nonzero(S_parallel_columns)
    # num_S_parallel can't be zero
    some_S_parallel = num_S_parallel != 1
    
    
    S_old_parallel_columns = (S_old.T @ v) == n
    some_S_old_parallel = jnp.any(S_old_parallel_columns)
    
    return jnp.logical_or(some_S_parallel, some_S_old_parallel)


# "If ind(1:t) is contained in ind_hist
#     goto (6) (break)
# end"
def _onenormest_check_ind_in_ind_hist(ind_tmp, ind_hist, t):
    ind_t = ind_tmp[0:t]
    mask = ind_hist[ind_t] == 1
    return jnp.all(mask)


def _onenormest_ind_not_in_ind_hist(ind_tmp, ind_hist, n, t):
    """
    "Replace ind(1:t) by the first t indices in ind(1:n) that are
    not in ind_hist."
    """
    ind = jnp.empty((t,), dtype=int)

    def populate_ind(ind_idx, args):
        ind, ind_tmp_idx = args

        # Find the index in ind_tmp that has not been already used
        ind_tmp_idx = lax.while_loop(
            lambda ind_tmp_idx: jnp.logical_and(ind_tmp_idx < n, ind_hist[ind_tmp[ind_tmp_idx]] == 1),
            lambda ind_tmp_idx: ind_tmp_idx + 1,
            ind_tmp_idx
        )

        value_for_ind, ind_tmp_idx = lax.cond(
            # If all values in ind_tmp have already been used, set the sentinel value n in ind
            ind_tmp_idx == n, 
            lambda: (n, ind_tmp_idx), 
            lambda: (ind_tmp[ind_tmp_idx], ind_tmp_idx + 1)
        )

        ind = ind.at[ind_idx].set(value_for_ind)

        return ind, ind_tmp_idx
        

    ind, _ind_tmp_idx = lax.fori_loop(
        0,
        t,
        populate_ind,
        # ind, ind_tmp_idx
        (ind, 0)
    )
    
    return ind


def _onenormest_elementary_vector_with_lookup(n, i, ind):
    """
    e_{ind[i]}
    """
    return _onenormest_elementary_vector(n, ind[i])


def _onenormest_elementary_vector(n, i):
    """
    e_i
    if i == n, then the zero vector is returned
    """
    v = jnp.zeros(n, dtype=float)
    v = lax.cond(i == n, lambda: v, lambda: v.at[i].set(1))
    return v


# Batches i to return a matrix (n, len(i)) with column elementary vectors
_onenormest_elementary_vectors_with_lookup = vmap(_onenormest_elementary_vector_with_lookup, in_axes=(None, 0, None), out_axes=1)