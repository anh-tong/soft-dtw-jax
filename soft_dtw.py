from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


@jax.custom_vjp
def soft_dtw(matrix: Float[Array, "n m"], gamma: float) -> float:
    return soft_dtw_fwd(matrix, gamma)[0]


def soft_dtw_fwd(
    matrix: Float[Array, "n m"], gamma: float
) -> Tuple[float, Tuple[Array, Array, float]]:
    """Forward pass for Soft-DTW"""
    R = _soft_dtw(matrix=matrix, gamma=gamma, return_last=False)
    res = (matrix, R, gamma)
    return R[-1, -1], res


def soft_dtw_bwd(res: Tuple[Array, Array, float], g: Array) -> Tuple[Array, None]:
    """
    Backward pass for Soft-DTW

    See Algorithm 2 of M. Cuturi, M. Blondel. 2017 (https://arxiv.org/pdf/1703.01541.pdf)

    Args:
        res: residuals that passed from forward pass. Contain
        g: gradient
    Return:
        Gradient w.r.t. input matrix and `None` indicates that there is no gradient for `gamma`
    """

    matrix, R, gamma = res

    n, m = matrix.shape

    # initialize padding boundary values
    D = jnp.pad(matrix, pad_width=((0, 1), (0, 1)), constant_values=0.0)
    R = jnp.pad(R, pad_width=((0, 1), (0, 1)), constant_values=-jnp.inf)
    R = R.at[-1, -1].set(R[-2, -2])

    # compute auxiliary matrix for a, b, c in Alg. 2
    # for the case of a: a[i,j] = R[i + 1, j] - R[i, j] - D[i + 1, j]
    A = R[1:, :] - R[:-1, :] - D[1:, :]
    A = jnp.where(R[1:, :] == -jnp.inf, -jnp.inf, A)
    A = jnp.where(R[:-1, :] == -jnp.inf, -jnp.inf, A)
    A = A[:, :-1]

    # for the case of b: b[i,j] = R[i, j + 1] - R[i, j] - D[i, j + 1]
    B = R[:, 1:] - R[:, :-1] - D[:, 1:]
    B = jnp.where(R[:, 1:] == -jnp.inf, -jnp.inf, B)
    B = jnp.where(R[:, :-1] == -jnp.inf, -jnp.inf, B)
    B = B[:-1, :]

    # for the case of c: c[i,j] = R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]
    C = R[1:, 1:] - R[:-1, :-1] - D[1:, 1:]
    C = jnp.where(R[1:, 1:] == -jnp.inf, -jnp.inf, C)
    C = jnp.where(R[:1, :1] == -jnp.inf, -jnp.inf, C)

    A = jnp.exp(A / gamma)
    B = jnp.exp(B / gamma)
    C = jnp.exp(C / gamma)

    # we will iterate over anti-diagonal, so we anti-diagonalize a, b, c
    A_ad = anti_diagonalize(A, fill_value=0.0)
    B_ad = anti_diagonalize(B, fill_value=0.0)
    C_ad = anti_diagonalize(C, fill_value=0.0)

    # make `init` for `jax.lax.scan`
    two_ago = jnp.full(n + 1, fill_value=0.0).at[-1].set(1.0)
    one_ago = jnp.full(n + 1, fill_value=0.0)
    init = (two_ago, one_ago)

    # body function for scan
    def scan_step(carry, index):
        two_ago, one_ago = carry
        diagonal, right, down = two_ago[1:], one_ago[1:], one_ago[:-1]

        def current(input):
            return jax.lax.dynamic_index_in_dim(input, index, 0, keepdims=False)

        a = current(A_ad)
        b = current(B_ad)
        c = current(C_ad)

        result = right * a + down * b + diagonal * c
        result = jnp.pad(result, (0, 1), constant_values=0.0)

        return (one_ago, result), result

    # # for testing/debug
    # carry = init
    # for index in range(n+m-2, -1, -1):
    #     carry, current = scan_step(carry, index)
    # # end debug

    _, result = jax.lax.scan(scan_step, init=init, xs=jnp.arange(m + n - 2, -1, -1))

    # ignored last column containing padded zeros
    result = result[:, :-1]
    # as scan run in reversed order, we need to reverse back
    result = result[::-1, :]

    # retrieve the matrix E
    mask = np.tri(n + m - 1, n, k=0, dtype=bool)
    mask = mask & mask[::-1, ::-1]
    result = result.T[mask.T].T
    result = result.reshape((n, m))

    return (result * g, None)


soft_dtw.defvjp(soft_dtw_fwd, soft_dtw_bwd)


def _soft_dtw(matrix: Float[Array, "n m"], gamma: float, return_last=True):

    transpose = False

    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
        transpose = True

    n, m = matrix.shape

    model_matrix, mask = anti_diagonalize(matrix, return_mask=True)

    init = (
        _pad_inf(model_matrix[0], 1, 0),
        _pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0),
    )

    def scan_step(carry, current_antidiagonal):
        two_ago, one_ago = carry

        diagonal = two_ago[:-1]
        right = one_ago[:-1]
        down = one_ago[1:]
        best = softmin(jnp.stack([diagonal, right, down], axis=-1), gamma)

        next_row = best + current_antidiagonal
        next_row = _pad_inf(next_row, 1, 0)

        return (one_ago, next_row), next_row

    _, result = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)

    if return_last:
        return result[-1]

    two_ago, one_ago = init
    result = jnp.concatenate([two_ago[None, :], one_ago[None, :], result], axis=0)
    result = result[:, 1:]
    result = result.T[mask.T].T
    result = result.reshape((n, m))

    if transpose:
        result = result.T

    return result


def _pad_inf(inp: Array, before: int, after: int):
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def _distance(x, y):
    return jnp.sum(jnp.square(x - y))


def anti_diagonalize(A, fill_value=jnp.inf, return_mask=False):
    """Construct a matrix"""
    n, m = A.shape
    mask = np.tri(n + m - 1, n, k=0, dtype=bool)
    mask = mask & mask[::-1, ::-1]

    ret = jnp.full((n, n + m - 1), fill_value=fill_value)
    ret = ret.at[mask.T].set(A.ravel()).T
    if return_mask:
        return ret, mask
    else:
        return ret


def softmin_raw(array: Array, gamma: float):
    """
    Softmin fuction
    """
    return -gamma * jax.nn.logsumexp(array / -gamma, axis=-1)


softmin = jax.custom_vjp(softmin_raw)


def sofmin_fwd(array: Array, gamma: float):
    """Forward pass of softmin"""
    return softmin(array, gamma), (array / -gamma,)


def softmin_bwd(res, g):
    """Backward pass of softmin"""
    (scaled_array,) = res
    grad = jnp.where(
        jnp.isinf(scaled_array),
        jnp.zeros(scaled_array.shape),
        jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1),
    )
    return (grad, None)


softmin.defvjp(sofmin_fwd, softmin_bwd)
