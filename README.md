## SoftDTW in JAX
Soft-DTW implementation in JAX with custom gradient

## Why implement SoftDTW again?

I find the implements using `jax.lax.scan` very interesting (see [this](https://github.com/khdlr/softdtw_jax/blob/main/softdtw_jax/softdtw_jax.py)). However, all available implementations do not actually follow the Algorithm 2 in [Soft-DTW: a Differentiable Loss Function for Time-Series](https://arxiv.org/pdf/1703.01541.pdf).

This small repository provides the implemetation of custom gradient for Soft-DTW in JAX. It is implemented using

 - `jax.custom_vjp` (see more in [JAX docs](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html))
 - Algorithm 2 in the Soft-DTW paper is done with `jax.lax.scan`


See the notebook `barycenter.ipynb` for a demomstration and a small performance comparison.