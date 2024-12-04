import jax
import jax.numpy as jnp
import equinox as eqx

class VectorQuantizer(eqx.Module):
    K: int = eqx.static_field()
    D: int = eqx.static_field()
    codebook: jax.Array

    codebook_avg: jax.Array
    cluster_size: jax.Array

    decay: float = eqx.static_field()
    eps: float = eqx.static_field()

    def __init__(
        self,
        num_vecs: int = 1024,
        num_dims: int = 512,
        decay: float = 0.99,
        eps: float = 1e-5,
        key=None,
    ):
        self.K = num_vecs
        self.D = num_dims

        self.decay = decay
        self.eps = eps

        # Init a matrix of vectors that will move with time
        self.codebook = jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="uniform"
        )(key, (num_dims, num_vecs))
        self.codebook_avg = jnp.copy(self.codebook)
        self.cluster_size = jnp.zeros(num_vecs)

    def __call__(self, x):
        # x has N vectors of the codebook dimension. We calculate the nearest neighbors and output those instead

        x = jnp.permute_dims(x, (1, 0))
        flatten = jax.numpy.reshape(x, (-1, self.D))
        a_squared = jnp.sum(flatten**2, axis=-1, keepdims=True)
        b_squared = jnp.sum(self.codebook**2, axis=0, keepdims=True)
        distance = a_squared + b_squared - 2 * jnp.matmul(flatten, self.codebook)

        codebook_indices = jnp.argmin(distance, axis=-1)
        z_q = self.codebook.T[codebook_indices]

        # Straight-through estimator
        z_q = flatten + jax.lax.stop_gradient(z_q - flatten)

        z_q = jnp.permute_dims(z_q, (1, 0))
        return z_q, self.codebook_updates(flatten, codebook_indices)

    def codebook_updates(self, flatten, codebook_indices):
        # Calculate the usage of various codes.
        codebook_onehot = jax.nn.one_hot(codebook_indices.T, self.K)
        codebook_onehot_sum = jnp.sum(codebook_onehot, axis=0)
        codebook_sum = jnp.dot(flatten.T, codebook_onehot)
        # We've just weighed the codebook vectors.

        # Basically count on average how many codes we're using
        new_cluster_size = (
            self.decay * self.cluster_size + (1 - self.decay) * codebook_onehot_sum
        )

        # Where is the average embedding at ?
        new_codebook_avg = (
            self.decay * self.codebook_avg.T + (1 - self.decay) * codebook_sum.T
        )

        n = jnp.sum(new_cluster_size)  # Over the total embeddings used
        new_cluster_size = (new_cluster_size + self.eps) / (n + self.K * self.eps) * n
        new_codebook = self.codebook_avg.T / new_cluster_size[:, None]

        updates = (new_cluster_size, new_codebook_avg.T, new_codebook.T)

        return updates, codebook_indices

    def embed_code(self, embed_id):
        return jax.nn.embedding(embed_id, self.codebook.T)
