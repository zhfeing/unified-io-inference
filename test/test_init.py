import jax
import jax.numpy as jnp
import jax.nn as nn

init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", -1, -2)
x = init(jax.random.PRNGKey(0), (128, 256))
print(x.std())
