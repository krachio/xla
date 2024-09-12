import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

def fn(x, y, z):
    return jnp.dot(x, y) / z

if __name__ == '__main__':
    x = jnp.array([1, 2, 3, 3], dtype=jnp.float32)
    y = jnp.array([4, 5, 6, 3], dtype=jnp.float32)
    z = jnp.array([4, 5, 6, 3], dtype=jnp.float32)

    foo_compiled = jax.jit(fn).lower(x, y, z).compile()
    with open('foo.hlo', 'w') as f:
        f.write(foo_compiled.as_text())
    
    