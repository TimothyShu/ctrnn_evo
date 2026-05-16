from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .genome import Genome, effective_weights


def forward_pass(
    genome: Genome,
    v0: jnp.ndarray,
    input_vec: jnp.ndarray,
    cfg: Config,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Run one world step: K inner CTRNN ticks via Euler integration.

    CTRNN update per tick:
        y        = tanh(v)
        dv/dt    = (-v + W_eff @ y + bias + input_vec) / tau
        v_next   = (v + dt * dv) * active_mask

    Args:
        genome:    single organism genome (unbatched)
        v0:        membrane potentials at start of world step [N_max]
        input_vec: external drive [N_max], nonzero only at input neuron slots
        cfg:       hyperparameter config

    Returns:
        v_final: membrane potentials after K ticks [N_max]
        output:  tanh(v_final[-n_out:])  [n_out]
        c_act:   activation cost normalised by K (scalar)
    """
    W_eff = effective_weights(genome)  # [N_max, N_max] — computed once per world step

    def tick(v: jnp.ndarray, _):
        y      = jnp.tanh(v)
        recur  = W_eff @ y
        dv     = (-v + recur + genome.bias + input_vec) / genome.tau
        v_next = (v + cfg.dt * dv) * genome.active_mask
        c_k    = jnp.sum(jnp.abs(y) * genome.active_mask)
        return v_next, c_k

    v_final, c_per_tick = jax.lax.scan(tick, v0, None, length=cfg.K)
    c_act  = jnp.mean(c_per_tick)          # normalised by K
    output = jnp.tanh(v_final[-cfg.n_out:])
    return v_final, output, c_act


# Vectorised over a population — call as batch_forward(pop_genome, v0s, inputs, cfg)
batch_forward = jax.vmap(forward_pass, in_axes=(0, 0, 0, None))
