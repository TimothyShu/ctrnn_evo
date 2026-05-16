from __future__ import annotations

import jax.numpy as jnp

from .config import Config
from .genome import Genome


def connection_cost(genome: Genome) -> float:
    """
    Sum of Euclidean wire lengths over all structurally present, active edges.

    Generalises the edge-count cost of Clune et al. (2013) to a spatially
    embedded substrate where evolved neuron positions determine wire length.
    """
    diff = genome.position[:, None, :] - genome.position[None, :, :]  # [N, N, 2]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))                       # [N, N]
    return jnp.sum(
        dist
        * genome.edge_mask
        * genome.active_mask[:, None]
        * genome.active_mask[None, :]
    )


def adjusted_fitness(
    f_raw: float,
    genome: Genome,
    c_act: float,
    cfg: Config,
) -> float:
    """
    Apply connection and activation cost penalties to raw task fitness.

        f = f_raw - lambda_conn * C_conn - lambda_act * C_act

    Setting a coefficient to 0.0 disables that cost term entirely.
    """
    c_conn = connection_cost(genome)
    return f_raw - cfg.lambda_conn * c_conn - cfg.lambda_act * c_act
