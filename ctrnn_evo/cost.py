from __future__ import annotations

import jax.numpy as jnp

from .config import Config
from .genome import Genome


def edge_count_cost(genome: Genome) -> float:
    """
    Count of active edges between active neurons — no spatial weighting.

    Implements the pure connection-count penalty of Clune et al. (2013):
    every edge costs the same regardless of how long it is.
    """
    active_pair = genome.active_mask[:, None] * genome.active_mask[None, :]
    return jnp.sum(genome.edge_mask * active_pair).astype(jnp.float32)


def dist_cost(genome: Genome) -> float:
    """
    Sum of Euclidean wire lengths over all active edges — distance-weighted.

    Penalises long axons more than short ones, reflecting the metabolic
    reality that long-range connections are expensive to build and maintain.
    Generalises edge_count_cost to spatially embedded networks.
    """
    diff = genome.position[:, None, :] - genome.position[None, :, :]  # [N, N, 2]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))                       # [N, N]
    active_pair = genome.active_mask[:, None] * genome.active_mask[None, :]
    return jnp.sum(dist * genome.edge_mask * active_pair)


def adjusted_fitness(
    f_raw: float,
    genome: Genome,
    c_act: float,
    cfg: Config,
) -> float:
    """
    Apply all three cost penalties to raw task fitness.

        f = f_raw
            - lambda_edge * C_edge   (edge count)
            - lambda_dist * C_dist   (total wire length)
            - lambda_act  * C_act    (mean activation per tick)

    Setting a coefficient to 0.0 disables that term entirely.
    Any combination of the three can be active simultaneously.
    """
    c_edge = edge_count_cost(genome)
    c_dist = dist_cost(genome)
    return (
        f_raw
        - cfg.lambda_edge * c_edge
        - cfg.lambda_dist * c_dist
        - cfg.lambda_act  * c_act
    )
