from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # Network capacity
    N_max: int = 64
    n_out: int = 2

    # Food types — n_in is derived automatically as 2 * n_food_types
    n_food_types: int = 1

    # Integration
    dt:       float = 0.5   # neural timestep (ms); must satisfy dt <= tau_min
    dt_world: float = 10.0  # world timestep (ms)
    K:        int   = 20    # inner CTRNN ticks per world step

    # Type-specific tau ranges (ms) — E, FS-I, SI-I
    tau_e_range:   Tuple[float, float] = (10.0,  100.0)
    tau_fsi_range: Tuple[float, float] = (1.0,    15.0)
    tau_sii_range: Tuple[float, float] = (80.0,  500.0)

    # Initial edge density (fraction of possible edges active at init)
    init_edge_density: float = 0.15

    # Cost coefficients (0 = disabled)
    lambda_edge: float = 0.0   # penalises edge count regardless of length
    lambda_dist: float = 0.0   # penalises total wire length (distance-weighted)
    lambda_act:  float = 0.0   # penalises mean neural activation per tick

    # Penalty warm-up — ramp all λ from 0 to their full values over this many
    # generations, so early-generation networks are not pruned before any
    # foraging strategy has evolved.  0 = disabled (lambdas are constant).
    penalty_warmup_gens: int = 0

    # Evolution
    population_size:  int = 1000
    tournament_size:  int = 4

    # Fitness metric
    # "survival": steps_survived / episode_steps  → [0, 1]
    # "food":     cumulative raw food score / (episode_steps * n_food_types)
    #             → can exceed 1.0 for agents that actively forage near hotspots
    fitness_mode: str = "survival"

    # Position sensors — if True, normalised (x, y) in [0, 1] are appended to the
    # sensor vector, giving the agent proprioceptive awareness of its arena location.
    # Without these, agents starting far from any food hotspot receive a zero food
    # signal and have no gradient to follow — they run open-loop into walls.
    # Adds 2 to n_in.  Default False for backward compatibility.
    position_sensors: bool = False

    # Derived — set by __post_init__, not a constructor argument
    n_in: int = field(init=False)

    def __post_init__(self):
        self.n_in = 2 * self.n_food_types + (2 if self.position_sensors else 0)

    def tau_range(self, neuron_type: int) -> Tuple[float, float]:
        return (self.tau_e_range, self.tau_fsi_range, self.tau_sii_range)[neuron_type]
