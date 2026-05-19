from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # Network capacity
    N_max: int = 64
    n_in:  int = 2
    n_out: int = 2

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

    # Evolution
    population_size:  int = 1000
    tournament_size:  int = 4

    def tau_range(self, neuron_type: int) -> Tuple[float, float]:
        return (self.tau_e_range, self.tau_fsi_range, self.tau_sii_range)[neuron_type]
