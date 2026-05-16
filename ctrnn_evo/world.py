"""
2D foraging world — Milestone 3.

To implement:
  - World             : dataclass holding agent states, food hotspots, energy levels
  - step_world        : advance world by one dt_world tick given agent actions
  - reset_world       : initialise a fresh episode
  - sensor_readout    : map world state around agent to input_vec [n_in]
  - Validation targets:
      * gradient-following hand-coded controller survives
      * random-walk controller reliably starves
      * food hotspots drift at the intended rate
      * energy economics balance (metabolism, eating, movement costs)
"""
